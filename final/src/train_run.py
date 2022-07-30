# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from training.engine import evaluate, train_one_epoch
from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import SLConfig
from util.utils import ModelEma, BestMetricHolder


def build_model_main(args):
	# we use register to maintain models from catdet6 on.
	from models.registry import MODULE_BUILD_FUNCS
	assert args.modelname in MODULE_BUILD_FUNCS._module_dict
	build_func = MODULE_BUILD_FUNCS.get(args.modelname)
	model, criterion, postprocessors = build_func(args)
	return model, criterion, postprocessors


HOME = Path(os.environ['HOME'])
ADDED_PARAMS = {
	"dataset_path": str(HOME / ".linx/datasets/linx_data"),
	"pretrain_model_path": str(HOME / ".linx/checkpoints/coco-dino-swin/checkpoint_best_regular.pth"),
	"backbone_dir": str(HOME / ".linx/backbones/swin/"),
	"device": 'cuda' if torch.cuda.is_available() else 'cpu'
}


class LinxModelBuilder(object):
	def __init__(self):
		print('model builder initiated')

	def train_model(self, config_file: str = 'config_args_all.json', **params_to_overwrite) -> Dict:

		# load cfg file and update the args
		print(f"Loading config file from {config_file}")
		args = cfg = SLConfig.fromfile(config_file)

		utils.init_distributed_mode(args)
		if params_to_overwrite:
			cfg.merge_from_dict({'config_file': config_file, **params_to_overwrite})

			if args.output_dir:
				Path(args.output_dir).mkdir(parents=True, exist_ok=True)

			if args.rank == 0:
				save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
				cfg.dump(save_cfg_path)
				save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
				with open(save_json_path, 'w') as f:
					json.dump(vars(args), f, indent=2)
			cfg_dict = cfg._cfg_dict.to_dict()
			args_vars = vars(args)
			for k, v in cfg_dict.items():
				if k not in args_vars:
					setattr(args, k, v)
				else:
					raise ValueError("Key {} can used by args only".format(k))

			# update some new args temporally
			if not getattr(args, 'use_ema', None):
				args.use_ema = False
			if not getattr(args, 'debug', None):
				args.debug = False

			# setup logger
			os.makedirs(args.output_dir, exist_ok=True)
			log_file = os.path.join(args.output_dir, 'info.txt')
			logger = setup_logger(output=log_file, distributed_rank=args.rank,color=False, name="linx")
			# logger.info("git:\n  {}\n".format(utils.get_sha()))
			# logger.info("Command: " + ' '.join(sys.argv))
			if args.rank == 0:
				save_json_path = os.path.join(args.output_dir, "config_args_all.json")
				with open(save_json_path, 'w') as f:
					json.dump(vars(args), f, indent=2)
				logger.info("Full config saved to {}".format(save_json_path))
			logger.info('world size: {}'.format(args.world_size))
			logger.info('rank: {}'.format(args.rank))
			logger.info('local_rank: {}'.format(args.local_rank))
			logger.info("args: " + str(args) + '\n')

			if args.frozen_weights is not None:
				assert args.masks, "Frozen training is meant for segmentation only"
			print(args)

			device = torch.device(args.device)

			# fix the seed for reproducibility
			seed = args.seed + utils.get_rank()
			torch.manual_seed(seed)
			np.random.seed(seed)
			random.seed(seed)

			# build model
			model, criterion, postprocessors = build_model_main(args)
			wo_class_error = False
			model.to(device)

			# ema
			if args.use_ema:
				ema_m = ModelEma(model, args.ema_decay)
			else:
				ema_m = None

			model_without_ddp = model
			if args.distributed:
				model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
																  find_unused_parameters=args.find_unused_params)
				model_without_ddp = model.module
			n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
			logger.info('number of params:' + str(n_parameters))
			logger.info(
				"params:\n" + json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad},
										 indent=2))

			param_dicts = get_param_dict(args, model_without_ddp)

			optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

			dataset_train = build_dataset(image_set='train', args=args)
			dataset_val = build_dataset(image_set='val', args=args)

			if args.distributed:
				sampler_train = DistributedSampler(dataset_train)
				sampler_val = DistributedSampler(dataset_val, shuffle=False)
			else:
				sampler_train = torch.utils.data.RandomSampler(dataset_train)
				sampler_val = torch.utils.data.SequentialSampler(dataset_val)

			batch_sampler_train = torch.utils.data.BatchSampler(
				sampler_train, args.batch_size, drop_last=True)

			data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
										   collate_fn=utils.collate_fn, num_workers=args.num_workers)
			data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
										 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

			if args.onecyclelr:
				lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
																   steps_per_epoch=len(data_loader_train),
																   epochs=args.epochs, pct_start=0.2)
			elif args.multi_step_lr:
				lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.001,
																	milestones=args.lr_drop_list)
			elif getattr(args, 'linear_scheduler_with_warmup', None):
				from training.scheduler import LinearSchedulerWithWarmup
				lr_warmup = args.linear_scheduler_with_warmup
				one_epoch_train_steps = math.ceil(len(dataset_train) / args.batch_size)
				lr_scheduler = LinearSchedulerWithWarmup(
					optimizer=optimizer,
					num_warmup_steps=int(lr_warmup * one_epoch_train_steps * args.epochs),
					num_training_steps=int(one_epoch_train_steps * args.epochs)
				)
			elif getattr(args, 'cosine_scheduler_with_warmup', None):
				from training.scheduler import CosineSchedulerWithWarmup
				lr_warmup = args.cosine_scheduler_with_warmup
				one_epoch_train_steps = math.ceil(len(dataset_train) / args.batch_size)
				lr_scheduler = CosineSchedulerWithWarmup(
					optimizer=optimizer,
					num_warmup_steps=int(lr_warmup * one_epoch_train_steps * args.epochs),
					num_training_steps=int(one_epoch_train_steps * args.epochs)
				)
			else:
				lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

			if args.dataset_file == "coco_panoptic":
				# We also evaluate AP during panoptic training, on original coco DS
				coco_val = datasets.coco.build("val", args)
				base_ds = get_coco_api_from_dataset(coco_val)
			else:
				base_ds = get_coco_api_from_dataset(dataset_val)

			if args.frozen_weights is not None:
				checkpoint = torch.load(args.frozen_weights, map_location='cpu')
				model_without_ddp.detr.load_state_dict(checkpoint['model'])

			output_dir = Path(args.output_dir)
			if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
				args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
			if args.resume:
				if args.resume.startswith('https'):
					checkpoint = torch.hub.load_state_dict_from_url(
						args.resume, map_location='cpu', check_hash=True)
				else:
					checkpoint = torch.load(args.resume, map_location='cpu')
				model_without_ddp.load_state_dict(checkpoint['model'])
				if args.use_ema:
					if 'ema_model' in checkpoint:
						ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
					else:
						del ema_m
						ema_m = ModelEma(model, args.ema_decay)

				if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
					optimizer.load_state_dict(checkpoint['optimizer'])
					lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
					args.start_epoch = checkpoint['epoch'] + 1

			if (not args.resume) and args.pretrain_model_path:
				checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
				from collections import OrderedDict
				_ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
				ignorelist = []

				def check_keep(keyname, ignorekeywordlist):
					for keyword in ignorekeywordlist:
						if keyword in keyname:
							ignorelist.append(keyname)
							return False
					return True

				logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
				_tmp_st = OrderedDict(
					{k: v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

				_load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
				logger.info(str(_load_output))

				if args.use_ema:
					if 'ema_model' in checkpoint:
						ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
					else:
						del ema_m
						ema_m = ModelEma(model, args.ema_decay)

			if args.eval:
				os.environ['EVAL_FLAG'] = 'TRUE'
				test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
													  data_loader_val, base_ds, device, args.output_dir,
													  wo_class_error=wo_class_error, args=args)
				if args.output_dir:
					utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

				log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}
				if args.output_dir and utils.is_main_process():
					with (output_dir / "log.txt").open("a") as f:
						f.write(json.dumps(log_stats) + "\n")

				return log_stats

			writer = None
			if utils.is_main_process():
				writer = SummaryWriter(str(output_dir / 'tensorboard'))

			print("Start training")
			start_time = time.time()
			best_map_holder = BestMetricHolder(use_ema=args.use_ema)
			last_stats = {}
			for epoch in range(args.start_epoch, args.epochs):
				epoch_start_time = time.time()
				if args.distributed:
					sampler_train.set_epoch(epoch)
				train_stats = train_one_epoch(
					model, criterion, data_loader_train, optimizer, device, epoch,
					args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args,
					logger=(logger if args.save_log else None), ema_m=ema_m)
				if args.output_dir:
					checkpoint_paths = [output_dir / 'checkpoint.pth']

				if not args.onecyclelr:
					lr_scheduler.step()
				if args.output_dir:
					checkpoint_paths = [output_dir / 'checkpoint.pth']
					# extra checkpoint before LR drop and every 100 epochs
					if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
						checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
					for checkpoint_path in checkpoint_paths:
						weights = {
							'model': model_without_ddp.state_dict(),
							'optimizer': optimizer.state_dict(),
							'lr_scheduler': lr_scheduler.state_dict(),
							'epoch': epoch,
							'args': args,
						}
						if args.use_ema:
							weights.update({
								'ema_model': ema_m.module.state_dict(),
							})
						utils.save_on_master(weights, checkpoint_path)

				# eval
				test_stats, coco_evaluator = evaluate(
					model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
					wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
				)
				# map_regular = test_stats['coco_eval_bbox'][0]
				map_regular = test_stats['coco_eval_bbox'][1] #AP50 기준
				_isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
				if _isbest:
					checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
					utils.save_on_master({
						'model': model_without_ddp.state_dict(),
						'optimizer': optimizer.state_dict(),
						'lr_scheduler': lr_scheduler.state_dict(),
						'epoch': epoch,
						'args': args,
					}, checkpoint_path)

				# write test status
				if utils.is_main_process():
					writer.add_scalar('test/AP', test_stats['coco_eval_bbox'][0], epoch)
					writer.add_scalar('test/AP50', test_stats['coco_eval_bbox'][1], epoch)
					writer.add_scalar('test/AP75', test_stats['coco_eval_bbox'][2], epoch)
					writer.add_scalar('test/APs', test_stats['coco_eval_bbox'][3], epoch)
					writer.add_scalar('test/APm', test_stats['coco_eval_bbox'][4], epoch)
					writer.add_scalar('test/APl', test_stats['coco_eval_bbox'][5], epoch)
					writer.add_scalar('test/class_error', test_stats['class_error'], epoch)
					writer.add_scalar('test/loss', test_stats['loss'], epoch)
					writer.add_scalar('test/loss_ce', test_stats['loss_ce'], epoch)
					writer.add_scalar('test/loss_bbox', test_stats['loss_bbox'], epoch)
					writer.add_scalar('test/loss_giou', test_stats['loss_giou'], epoch)
					for key, value in test_stats.items():
						if "corr" in key:
							writer.add_scalar('test/' + key, value, epoch)

				log_stats = {
					**{f'train_{k}': v for k, v in train_stats.items()},
					**{f'test_{k}': v for k, v in test_stats.items()},
				}

				# eval ema
				if args.use_ema:
					ema_test_stats, ema_coco_evaluator = evaluate(
						ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
						wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
					)
					log_stats.update({f'ema_test_{k}': v for k, v in ema_test_stats.items()})
					map_ema = ema_test_stats['coco_eval_bbox'][0]
					_isbest = best_map_holder.update(map_ema, epoch, is_ema=True)
					if _isbest:
						checkpoint_path = output_dir / 'checkpoint_best_ema.pth'
						utils.save_on_master({
							'model': ema_m.module.state_dict(),
							'optimizer': optimizer.state_dict(),
							'lr_scheduler': lr_scheduler.state_dict(),
							'epoch': epoch,
							'args': args,
						}, checkpoint_path)
				log_stats.update(best_map_holder.summary())

				ep_paras = {
					'epoch': epoch,
					'n_parameters': n_parameters
				}
				log_stats.update(ep_paras)
				try:
					log_stats.update({'now_time': str(datetime.datetime.now())})
				except:
					pass

				epoch_time = time.time() - epoch_start_time
				epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
				log_stats['epoch_time'] = epoch_time_str
				last_stats = log_stats

				logger.info(f'result:{json.dumps(log_stats)}')
				if args.output_dir and utils.is_main_process():
					with (output_dir / "log.txt").open("a") as f:
						f.write(json.dumps(log_stats) + "\n")

					# for evaluation logs
					if coco_evaluator is not None:
						(output_dir / 'eval').mkdir(exist_ok=True)
						if "bbox" in coco_evaluator.coco_eval:
							filenames = ['latest.pth']
							if epoch % 50 == 0:
								filenames.append(f'{epoch:03}.pth')
							for name in filenames:
								torch.save(coco_evaluator.coco_eval["bbox"].eval,
										   output_dir / "eval" / name)

			total_time = time.time() - start_time
			total_time_str = str(datetime.timedelta(seconds=int(total_time)))
			print('Training time {}'.format(total_time_str))

			# remove the copied files.
			copyfilelist = vars(args).get('copyfilelist')
			if copyfilelist and args.local_rank == 0:
				from datasets.data_util import remove
				for filename in copyfilelist:
					print("Removing: {}".format(filename))
					remove(filename)
			return last_stats


if __name__ == '__main__':
	# parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
	# args = parser.parse_args()
	# main(args)
	output_dir = "../models/finetune-0731"
	builder = LinxModelBuilder()
	builder.train_model(config_file='config_args_all.json', output_dir=output_dir, **ADDED_PARAMS)
