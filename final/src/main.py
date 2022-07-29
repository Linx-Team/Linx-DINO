# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import ModelEma, BestMetricHolder

PARAMS = SLConfig({
	"backbone": "swin_L_384_22k",
	"backbone_dir": "/home/ubuntu/.linx/backbones/swin/",
	"backbone_freeze_keywords": None,
	"batch_size": 2,
	"clip_max_norm": 0.1,
	"dabdetr_yolo_like_anchor_update": False,
	"dabdetr_deformable_encoder": False,
	"dabdetr_deformable_decoder": False,
	"data_aug_scales": [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
	"data_aug_max_size": 1333,
	"data_aug_scales2_resize": [400, 500, 600],
	"data_aug_scales2_crop": [384, 600],
	"data_aug_scale_overlap": None,
	"dataset_file": "linx",
	"dataset_path": "/home/ubuntu/.linx/datasets/linx_data",
	"ddetr_lr_param": False,
	"dilation": False,
	"dim_feedforward": 2048,
	"device": "cuda",
	"dn_bbox_coef": 1.0,
	"dn_box_noise_scale": 1.0,
	"dn_label_coef": 1.0,
	"dn_labelbook_size": 7,
	"dn_scalar": 100,
	"dropout": 0.0,
	"embed_init_tgt": True,
	"epochs": 50,
	"eval": False,
	"find_unused_params": False,
	"finetune_ignore": ["label_enc.weight", "class_embed"],
	"fix_size": False,
	"frozen_weights": None,
	"hidden_dim": 256,
	"lr": 1e-04,
	"lr_backbone": 1e-05,
	"lr_backbone_names": ["backbone.0"],
	"lr_drop": 33,
	"lr_drop_list": [33, 45],
	"lr_linear_proj_names": ["reference_points", "sampling_offsets"],
	"lr_linear_proj_mult": 0.1,
	"modelname": "dino",
	"multi_step_lr": False,
	"nheads": 8,
	"num_classes": 7,
	"num_queries": 900,
	"num_workers": 10,
	"output_dir": "logs/dino/finetune-0724",
	"param_dict_type": "default",
	"pdetr3_bbox_embed_diff_each_layer": False,
	"pdetr3_refHW": -1,
	"pe_temperatureH": 20,
	"pe_temperatureW": 20,
	"position_embedding": "sine",
	"pretrain_model_path": "/home/ubuntu/.linx/checkpoints/coco-dino-swin/checkpoint_best_regular.pth",
	"query_dim": 4,
	"remove_difficult": False,
	"resume": "",
	"return_interm_indices": [1, 2, 3],
	"save_checkpoint_interval": 1,
	"save_log": True,
	"save_results": True,
	"seed": 42,
	"start_epoch": 0,
	"strong_aug": False,
	"test": False,
	"two_stage_type": "standard",
	"two_stage_pat_embed": 0,
	"two_stage_add_query_num": 0,
	"two_stage_bbox_embed_share": False,
	"two_stage_class_embed_share": False,
	"two_stage_learn_wh": False,
	"two_stage_default_hw": 0.05,
	"two_stage_keep_all_tokens": False,
	"use_checkpoint": True,
	"use_ema": False,
	"use_dn": True,
	"world_size": 1,
	"weight_decay": 0.001,

	# shold not be configurable - let's hardcode
	"enc_layers": 6,
	"dec_layers": 6,
	"unic_layers": 0,
	"pre_norm": False,

	"num_patterns": 0,

	"random_refpoints_xy": False,
	"fix_refpoints_hw": -1,

	"use_deformable_box_attn": False,
	"box_attn_type": "roi_align",
	"dec_layer_number": None,
	"num_feature_levels": 4,
	"enc_n_points": 4,
	"dec_n_points": 4,
	"decoder_layer_noise": False,
	"dln_xy_noise": 0.2,
	"dln_hw_noise": 0.2,
	"add_channel_attention": False,
	"add_pos_value": False,
	"num_select": 300,
	"transformer_activation": "relu",
	"batch_norm_type": "FrozenBatchNorm2d",
	"masks": False,
	"aux_loss": True,
	"set_cost_class": 2.0,
	"set_cost_bbox": 5.0,
	"set_cost_giou": 2.0,
	"cls_loss_coef": 1.0,
	"mask_loss_coef": 1.0,
	"dice_loss_coef": 1.0,
	"bbox_loss_coef": 5.0,
	"giou_loss_coef": 2.0,
	"enc_loss_coef": 1.0,
	"interm_loss_coef": 1.0,
	"no_interm_box_loss": False,
	"focal_alpha": 0.25,
	"decoder_sa_type": "sa",
	"matcher_type": "HungarianMatcher",
	"decoder_module_seq": ["sa", "ca", "ffn"],
	"nms_iou_threshold": -1,
	"dec_pred_bbox_embed_share": True,
	"dec_pred_class_embed_share": True,
	"dn_number": 100,
	"dn_label_noise_ratio": 0.5,
	"match_unstable_error": True,
	"ema_decay": 0.9997,
	"ema_epoch": 0,
	"use_detached_boxes_dec_out": False,
})


def get_args_parser():
	parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
	parser.add_argument('--config_file', '-c', type=str, required=True)
	parser.add_argument('--options',
						nargs='+',
						action=DictAction,
						help='override some settings in the used config, the key-value pair '
							 'in xxx=yyy format will be merged into config file.')

	# dataset parameters
	parser.add_argument('--dataset_file', default='coco')
	parser.add_argument('--coco_path', type=str, default='~/.linx/datasets/coco_2017')
	parser.add_argument('--coco_panoptic_path', type=str)
	parser.add_argument('--remove_difficult', action='store_true')
	parser.add_argument('--fix_size', action='store_true')

	# training parameters
	parser.add_argument('--output_dir', default='logs/dino/R50',
						help='path where to save, empty for no saving')
	parser.add_argument('--note', default='',
						help='add some notes to the experiment')
	parser.add_argument('--device', default='cuda',
						help='device to use for training / testing')
	parser.add_argument('--seed', default=42, type=int)
	parser.add_argument('--resume', default='', help='resume from checkpoint')
	parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
	parser.add_argument('--finetune_ignore', type=str, nargs='+')
	parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
						help='start epoch')
	parser.add_argument('--strong_aug', action='store_true')
	parser.add_argument('--eval', action='store_true')
	parser.add_argument('--num_workers', default=10, type=int)
	parser.add_argument('--test', action='store_true')
	parser.add_argument('--debug', action='store_true')
	parser.add_argument('--find_unused_params', action='store_true')

	parser.add_argument('--save_results', action='store_true')
	parser.add_argument('--save_log', action='store_true')

	# distributed training parameters
	parser.add_argument('--world_size', default=1, type=int,
						help='number of distributed processes')
	parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
	parser.add_argument('--rank', default=0, type=int,
						help='number of distributed processes')
	parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
	parser.add_argument('--amp', action='store_true',
						help="Train with mixed precision")

	return parser


def build_model_main(args):
	# we use register to maintain models from catdet6 on.
	from models.registry import MODULE_BUILD_FUNCS
	assert args.modelname in MODULE_BUILD_FUNCS._module_dict
	build_func = MODULE_BUILD_FUNCS.get(args.modelname)
	model, criterion, postprocessors = build_func(args)
	return model, criterion, postprocessors


def main(args):
	utils.init_distributed_mode(args)
	# load cfg file and update the args
	print("Loading config file from {}".format(args.config_file))
	time.sleep(args.rank * 0.02)
	cfg = SLConfig.fromfile(args.config_file)
	if args.options is not None:
		cfg.merge_from_dict(args.options)
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
	logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False,
						  name="detr")
	logger.info("git:\n  {}\n".format(utils.get_sha()))
	logger.info("Command: " + ' '.join(sys.argv))
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
		"params:\n" + json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

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
														   steps_per_epoch=len(data_loader_train), epochs=args.epochs,
														   pct_start=0.2)
	elif args.multi_step_lr:
		lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.001, milestones=args.lr_drop_list)
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

		return

	writer = None
	if utils.is_main_process():
		writer = SummaryWriter(str(output_dir / 'tensorboard'))

	print("Start training")
	start_time = time.time()
	best_map_holder = BestMetricHolder(use_ema=args.use_ema)
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
		map_regular = test_stats['coco_eval_bbox'][0]
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


if __name__ == '__main__':
	parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
	args = parser.parse_args()
	if args.output_dir:
		Path(args.output_dir).mkdir(parents=True, exist_ok=True)
	main(args)
