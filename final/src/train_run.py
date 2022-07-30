import datetime
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

import util.misc as utils
from datasets import build_dataset
from models.dino.dino import build_dino
from training.engine import evaluate, train_one_epoch
from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import SLConfig
from util.utils import ModelEma, BestMetricHolder

HOME = Path(os.environ['HOME'])
PARAMS = SLConfig({
	"backbone": "swin_L_384_22k",
	"backbone_dir": str(HOME /".linx/backbones/swin/"),
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
	"dataset_path": str(HOME / ".linx/datasets/linx_data"),
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
	"focal_alpha": 0.25,
	"focal_gamma": 2.0,
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
	"output_dir": "../models/finetune-0730",
	"param_dict_type": "default",
	"pdetr3_bbox_embed_diff_each_layer": False,
	"pdetr3_refHW": -1,
	"pe_temperatureH": 20,
	"pe_temperatureW": 20,
	"position_embedding": "sine",
	"pretrain_model_path": str(HOME / ".linx/checkpoints/coco-dino-swin/checkpoint_best_regular.pth"),
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
	"decoder_sa_type": "sa",
	"matcher_type": "HungarianMatcher",
	"decoder_module_seq": ["sa", "ca", "ffn"],
	"nms_iou_threshold": -1,
	"dec_pred_bbox_embed_share": True,
	"dec_pred_class_embed_share": True,
	"dn_number": 100,
	"dn_label_noise_ratio": 0.5,
	"match_unstable_error": True,
	"use_detached_boxes_dec_out": False,
	"wo_class_error": False
})

# fix the seed for reproducibility
seed = PARAMS.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class ModelBuilder:

	def __init__(self, **kwargs):
		if kwargs:
			PARAMS.merge_from_dict(kwargs)  # overwrite default parameters
		model, criterion, postprocessors = build_dino(args=PARAMS)

		self.params = PARAMS
		self.model = model
		self.criterion = criterion
		self.postprocessors = postprocessors

	def train_model(self, **params_to_override) -> Dict:
		if params_to_override:
			self.params.merge_from_dict(params_to_override)

		args = cfg = self.params
		model = self.model
		criterion = self.criterion
		postprocessors = self.postprocessors

		# setup logger
		os.makedirs(args.output_dir, exist_ok=True)
		output_dir = Path(args.output_dir)
		log_file_path = output_dir / 'info.txt'
		logger = setup_logger(output=str(log_file_path), color=False, name="linx")
		logger.info(f"args: {args} \n")

		# save config
		save_cfg_path = output_dir / "config_cfg.py"
		cfg.dump(save_cfg_path)
		save_json_path = output_dir / "config_args_all.json"
		with open(save_json_path, 'w') as f:
			json.dump(cfg.config_dict, f, indent=2)
		logger.info(f"Full config saved to {save_json_path}")

		device = torch.device(args.device)

		# build model
		model.to(device)

		# parameter logs
		n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
		named_parameters = model.named_parameters()
		logger.info(f'number of params:{n_parameters}')
		logger.debug("params:\n" + json.dumps({n: p.numel() for n, p in named_parameters if p.requires_grad}, indent=2))

		param_dicts = get_param_dict(args, model)
		optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

		dataset_train = build_dataset(image_set='train', args=args)
		dataset_val = build_dataset(image_set='val', args=args)

		sampler_train = RandomSampler(dataset_train)

		data_loader_train = DataLoader(
			dataset_train,
			batch_size=args.batch_size,
			drop_last=True, collate_fn=utils.collate_fn, num_workers=args.num_workers
		)
		data_loader_val = DataLoader(
			dataset_val,
			batch_size=args.batch_size,
			drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers
		)

		# learning rate scheduler
		if args.multi_step_lr:
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

		# if args.dataset_file == "coco_panoptic":
		# 	# We also evaluate AP during panoptic training, on original coco DS
		# 	coco_val = datasets.coco.build("val", args)
		# 	base_ds = get_coco_api_from_dataset(coco_val)
		# else:
		# 	base_ds = get_coco_api_from_dataset(dataset_val)

		# if args.frozen_weights is not None:
		# 	checkpoint = torch.load(args.frozen_weights, map_location='cpu')
		# 	model.detr.load_state_dict(checkpoint['model'])
		#

		resume_pth = None
		if (output_dir / 'checkpoint.pth').exists():
			resume_pth = str(output_dir / 'checkpoint.pth')
			checkpoint = torch.load(resume_pth, map_location='cpu')
			model.load_state_dict(checkpoint['model'])

			if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
				optimizer.load_state_dict(checkpoint['optimizer'])
				lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
				args.start_epoch = checkpoint['epoch'] + 1

		if (not resume_pth) and args.pretrain_model_path:
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

			_load_output = model.load_state_dict(_tmp_st, strict=False)
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
												  wo_class_error=args.wo_class_error, args=args)
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
						'model': model.state_dict(),
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
				wo_class_error=args.wo_class_error, args=args, logger=(logger if args.save_log else None)
			)
			map_regular = test_stats['coco_eval_bbox'][0]
			_isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
			if _isbest:
				checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
				utils.save_on_master({
					'model': model.state_dict(),
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
					wo_class_error=args.wo_class_error, args=args, logger=(logger if args.save_log else None)
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
	builder = ModelBuilder()
	result = builder.train_model(device='cpu')
	print(result)
