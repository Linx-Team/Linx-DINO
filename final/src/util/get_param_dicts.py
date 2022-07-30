import torch.nn as nn


def match_name_keywords(n: str, name_keywords: list):
	out = False
	for b in name_keywords:
		if b in n:
			out = True
			break
	return out


def get_param_dict(args, model: nn.Module):
	param_dict_type = getattr(args, 'param_dict_type', 'default')

	# by default
	if param_dict_type == 'default':
		param_dicts = [
			{
				"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
			},
			{
				"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
				"lr": args.lr_backbone,
			}
		]
		return param_dicts
	else:
		raise NotImplementedError()
