# mostly copied from huggingface transformer:
# https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/optimization.py
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer


class LinearSchedulerWithWarmup(LambdaLR):
	def __init__(self, optimizer: Optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
		"""
		Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
		a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
		Args:
			optimizer ([`~torch.optim.Optimizer`]):
				The optimizer for which to schedule the learning rate.
			num_warmup_steps (`int`):
				The number of steps for the warmup phase.
			num_training_steps (`int`):
				The total number of training steps.
			last_epoch (`int`, *optional*, defaults to -1):
				The index of the last epoch when resuming training.
		Return:
			torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
		"""
		self.optimizer = optimizer
		self.num_warmup_steps = num_warmup_steps
		self.num_training_steps = num_training_steps
		self.last_epoch = last_epoch

		super(LinearSchedulerWithWarmup, self).__init__(optimizer, lr_lambda=self.lr_lambda, last_epoch=last_epoch)

	def lr_lambda(self, current_step: int):
		num_warmup_steps = self.num_warmup_steps
		num_training_steps = self.num_training_steps

		if current_step < num_warmup_steps:
			return float(current_step) / float(max(1, num_warmup_steps))
		return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
