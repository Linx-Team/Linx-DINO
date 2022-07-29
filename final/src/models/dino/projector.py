import torch.nn as nn


class Projector(nn.Module):
	""" Logistic regression or MLP classifier """

	def __init__(self, d_input, d_output, projection_type="mlp", dropout=0.2, d_hidden=512):
		super(Projector, self).__init__()
		if projection_type == "log_reg":
			projector = nn.Sequential(
				nn.Linear(d_input, d_output)
			)
		elif projection_type == "mlp":
			projector = nn.Sequential(
				nn.Linear(d_input, d_hidden),
				nn.Tanh(),
				nn.LayerNorm(d_hidden),
				nn.Dropout(dropout),
				nn.Linear(d_hidden, d_output),
			)
		elif projection_type == "fancy_mlp":  # What they did in Infersent.
			projector = nn.Sequential(
				nn.Linear(d_input, d_hidden),
				nn.Tanh(),
				nn.LayerNorm(d_hidden),
				nn.Dropout(dropout),
				nn.Linear(d_hidden, d_hidden),
				nn.Tanh(),
				nn.LayerNorm(d_hidden),
				nn.Dropout(p=dropout),
				nn.Linear(d_hidden, d_output),
			)
		elif projection_type == 'dense_relu':
			projector = nn.Sequential(
				nn.Linear(d_input, d_hidden),
				nn.ReLU(),
				nn.Dropout(dropout),
				nn.Linear(d_hidden, d_output),
			)
		elif projection_type == 'electra_head':
			projector = nn.Sequential(
				nn.Linear(d_input, d_hidden),
				nn.GELU(),
				nn.Dropout(dropout),
				nn.Linear(d_hidden, d_output),
			)
		elif projection_type == 'dense_norm':
			projector = nn.Sequential(
				nn.Linear(d_input, d_hidden),
				nn.ReLU(),
				nn.LayerNorm(d_hidden),
				nn.Dropout(dropout),
				nn.Linear(d_hidden, d_output),
			)
		elif projection_type == 'double_relu':
			projector = nn.Sequential(
				nn.Linear(d_input, d_hidden),
				nn.ReLU(),
				nn.Dropout(dropout),
				nn.Linear(d_hidden, d_hidden),
				nn.ReLU(),
				nn.Dropout(dropout),
				nn.Linear(d_hidden, d_output),
			)
		elif projection_type == 'two_sigma_1d':
			projector = nn.Sequential(
				nn.BatchNorm1d(num_features=d_input),
				nn.Linear(d_input, 32),
				nn.ReLU(),
				nn.Dropout(dropout),
				nn.Linear(32, 16),
				nn.ReLU(),
				nn.Dropout(dropout),
				nn.Linear(16, 8),
				nn.ReLU(),
				nn.Dropout(dropout),
				nn.Linear(8, d_output),
			)
		else:
			raise ValueError(f"{self.__class__.__name__} type {projection_type} not found")
		self._projector = projector

	def forward(self, seq_emb):
		logits = self._projector(seq_emb)
		return logits

	@property
	def weight(self):
		return self._projector.weight

	@property
	def bias(self):
		return self._projector[-1].bias
