import logging
import os
from typing import Dict

import optuna
import torch
from train_run import LinxModelBuilder, ADDED_PARAMS

logger = logging.getLogger(__name__)
BASE_METRIC_KEY = 'all_best_res'
opt_proc_num = os.environ.get('OPT_PROC_NUM', 0)


def objective(trial: optuna.Trial, DEFAULT_CONFIGS: Dict, process_number=0):
	opt_params = {
		'gause_noise_p': trial.suggest_float('gause_noise_p', 0, 0.5, step=0.1),
		'brightness_limit': trial.suggest_float('brightness_limit', 0, 0.5, step=0.1),
		'contrast_limit': trial.suggest_float('contrast_limit', 0, 0.5, step=0.1),
		'gamma_limit_min': trial.suggest_float('gamma_limit_min', 80, 100, step=10),
		'gamma_limit_max': trial.suggest_float('gamma_limit_max', 100, 120, step=10),
		'random_gamma_p': trial.suggest_float('random_gamma_p', 0, 0.5, step=0.1),
		'quality_lower': trial.suggest_float('quality_lower', 65, 85, step=10),
		'image_compression_p': trial.suggest_float('image_compression_p', 0, 0.5, step=0.1),

		'epochs': 1, #poch limit
		'output_dir': DEFAULT_CONFIGS['output_dir'] + f'_{process_number}_opt_{trial.number}',
	}

	logger.info(f'trial{trial.number}  with this param : {opt_params}')
	# logger.info(opt_params)
	builder = LinxModelBuilder()
	metrics = builder.train_model(**ADDED_PARAMS, **opt_params)
	logger.info(f'{trial.number} done with {metrics}')
	score = metrics[BASE_METRIC_KEY]
	return round(score, 6)


if __name__ == '__main__':
	optimization = True
	PARAMS = {
		"output_dir": '../models/finetune-0731'
	}
	if optimization:
		study = optuna.create_study(direction='maximize', study_name='param_finder')
		process_number = 0
		study.optimize(lambda trial: objective(trial, PARAMS, process_number=process_number), n_trials=100)
		logger.info(f'best trial AP50 score : {study.best_trial.value}, which params : {study.best_trial.params}')
		fig = optuna.visualization.plot_optimization_history(study)
		fig.write_image(file=f'optimization_history_{process_number}_{opt_proc_num}.png', format='png')
		fig2 = optuna.visualization.plot_param_importances(study)
		fig2.write_image(file=f'param_importances_{process_number}_{opt_proc_num}.png', format='png')
