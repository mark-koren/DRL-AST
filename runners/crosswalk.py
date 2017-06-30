from rllab.rllab.algos.trpo import TRPO
from rllab.rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.rllab.baselines.zero_baseline import ZeroBaseline
from rllab.rllab.envs.normalized_env import normalize
from rllab.rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from mylab.crosswalk_env import CrosswalkEnv
from mylab.crosswalk_sensor_env import CrosswalkSensorEnv
import rllab.misc.logger as logger
import os.path as osp
import argparse
from save_trials import *

parser = argparse.ArgumentParser()
# Logger Params
parser.add_argument('--exp_name', type=str, default='crosswalk_exp')
parser.add_argument('--tabular_log_file', type=str, default='tab.txt')
parser.add_argument('--text_log_file', type=str, default='tex.txt')
parser.add_argument('--params_log_file', type=str, default='args.txt')
parser.add_argument('--snapshot_mode', type=str, default='all')
parser.add_argument('--log_tabular_only', type=bool, default=False)
parser.add_argument('--log_dir', type=str, default='./rl_logs/sensor3')
parser.add_argument('--args_data', type=str, default=None)

args = parser.parse_args()

log_dir = args.log_dir

tabular_log_file = osp.join(log_dir, args.tabular_log_file)
text_log_file = osp.join(log_dir, args.text_log_file)
params_log_file = osp.join(log_dir, args.params_log_file)

logger.log_parameters_lite(params_log_file, args)
logger.add_text_output(text_log_file)
logger.add_tabular_output(tabular_log_file)
prev_snapshot_dir = logger.get_snapshot_dir()
prev_mode = logger.get_snapshot_mode()
logger.set_snapshot_dir(log_dir)
logger.set_snapshot_mode(args.snapshot_mode)
logger.set_log_tabular_only(args.log_tabular_only)
logger.push_prefix("[%s] " % args.exp_name)

env = normalize(CrosswalkSensorEnv())
policy = GaussianMLPPolicy(env_spec=env.spec,
                           hidden_sizes=(512, 256, 128, 64, 32))
baseline = LinearFeatureBaseline(env_spec=env.spec)

iters = 500
algo = TRPO(
    env=env,
    policy=policy,
    baseline=LinearFeatureBaseline(env_spec=env.spec),
    batch_size=4000,
    step_size=0.1,
    n_itr=iters
)
algo.train()

header = 'trial, step, v, x_car, y_car, x_ped, y_ped, del_x, del_y, sensor_A, sensor_B, sensor_C, reward, v_new, x_car_new, y_car_new, x_ped_new, y_ped_new'
save_trials(iters, args.log_dir, header)