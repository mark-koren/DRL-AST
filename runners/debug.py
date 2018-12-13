from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.normalized_env import normalize
from rllab.sampler import parallel_sampler
from rllab.sampler.stateful_pool import singleton_pool
import rllab.misc.logger as logger

from mylab.crosswalk_env_2 import CrosswalkEnv
from mylab.crosswalk_sensor_env_2 import CrosswalkSensorEnv

import os.path as osp
import argparse
from save_trials import *
import pdb
import tensorflow as tf

parser = argparse.ArgumentParser()
#Algo Params
parser.add_argument('--iters', type=int, default=101)
parser.add_argument('--batch_size', type=int, default=4000)
parser.add_argument('--step_size', type=float, default=0.1)
parser.add_argument('--store_paths', type=bool, default=True)
# Logger Params
parser.add_argument('--exp_name', type=str, default='crosswalk_exp')
parser.add_argument('--tabular_log_file', type=str, default='tab.txt')
parser.add_argument('--text_log_file', type=str, default='tex.txt')
parser.add_argument('--params_log_file', type=str, default='args.txt')
parser.add_argument('--snapshot_mode', type=str, default="gap")
parser.add_argument('--snapshot_gap', type=int, default=100)
parser.add_argument('--log_tabular_only', type=bool, default=False)
parser.add_argument('--log_dir', type=str, default='../../../../scratch/mkoren/run1')
parser.add_argument('--args_data', type=str, default=None)
#Environement Params

parser.add_argument('--dt', type=float, default=0.1)
parser.add_argument('--num_peds', type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.85)
parser.add_argument('--beta', type=float, default=0.005)
parser.add_argument('--v_des', type=float, default=11.4)
parser.add_argument('--delta', type=float, default=4.0)
parser.add_argument('--t_headway', type=float, default=1.5)
parser.add_argument('--a_max', type=float, default=3.0)
parser.add_argument('--s_min', type=float, default=4.0)
parser.add_argument('--d_cmf', type=float, default=2.0)
parser.add_argument('--d_max', type=float, default=9.0)
parser.add_argument('--min_dist_x', type=float, default=2.5)
parser.add_argument('--min_dist_y', type=float, default=1.4)
parser.add_argument('--car_init_x', type=float, default=-35.0)
parser.add_argument('--car_init_y', type=float, default=0.0)
parser.add_argument('--x_accel_low', type=float, default=-1.0)
parser.add_argument('--y_accel_low', type=float, default=-1.0)
parser.add_argument('--x_accel_high', type=float, default=1.0)
parser.add_argument('--y_accel_high', type=float, default=1.0)
parser.add_argument('--x_boundary_low', type=float, default=-10.0)
parser.add_argument('--y_boundary_low', type=float, default=-10.0)
parser.add_argument('--x_boundary_high', type=float, default=10.0)
parser.add_argument('--y_boundary_high', type=float, default=10.0)
parser.add_argument('--x_v_low', type=float, default=-10.0)
parser.add_argument('--y_v_low', type=float, default=-10.0)
parser.add_argument('--x_v_high', type=float, default=10.0)
parser.add_argument('--y_v_high', type=float, default=10.0)
parser.add_argument('--mean_x', type=float, default=0.0)
parser.add_argument('--mean_y', type=float, default=0.0)
parser.add_argument('--cov_x', type=float, default=0.01)
parser.add_argument('--cov_y', type=float, default=0.1)


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
logger.set_snapshot_gap(args.snapshot_gap)
logger.set_log_tabular_only(args.log_tabular_only)
logger.push_prefix("[%s] " % args.exp_name)

env = CrosswalkSensorEnv(ego=None,
                                   num_peds=args.num_peds,
                                   dt=args.dt,
                                   alpha = args.alpha,
                                   beta = args.beta,
                                   v_des=args.v_des,
                                   delta=args.delta,
                                   t_headway=args.t_headway,
                                   a_max=args.a_max,
                                   s_min=args.s_min,
                                   d_cmf=args.d_cmf,
                                   d_max=args.d_max,
                                   min_dist_x=args.min_dist_x,
                                   min_dist_y=args.min_dist_y,
                                   x_accel_low=args.x_accel_low,
                                   y_accel_low=args.y_accel_low,
                                   x_accel_high=args.x_accel_high,
                                   y_accel_high=args.y_accel_high,
                                   x_boundary_low=args.x_boundary_low,
                                   y_boundary_low=args.y_boundary_low,
                                   x_boundary_high=args.x_boundary_high,
                                   y_boundary_high=args.y_boundary_high,
                                   x_v_low=args.x_v_low,
                                   y_v_low=args.y_v_low,
                                   x_v_high=args.x_v_high,
                                   y_v_high=args.y_v_high,
                                   mean_x=args.mean_x,
                                   mean_y=args.mean_y,
                                   cov_x=args.cov_x,
                                   cov_y=args.cov_y,
                                   car_init_x=args.car_init_x,
                                   car_init_y=args.car_init_y,
                                   mean_sensor_noise = 0.0,
                                   cov_sensor_noise = 0.1)

obs_old = np.array([0.0, 1.0, 0.5,-2.0]) - np.array([11.17, 0.0, 35.0, 0.0])
meas = obs_old + 0.1 * np.array([[1.0,1.0,1.0,1.0]])
# print(obs_old)
# print(meas)
print(env.tracker(obs_old.reshape((1,4)), meas))

action = np.array([-0.233951, -0.388198, 1.0607, 0.4915, 0.3592, 0.6163])
d = env.mahalanobis_d(action)
print(action)
print(d)
print(-np.log(1 + d))
# pdb.set_trace()

dist = 10.0
v_oth = 0.0
v_ego = 11.4
obs = np.array([[v_oth, 0.0, dist, 0.0]])
print(env.update_car(obs, v_ego))
