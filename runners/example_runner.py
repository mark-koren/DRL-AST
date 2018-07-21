from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from sandbox.rocky.tf.policies.gaussian_gru_policy import GaussianGRUPolicy
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.normalized_env import normalize
from rllab.sampler import parallel_sampler
from rllab.sampler.stateful_pool import singleton_pool
import rllab.misc.logger as logger


import os.path as osp
import argparse
from save_trials import *
import pdb
import tensorflow as tf

from mylab.simulators.example_av_simulator import ExampleAVSimulator
from mylab.rewards.example_av_reward import ExampleAVReward
from mylab.spaces.example_av_spaces import ExampleAVSpaces

from mylab.envs.ast_env import ASTEnv
from mylab.ast_vectorized_sampler import ASTVectorizedSampler

#Logger Params
parser = argparse.ArgumentParser()

parser.add_argument('--exp_name', type=str, default='crosswalk_exp')
parser.add_argument('--tabular_log_file', type=str, default='tab.txt')
parser.add_argument('--text_log_file', type=str, default='tex.txt')
parser.add_argument('--params_log_file', type=str, default='args.txt')
parser.add_argument('--snapshot_mode', type=str, default="gap")
parser.add_argument('--snapshot_gap', type=int, default=10)
parser.add_argument('--log_tabular_only', type=bool, default=False)
parser.add_argument('--log_dir', type=str, default='.')
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
logger.set_snapshot_gap(args.snapshot_gap)
logger.set_log_tabular_only(args.log_tabular_only)
logger.push_prefix("[%s] " % args.exp_name)

sim = ExampleAVSimulator()

reward_function = ExampleAVReward()

spaces = ExampleAVSpaces()

env = TfEnv(normalize(ASTEnv(action_only=args.action_only,
                             sample_init_state=False,
                             s_0=[-0.5, -4.0, 1.0, 11.17, -35.0],
                             simulator=sim,
                             reward_function=reward_function,
                             spaces=spaces
                             )))


policy = GaussianLSTMPolicy(name='lstm_policy',
                            env_spec=env.spec,
                            hidden_dim=args.hidden_dim,
                            use_peepholes=args.use_peepholes)

baseline = LinearFeatureBaseline(env_spec=env.spec)


optimizer = ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))


sampler_cls = ASTVectorizedSampler

algo = TRPO(
    env=env,
    policy=policy,
    baseline=LinearFeatureBaseline(env_spec=env.spec),
    batch_size=args.batch_size,
    step_size=args.step_size,
    n_itr=args.iters,
    store_paths=True,
    optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)),
    max_path_length=50,
    sampler_cls=sampler_cls,
    sampler_args={"sim": sim,
                  "reward_function": reward_function})

saver = tf.train.Saver(save_relative_paths=True)
with tf.Session() as sess:
    algo.train(sess=sess)

    header = 'trial, step, ' + 'v_x_car, v_y_car, x_car, y_car, '
    for i in range(0,args.num_peds):
        header += 'v_x_ped_' + str(i) + ','
        header += 'v_y_ped_' + str(i) + ','
        header += 'x_ped_' + str(i) + ','
        header += 'y_ped_' + str(i) + ','

    for i in range(0,args.num_peds):
        header += 'a_x_'  + str(i) + ','
        header += 'a_y_' + str(i) + ','
        header += 'noise_v_x_' + str(i) + ','
        header += 'noise_v_y_' + str(i) + ','
        header += 'noise_x_' + str(i) + ','
        header += 'noise_y_' + str(i) + ','

    header += 'reward'
    save_trials(args.iters, args.log_dir, header, sess, save_every_n=args.snapshot_gap)
    saver.save(sess, args.log_dir + '/' + args.exp_name)


