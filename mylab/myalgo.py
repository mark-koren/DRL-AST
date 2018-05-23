from rllab.misc import ext
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf
import pdb

class PPO(BatchPolopt):
    """
    Proximal Policy Optimization.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            step_size=0.01,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.step_size = step_size
        super(PPO, self).__init__(**kwargs)

    @overrides
    def init_opt(self):
        if isinstance(self., float):
            lr = constfn(lr)
        else:
            assert callable(lr)
        if isinstance(cliprange, float):
            cliprange = constfn(cliprange)
        else:
            assert callable(cliprange)
        total_timesteps = int(total_timesteps)

        nenvs = 1 #env.num_self.envs
        ob_space = self.env.observation_space
        ac_space = self.env.action_space
        nbatch = nenvs * nsteps
        nbatch_train = nbatch // nminibatches

        make_model = lambda: Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs,
                                   nbatch_train=nbatch_train,
                                   nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                                   max_grad_norm=max_grad_norm)
        if save_interval and logger.get_dir():
            import cloudpickle
            with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
                fh.write(cloudpickle.dumps(make_model))
        self.model = make_model()
        self.runner = Runner(env=self.env, model=self.model, nsteps=nsteps, gamma=self.discount, lam=self.gae_lambda)


    @overrides
    def optimize_policy(self, itr, samples_data):
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)
        logger.log("Computing loss before")
        loss_before = self.optimizer.loss(all_input_values)
        logger.log("Computing KL before")
        mean_kl_before = self.optimizer.constraint_val(all_input_values)
        logger.log("Optimizing")
        self.optimizer.optimize(all_input_values)
        logger.log("Computing KL after")
        mean_kl = self.optimizer.constraint_val(all_input_values)
        logger.log("Computing loss after")
        loss_after = self.optimizer.loss(all_input_values)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKLBefore', mean_kl_before)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        pdb.set_trace()
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
def constfn(val):
    def f(_):
        return val
    return f