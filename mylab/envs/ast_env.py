from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Box
import numpy as np
from simulators.example_av_simulator import ExampleAVSimulator
from mylab.example_av_reward import ExampleAVReward
import pdb


class ASTEnv(Env):
    def __init__(self,
                 num_peds=1,
                 max_path_length = 50,
                 v_des=11.17,
                 x_accel_low=-1.0,
                 y_accel_low=-1.0,
                 x_accel_high=1.0,
                 y_accel_high=1.0,
                 x_boundary_low=-10.0,
                 y_boundary_low=-10.0,
                 x_boundary_high=10.0,
                 y_boundary_high=10.0,
                 x_v_low=-10.0,
                 y_v_low=-10.0,
                 x_v_high=10.0,
                 y_v_high=10.0,
                 car_init_x=35.0,
                 car_init_y=0.0,
                 action_only=True,
                 sample_init_state=False,
                 s_0=None,
                 simulator=None,
                 reward_function=None):
        # Constant hyper-params -- set by user
        self.c_num_peds = num_peds
        self.c_max_path_length = max_path_length
        self.c_v_des = v_des
        self.c_x_accel_low = x_accel_low
        self.c_y_accel_low = y_accel_low
        self.c_x_accel_high = x_accel_high
        self.c_y_accel_high = y_accel_high
        self.c_x_boundary_low = x_boundary_low
        self.c_y_boundary_low = y_boundary_low
        self.c_x_boundary_high = x_boundary_high
        self.c_y_boundary_high = y_boundary_high
        self.c_x_v_low = x_v_low
        self.c_y_v_low = y_v_low
        self.c_x_v_high = x_v_high
        self.c_y_v_high = y_v_high
        self.c_car_init_x = car_init_x
        self.c_car_init_y = car_init_y

        self.action_only = action_only

        # These are set by reset, not the user
        self._done = False
        self._reward = 0.0
        self._info = []
        self._step = 0
        self._action = None
        self._actions = []
        self._first_step = True
        self.low_start_bounds = [-1.0, -4.25, -1.0, 5.0, 0.0, -6.0, 0.0, 5.0]
        self.high_start_bounds = [0.0, -3.75, 0.0, 9.0, 1.0, -2.0, 1.0, 9.0]
        self.v_start = [1.0, -1.0, 1.0, -1.0]

        if s_0 is None:
            self._init_state = self.observation_space.sample()
        else:
            self._init_state = s_0
        self._sample_init_state = sample_init_state
        self.simulator = simulator
        if self.simulator is None:
            self.simulator = ExampleAVSimulator()
        self.reward_function = reward_function
        if self.reward_function is None:
            self.reward_function = ExampleAVReward()

        super().__init__()

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the environment
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """
        self._action = action
        self._actions.append(action)
        # Update simulation step
        obs = self.simulator.step(self._action)
        if obs is None:
            obs = self._init_state
        if self.simulator.is_goal():
            self._done = True
        # Calculate the reward for this step
        self._reward = self.reward_function.give_reward(self._action, self.simulator.get_reward_info())
        # Update instance attributes
        # self.log()
        # if self._step == self.c_max_path_length - 1:
        #     # pdb.set_trace()
        #     self.simulator.simulate(self._actions)
        self._step = self._step + 1

        return Step(observation=obs,
                    reward=self._reward,
                    done=self._done,
                    info={'cache': self._info})

    def simulate(self, actions):
        if self._sample_init_state:
            self._init_state = self.observation_space.sample()
        self.simulator.simulate(actions)

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self._actions = []
        if self._sample_init_state:
            self._init_state = self.observation_space.sample()

        return self.simulator.reset(self._init_state)

    @property
    def action_space(self):
        """
        Returns a Space object
        """
        low = np.array([self.c_x_accel_low, self.c_y_accel_low, 0.0, 0.0, 0.0, 0.0])
        high = np.array([self.c_x_accel_high, self.c_y_accel_high, 1.0, 1.0, 1.0, 1.0])

        for i in range(1, self.c_num_peds):
            low = np.hstack((low, np.array([self.c_x_accel_low, self.c_y_accel_low, 0.0, 0.0, 0.0, 0.0])))
            high = np.hstack((high, np.array([self.c_x_accel_high, self.c_y_accel_high, 1.0, 1.0, 1.0, 1.0])))

        return Box(low=low, high=high)

    @property
    def observation_space(self):
        """
        Returns a Space object
        """

        low = np.array([self.c_x_v_low, self.c_y_v_low, self.c_x_boundary_low, self.c_y_boundary_low])
        high = np.array([self.c_x_v_high, self.c_y_v_high, self.c_x_boundary_high, self.c_y_boundary_high])

        for i in range(1, self.c_num_peds):
            low = np.hstack(
                (low, np.array([self.c_x_v_low, self.c_y_v_low, self.c_x_boundary_low, self.c_y_boundary_low])))
            high = np.hstack(
                (high, np.array([self.c_x_v_high, self.c_y_v_high, self.c_x_boundary_high, self.c_y_boundary_high])))

        if self.action_only:
            low = self.low_start_bounds[:self.c_num_peds * 2]
            low = low + np.ndarray.tolist(0.0 * np.array(self.v_start))[:self.c_num_peds]
            low = low + [0.75 * self.c_v_des]
            low = low + [0.75 * self.c_car_init_x]
            high = self.high_start_bounds[:self.c_num_peds * 2]
            high = high + np.ndarray.tolist(2.0 * np.array(self.v_start))[:self.c_num_peds]
            high = high + [1.25 * self.c_v_des]
            high = high + [1.25 * self.c_car_init_x]

        # pdb.set_trace()
        return Box(low=np.array(low), high=np.array(high))

    def get_cache_list(self):
        return self._info

    def log(self):
        self.simulator.log()


