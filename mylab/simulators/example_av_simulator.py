#import base Simulator class
from mylab.simulators.simulator import Simulator
#Used for math and debugging
import numpy as np
import pdb
#Define the class
class ExampleAVSimulator(Simulator):
    """
    Class template for a non-interactive simulator.
    """
    #Accept parameters for defining the behavior of the system under test[SUT]
    def __init__(self,
                 ego = None,
                 num_peds = 1,
                 dt = 0.1,
                 alpha = 0.85,
                 beta = 0.005,
                 v_des = 11.17,
                 delta = 4.0,
                 t_headway = 1.5,
                 a_max = 3.0,
                 s_min = 4.0,
                 d_cmf = 2.0,
                 d_max = 9.0,
                 min_dist_x = 2.5,
                 min_dist_y = 1.4,
                 car_init_x = 35.0,
                 car_init_y = 0.0,
                 action_only = True,
                 **kwargs):
        #Constant hyper-params -- set by user
        self.c_num_peds = num_peds
        self.c_dt = dt
        self.c_alpha = alpha
        self.c_beta = beta
        self.c_v_des = v_des
        self.c_delta = delta
        self.c_t_headway = t_headway
        self.c_a_max = a_max
        self.c_s_min = s_min
        self.c_d_cmf = d_cmf
        self.c_d_max = d_max
        self.c_min_dist = np.array([min_dist_x, min_dist_y])
        self.c_car_init_x = car_init_x
        self.c_car_init_y = car_init_y
        self.action_only = action_only

        #These are set by reset, not the user
        self._car = np.zeros((4))
        self._car_accel = np.zeros((2))
        self._peds = np.zeros((self.c_num_peds, 4))
        self._measurements = np.zeros((self.c_num_peds, 4))
        self._car_obs = np.zeros((self.c_num_peds, 4))
        self._env_obs = np.zeros((self.c_num_peds, 4))
        self._done = False
        self._reward = 0.0
        self._info = []
        self._step = 0
        self._action = None
        self._first_step = True
        self.directions = np.random.randint(2, size=self.c_num_peds) * 2 - 1
        self.y = np.random.rand(self.c_num_peds) * 14 - 5
        self.x = np.random.rand(self.c_num_peds) * 4 - 2
        self.low_start_bounds = [-1.0, -4.25, -1.0, 5.0, 0.0, -6.0, 0.0, 5.0]
        self.high_start_bounds = [0.0, -3.75, 0.0, 9.0, 1.0, -2.0, 1.0, 9.0]
        self.v_start = [1.0, -1.0, 1.0, -1.0]
        self._state = None

        super().__init__(**kwargs)



    def simulate(self, actions, s_0):
        """
        Run/finish the simulation
        Input
        -----
        action : A sequential list of actions taken by the simulation
        Outputs
        -------
        (terminal_index)
        terminal_index : The index of the action that resulted in a state in the goal set E. If no state is found
                        terminal_index should be returned as -1.

        """
        path_length = 0
        self.reset(s_0)
        self._info  = []
        while path_length < self.c_max_path_length:
            self._action = actions[path_length]
            self.update_peds()
            self._car = self.move_car(self._car, self._car_accel)
            noise = self._action.reshape((self.c_num_peds,6))[:, 2:6]
            self._measurements = self.sensors(self._car, self._peds, noise)
            self._car_obs = self.tracker(self._car_obs, self._measurements)
            self._car_accel[0] = self.update_car(self._car_obs, self._car[0])
            self.observe()
            self.log()
            if self.is_goal():
                return path_length, np.array(self._info)
            path_length = path_length + 1
        # self._is_terminal = True
        self._is_terminal = True
        return -1, np.array(self._info)

    def step(self, action):
        """
        Handle anything that needs to take place at each step, such as a simulation update or write to file
        Input
        -----
        action : action taken on the turn
        Outputs
        -------
        (terminal_index)
        terminal_index : The index of the action that resulted in a state in the goal set E. If no state is found
                        terminal_index should be returned as -1.

        """
        return None

    def get_reward_info(self):
        """
        returns any info needed by the reward function to calculate the current reward
        """

        return {"peds": self._peds,
                "car": self._car,
                "is_goal": self.is_goal(),
                "is_terminal": self._is_terminal}

    def is_goal(self):
        """
        returns whether the current state is in the goal set
        :return: boolean, true if current state is in goal set.
        """
        dist = self._peds[:, 2:4] - self._car[2:4]
        if (np.any(np.all(np.less_equal(abs(dist), self.c_min_dist), axis=1))):
            return True

        return False

    def reset(self, s_0):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """

        # self._car = np.zeros((4))
        # self._car_accel = np.zeros((2))
        # self._peds = np.zeros((self._num_peds, 4))
        # self._measurements = np.zeros()
        # self._car_obs = np.zeros()
        # self._env_obs = np.zeros()
        # self._done = False
        # self._reward = 0.0
        # self._info = []
        self._info = []
        self._step = 0
        self._is_terminal = False
        self.init_conditions = s_0
        # self.init_conditions[0] = 0.0
        # self.init_conditions[1] = -4.0
        # v_des = np.random.uniform(self.c_v_des*0.75, self.c_v_des*1.25)
        v_des = self.init_conditions[3*self.c_num_peds]
        # car_init_x = np.random.uniform(self.c_car_init_x*0.75, self.c_car_init_x*1.25)
        car_init_x = self.init_conditions[3*self.c_num_peds + 1]
        self._car = np.array([v_des, 0.0, car_init_x, self.c_car_init_y])
        self._car_accel = np.zeros((2))
        # pos = np.random.uniform(self.low_start_bounds, self.high_start_bounds)
        pos = self.init_conditions[0:2*self.c_num_peds]
        self.x = pos[0:self.c_num_peds*2:2]
        self.y = pos[1:self.c_num_peds*2:2]
        # for i in range(self.c_num_peds):
        #     self._peds[i,0:4] = np.array([0.0, self.v_start[i], self.x[i],self.y[i]])
        # pdb.set_trace()
        # v_start = np.random.uniform(
        #     np.array(self.v_start[0:self.c_num_peds])*0.0,
        #     np.array(self.v_start[0:self.c_num_peds])*2.0)
        v_start = self.init_conditions[2*self.c_num_peds:3*self.c_num_peds]
        self._peds[0:self.c_num_peds, 0] = np.zeros((self.c_num_peds))
        self._peds[0:self.c_num_peds, 1] = v_start
        self._peds[0:self.c_num_peds, 2] = self.x
        self._peds[0:self.c_num_peds, 3] = self.y
        # self._peds[1, 0:4] = np.array([0.0, 1.0, 0.5, -2.0])
        # self._peds[1, 0:4] = np.array([0.0, -1.0, 0.0, 5.0])
        # self._peds[1, 0:4] = np.array([0.0, 1.0, 0.5, -4.0])
        # self._peds[2, 0:4] = np.array([0.0, 1.0, -0.5, -4.0])
        # self._peds[:, 0] = 0.0
        # self._peds[:, 1] = self.directions
        # self._peds[:,2] = self.x
        # self._peds[:, 3] = self.y
        # dist = self._peds[:, 2:4] - self._car[2:4]

        self._measurements = self._peds - self._car
        self._env_obs = self._measurements
        self._car_obs = self._measurements
        self._first_step = True
        if self.action_only:
            # self.init_conditions = np.ndarray.flatten(np.array([self.x, self.y, v_start, v_des, car_init_x]))
            # pdb.set_trace()
            return self.init_conditions
        else:
            self._car = np.array([self.c_v_des, 0.0, self.c_car_init_x, self.c_car_init_y])
            self._car_accel = np.zeros((2))
            self._peds[:, 0:4] = np.array([0.0, 1.0, -0.5, -4.0])
            self._measurements = self._peds - self._car
            self._env_obs = self._measurements
            self._car_obs = self._measurements
            return np.ndarray.flatten(self._measurements)
        # return np.ndarray.flatten(np.zeros_like(self._measurements))



    def sensors(self, car, peds, noise):

        measurements = peds + noise
        # if np.any(np.isnan(measurements)):
        #     pdb.set_trace()
        # print('peds: ', peds)
        # print('car: ', car)
        # print('noise: ', noise)
        # print('Measurements: ', measurements)
        # pdb.set_trace()
        return measurements

    def tracker(self, observation_old, measurements):
        observation = np.zeros_like(observation_old)
        # print('Observation(k-1): ', observation_old)
        # print('Measurements: ', measurements)

        observation[:, 0:2] = observation_old[:, 0:2]
        observation[:, 2:4] = observation_old[:, 2:4] + self.c_dt * observation_old[:, 0:2]
        # print('Expected(k): ', observation)
        residuals = measurements[:, 2:4] - observation[:, 2:4]

        observation[:, 2:4] += self.c_alpha * residuals
        observation[:, 0:2] += self.c_beta / self.c_dt * residuals
        # print('Residuals: ', residuals)
        # print('Observation: ', observation)
        # pdb.set_trace()

        return observation

    def update_car(self, obs, v_car):

        cond = np.repeat(np.resize(np.logical_and(obs[:, 3] > -1.5, obs[:, 3] < 4.5), (self.c_num_peds, 1)), 4, axis=1)
        in_road = np.expand_dims(np.extract(cond, obs), axis=0)

        if in_road.size != 0:
            mins = np.argmin(in_road.reshape((-1, 4)), axis=0)
            v_oth = obs[mins[3], 0]
            s_headway = obs[mins[3], 2] - self._car[2]

            del_v = v_oth - v_car
            s_des = self.c_s_min + v_car * self.c_t_headway - v_car * del_v / (2 * np.sqrt(self.c_a_max * self.c_d_cmf))
            if self.c_v_des > 0.0:
                v_ratio = v_car / self.c_v_des
            else:
                v_ratio = 1.0

            a = self.c_a_max * (1.0 - v_ratio ** self.c_delta - (s_des / s_headway) ** 2)

        else:
            del_v = self.c_v_des - v_car
            a = del_v

        if np.isnan(a):
            pdb.set_trace()

        # print('accel: ', a)
        # pdb.set_trace()
        return np.clip(a, -self.c_d_max, self.c_a_max)

    def move_car(self, car, accel):
        car[2:4] += self.c_dt * car[0:2]
        car[0:2] += self.c_dt * accel
        return car

    def update_peds(self):
        # Update ped state from actions
        action = self._action.reshape((self.c_num_peds, 6))[:, 0:2]

        mod_a = np.hstack((action,
                           self._peds[:, 0:2] + 0.5 * self.c_dt * action))
        if np.any(np.isnan(mod_a)):
            pdb.set_trace()

        self._peds += self.c_dt * mod_a
        if np.any(np.isnan(self._peds)):
            pdb.set_trace()

    def observe(self):
        self._env_obs = self._peds - self._car

    def log(self):
        # cache = np.zeros(((2 + #itr/step place holder
        #                       4 + #car state
        #                       4*self._num_peds + #ped states
        #                       2*self._num_peds + #ped actions
        #                       1 ))) # reward
        # cache[1] = self._step
        # cache[2:2+self._state.shape[-1]] = self._state
        # cache[2+self._state.shape[-1]:2+self._state.shape[-1]+action.shape[-1]] = action
        # cache[-1] = reward
        cache = np.hstack([0.0,  # Dummy, will be filled in with trial # during post processing in save_trials.py
                           self._step,
                           np.ndarray.flatten(self._car),
                           np.ndarray.flatten(self._peds),
                           np.ndarray.flatten(self._action),
                           0.0])
        # # pdb.set_trace()
        self._info.append(cache)
        self._step += 1
        # if np.isnan(self._reward):
        #     pdb.set_trace()

