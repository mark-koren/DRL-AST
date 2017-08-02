from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Box
import numpy as np

import pdb

class CrosswalkSensorEnv(Env):
    def __init__(self, ego, num_peds, dt, alpha, beta, v_des, delta, t_headway,
                 a_max, s_min, d_cmf, d_max, min_dist_x, min_dist_y,
                 x_accel_low, y_accel_low, x_accel_high,y_accel_high,
                 x_boundary_low, y_boundary_low, x_boundary_high, y_boundary_high,
                 x_v_low, y_v_low, x_v_high, y_v_high,
                 mean_x, mean_y, cov_x, cov_y,
                car_init_x, car_init_y,
                 mean_sensor_noise, cov_sensor_noise):
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
        self.c_mean_x = mean_x
        self.c_mean_y = mean_y
        self.c_cov_x = cov_x
        self.c_cov_y = cov_y
        self.c_car_init_x = car_init_x
        self.c_car_init_y = car_init_y
        self.c_mean_sensor_noise = mean_sensor_noise
        self.c_cov_sensor_noise = cov_sensor_noise

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
        #Move the pedestrians
        self.update_peds()
        #Calculate the reward for this step
        self.give_reward()
        #Move the car from accel decided at last step
        self._car = self.move_car(self._car, self._car_accel)
        #Give the car the noisy measurements
        noise = action.reshape((self.c_num_peds,6))[:,2:6]
        self._measurements = self.sensors(self._car, self._peds, noise)
        #Use Alpha-Beta tracker to update car observation
        self._car_obs = self.tracker(self._car_obs, self._measurements)
        #Decide the accel for next step
        self.update_car(self._car_obs, self._car[0])
        #Give the obs for the ped for next step
        self.observe()
        #Update instance attributes
        self.log()

        return Step(observation=np.ndarray.flatten(self._env_obs),
                    reward=self._reward,
                    done=self._done,
                    info={'cache':self._info})

    def mahalanobis_d(self, action):
        #TODO get mean and covariance from G
        #load G
        #bundle = G(self._state)
        #mean = G[0:2].T
        #cov = np.array([[G[2], 0],[0, G[3]])
        mean = np.zeros((6*self.c_num_peds,1))
        # mean = np.array([[self.c_mean_x],[self.c_mean_y]])
        # cov = np.array([[self.c_cov_x, 0], [0, self.c_cov_y]])
        cov = np.zeros((self.c_num_peds, 6))
        cov[:,0:6] = np.array([self.c_cov_x, self.c_cov_y,
                               self.c_cov_sensor_noise, self.c_cov_sensor_noise,
                               self.c_cov_sensor_noise, self.c_cov_sensor_noise])
        big_cov = np.diagflat(cov)

        # inv_cov = np.linalg.inv(cov)
        # big_cov_diag = np.zeros((2*self.c_num_peds))
        # big_cov_diag[::2] = inv_cov[0,0]
        # big_cov_diag[1::2] = inv_cov[1, 1]
        # big_cov = np.diag(big_cov_diag)

        dif = np.copy(action)
        dif[::2] -= mean[0,0]
        dif[1::2] -= mean[1, 0]
        dist = np.dot(np.dot(dif.T, big_cov), dif)

        return np.sqrt(dist)


    def reset(self):
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

        self._car = np.array([self.c_v_des, 0.0, self.c_car_init_x, self.c_car_init_y])
        self._car_accel = np.zeros((2))
        self._peds[:,0:4] = np.array([0.0,1.0,0.0,-3.0])

        dist = self._peds[:, 2:4] - self._car[2:4]

        self._measurements = self._peds - self._car
        self._env_obs = self._measurements
        self._car_obs = self._measurements
        return np.ndarray.flatten(self._measurements)

    @property
    def action_space(self):
        """
        Returns a Space object
        """
        low = np.array([self.c_x_accel_low,self.c_y_accel_low, 0.0, 0.0, 0.0, 0.0])
        high = np.array([self.c_x_accel_high, self.c_y_accel_high, 1.0, 1.0, 1.0, 1.0])

        for i in range(1, self.c_num_peds):
            low = np.hstack((low, np.array([self.c_x_accel_low,self.c_y_accel_low, 0.0, 0.0, 0.0, 0.0])))
            high = np.hstack((high, np.array([self.c_x_accel_high,self.c_y_accel_high, 1.0, 1.0, 1.0, 1.0])))

        return Box(low=low, high=high)

    @property
    def observation_space(self):
        """
        Returns a Space object
        """

        low = np.array([self.c_x_v_low, self.c_y_v_low, self.c_x_boundary_low, self.c_y_boundary_low])
        high = np.array([self.c_x_v_high, self.c_y_v_high, self.c_x_boundary_high, self.c_y_boundary_high])

        for i in range(1, self.c_num_peds):
            low = np.hstack((low, np.array([self.c_x_v_low, self.c_y_v_low, self.c_x_boundary_low, self.c_y_boundary_low])))
            high = np.hstack((high, np.array([self.c_x_v_high, self.c_y_v_high, self.c_x_boundary_high, self.c_y_boundary_high])))

        return Box(low=low, high=high)

    def render(self):
        print(':(')

    def get_cache_list(self):
        return self._info

    def sensors(self, car, peds, noise):

        measurements = peds - car + noise
        # if np.any(np.isnan(measurements)):
        #     pdb.set_trace()

        return measurements

    def tracker(self, observation_old, measurements):
        observation = np.zeros_like(observation_old)
        # print('Observation(k-1): ', observation_old)
        # print('Measurements: ', measurements)

        observation[:, 0:2] = observation_old[:, 0:2]
        observation[:, 2:4] = observation_old[:, 2:4] + self.c_dt * observation_old[:, 0:2]
        # print('Expected(k): ', observation)
        residuals = measurements[:, 2:4] - observation[:, 2:4]

        observation[:,0:2] += self.c_alpha * residuals
        observation[:, 2:4] += self.c_beta / self.c_dt * residuals
        # print('Residuals: ', residuals)
        # print('Observation: ', observation)
        # pdb.set_trace()


        return observation

    def update_car(self, obs, v_car):

        mins = np.argmin(obs, axis=0)

        v_oth = obs[mins[3], 0]
        s_headway = obs[mins[3], 2]

        del_v = v_oth - v_car
        s_des = self.c_s_min + v_car * self.c_t_headway - v_car * del_v / (2 * np.sqrt(self.c_a_max * self.c_d_cmf))
        if self.c_v_des > 0.0:
            v_ratio = v_car / self.c_v_des
        else:
            v_ratio = 1.0

        a = self.c_a_max * (1.0 - v_ratio**self.c_delta - (s_des/s_headway)**2)
        if np.isnan(a):
            pdb.set_trace()

        return np.clip(a, -self.c_d_max, self.c_a_max)

    def move_car(self, car, accel):
        car[2:4] += self.c_dt * car[0:2]
        car[0:2] += self.c_dt * accel
        return car

    def update_peds(self):
        #Update ped state from actions
        action = self._action.reshape((self.c_num_peds, 6))[:, 0:2]

        mod_a = np.hstack((action,
                           self._peds[:, 0:2] + 0.5 * self.c_dt * action))
        if np.any(np.isnan(mod_a)):
            pdb.set_trace()

        self._peds += self.c_dt * mod_a
        if np.any(np.isnan(self._peds)):
            pdb.set_trace()

    def give_reward(self):
        # pdb.set_trace()
        dist = self._peds[:, 2:4] - self._car[2:4]


        #update reward and done bool
        self._done = True
        # pdb.set_trace()
        if(np.any(np.all(np.less_equal(abs(dist), self.c_min_dist), axis=1))):
            self._reward = 0
        elif(np.all(dist[:,0]) < -3.0 or self._step > 50):
            self._reward = -10000 - 1000 * np.min(np.linalg.norm(dist, axis=1))
        else:
            self._done = False
            self._reward = -np.log(1 + self.mahalanobis_d(self._action))

        if np.any(dist < -10.0) or self._step > 100:
            self._done = True

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
        cache = np.hstack([0.0, #Dummy, will be filled in with trial # during post processing in save_trials.py
                           self._step,
                           np.ndarray.flatten(self._car),
                           np.ndarray.flatten(self._peds),
                           np.ndarray.flatten(self._action),
                           self._reward])
        # pdb.set_trace()
        self._info = cache
        self._step += 1
        if np.isnan(self._reward):
            pdb.set_trace()

