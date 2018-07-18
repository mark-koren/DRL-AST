from mylab.rewards.ast_reward import ASTReward
import numpy as np
import pdb

class ExampleAVReward(ASTReward):
    def __init__(self, num_peds = 1, cov_x = 0.1, cov_y = 0.01, cov_sensor_noise = 0.1):
        self.c_num_peds = num_peds
        self.c_cov_x = cov_x
        self.c_cov_y = cov_y
        self.c_cov_sensor_noise = cov_sensor_noise
        super().__init__()

    def give_reward(self, action, info):
        peds = info["peds"]
        car = info["car"]
        is_goal = info["is_goal"]
        is_terminal = info["is_terminal"]
        dist = peds[:, 2:4] - car[2:4]

        # update reward and done bool

        if (is_goal):
            reward = 0
        elif (is_terminal):
            reward = -10000 - 1000 * np.min(np.linalg.norm(dist, axis=1))
        else:
            reward = -np.log(1 + self.mahalanobis_d(action))

        return reward
        # if (np.any(np.all(np.less_equal(abs(dist), self.c_min_dist), axis=1))):
        #     self._reward = 0
        # elif (np.all(np.less_equal(dist[:, 0], -3.0)) or self._step > 50):
        #     self._reward = -10000 - 1000 * np.min(np.linalg.norm(dist, axis=1))
        # else:
        #     self._reward = -np.log(1 + self.mahalanobis_d(self._action))

        # if np.any(dist) < -10.0 or self._step > 100:

        #     self._done = True

    def mahalanobis_d(self, action):
        # load G
        # bundle = G(self._state)
        # mean = G[0:2].T
        # cov = np.array([[G[2], 0],[0, G[3]])
        mean = np.zeros((6 * self.c_num_peds, 1))
        # mean = np.array([[self.c_mean_x],[self.c_mean_y]])
        # cov = np.array([[self.c_cov_x, 0], [0, self.c_cov_y]])
        cov = np.zeros((self.c_num_peds, 6))
        cov[:, 0:6] = np.array([self.c_cov_x, self.c_cov_y,
                                self.c_cov_sensor_noise, self.c_cov_sensor_noise,
                                self.c_cov_sensor_noise, self.c_cov_sensor_noise])
        big_cov = np.diagflat(cov)

        # inv_cov = np.linalg.inv(cov)
        # big_cov_diag = np.zeros((2*self.c_num_peds))
        # big_cov_diag[::2] = inv_cov[0,0]
        # big_cov_diag[1::2] = inv_cov[1, 1]
        # big_cov = np.diag(big_cov_diag)

        dif = np.copy(action)
        dif[::2] -= mean[0, 0]
        dif[1::2] -= mean[1, 0]
        dist = np.dot(np.dot(dif.T, np.linalg.inv(big_cov)), dif)
        # print(dist)
        return np.sqrt(dist)