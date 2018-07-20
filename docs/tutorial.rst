Tutorial
******************
.. _introduction:

1 Introduction
===============

This tutorial is intended for readers to learn how to use this package with their own simulator.
Matery of the underlying theory would be helpful, but is not needed for installation. Please install 
package before proceeding.

.. _about-ast:

1.1 About AST
-----------------
Adaptive Stress Testing is a way of finding flaws in an autonomous agent. For any non-trivial problem, 
searching the space of a stochastic simulation is intractable, and grid searches do not perform well.
By modeling the search as a Markov Decision Process, we can use reinforcement learning to find the
most probable failure. AST treats the simulator as a black box, and only needs access in a few specific
ways. To interface a simulator to the AST packages, a few things will be needed:

* A **Simulator** is a wraper that exposes the simulation software to this package. See the Simulator section for details on Interactive vs. Non-Interactive Simulators
* A **Reward** function dictates the optimization goals of the algorithm. 
* A **Runner** collects all of the run options and starts the method.
* **Space** objects give information on the size and limits of a space. This will be used to
define the **Observation Space** and the **Action Space**

.. _about-this-tutorial:

1.2 About this tutorial
------------------------

In this tutorial, we will create a simple ring road network, which in the
absence of autonomous vehicles, experience a phenomena known as "stop-and-go
waves". An autonomous vehicle in then included into the network and trained
to attenuate these waves. The remainder of the tutorial is organized as follows:

-  In Sections 2, 3, and 4, we create the primary classes needed to run
   a ring road experiment.
-  In Section 5, we run an experiment in the absence of autonomous
   vehicles, and witness the performance of the network.
-  In Section 6, we run the experiment with the inclusion of autonomous
   vehicles, and discuss the changes that take place once the
   reinforcement learning algorithm has converged.


.. _creating-a-simulator:

2 Creating a Simulator
======================

This sections explains how to create a wrapper that exposes your simulator to the AST package. The 
wrapper allows the AST solver to specify actions to control the stochasticity in the simulation. 
Examples of stochastic simulation elements would be an actor, like a pedestrian or a car, or noise
elements, like on the beams of a LIDAR sensor. The simulator must be able to reset on command, and 
detect if a goal state had been reached. The simulator state can be used, but is not neccessary. 
Interactive simulations are optional as well.

.. _interactive-simulations:

2.1 Interactive Simulations
---------------------------

An Interactive Simulation is one in which control can be injected at each step during the actual simulation run. 
For example, if a simulation is run by creating a specification file, and no other control is possible, that 
simulation would not be interactive. A simulation must be interactive for simulation state to be accesable 
to the AST solver. Passing the simulation state to the solver may reduce the number of episodes needed to
converge to a solution. However, pausing the simulation at each step may introduce overhead which slows
the execution. Neither variant is inherently better, so use whatever is appropriate for your project.

.. _inheriting-the-base-simulator:

2.2 Inheriting the Base Simulator
---------------------------------

Start by creating a file named ``example_av_simulator.py`` in the ``simulators`` folder. Create a class titled
``ExampleAVSimulator``, which inherits from ``Simulator``.

::

	#import base Simulator class
	from mylab.simulators.simulator import Simulator

	#Used for math and debugging
	import numpy as np
	import pdb

	#Define the class
	class ExampleAVSimulator(Simulator):

The base generator accepts one input:

* **max_path_length**: The horizon of the simulation, in number of timesteps

A child of the Simulator class is required to define the following five functions: ``simulate``, ``step``, ``reset``, ``get_reward_info``, and ``is_goal``. An optional ``log`` function may also be implemented. 

.. _initializing-the-example-simulator:

2.3 Initializing the Example Simulator
--------------------------------------

Our example simulator will control a modified version of the Intelligent Driver Model (IDM) as our SUT, while adding sensor noise and filtering it out with an alpha-beta tracker. Initial simulation conditions are needed here as well. Because of all thise, the Simulator accepts a number of inputs:

* **num\_peds**: The number of pedestrians in the scenario
* **dt**: The length of the time step, in seconds
* **alpha**: A hyperparameter controlling the alpha-beta tracker that filters noise from the sensors
* **beta**: A hyperparameter controlling the alpha-beta tracker that filters noise from the sensors
* **v\_des**: The desired speed of the SUT
* **t\_headway**: An IDM hyperparameter that controls the target seperation between the SUT and the agent it is following, measured in seconds
* **a\_max**: An IDM hyperparameter that controls the maximum acceleration of the SUT
* **s\_min**: An IDM hyperparameter that controls the minimum distance between the SUT and the agent it is following
* **d\_cmf**: An IDM hyperparameter that controls the maximum comfortable decceleration of the SUT (a soft maximum that is only violated to avoid crashes)
* **d\_max**: An IDM hyperparameter that controls the maximum decceleration of the SUT
* **min\_dist\_x**: Defines the length of the hitbox in the x direction
* **min\_dist\_y**: Defines the length of the hitbox in the y direction
* **car\_init\_x**: Specifies the initial x-position of the SUT
* **car\_init\_y**: Specifies the initial y-position of the SUT
* **action\_only**: A boolean value specifying whether the simulation state is unobserved, so only the previous action will be used as input to the policy. Only set to False if you have an interactive simulatior with an observable state, and you would like to pass that state as part of the input to the policy (see `section 2.1`_)
* **kwargs**: Any keyword arguement not listed here. In particular, ``max_path_length`` should be pased to the base Simulator as one of the **kwargs.

.. _section 2.1: interactive-simulations_

In addition, there are a number of member variables that need to be initialized. The code is below:
::
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

        #initialize the base Simulator
        super().__init__(**kwargs)

.. _the-simulate-function:

2.4 The ``simulate`` function:
------------------------------

The simulate function runs a simulation using previously generated actions from the policy to control the stochasticity. The simulate function accepts a list of actions and an intitial state. It should run the simulation, then return the timestep that the goal state was achieved, or a -1 if the horizon was reached first. In addition, this function should return any simulation info needed for post-analysis. To do this, first add the following code to the file to handle the simulation aspect:
:: 
    def sensors(self, car, peds, noise):

        measurements = peds + noise
        return measurements

    def tracker(self, observation_old, measurements):
        observation = np.zeros_like(observation_old)

        observation[:, 0:2] = observation_old[:, 0:2]
        observation[:, 2:4] = observation_old[:, 2:4] + self.c_dt * observation_old[:, 0:2]
        residuals = measurements[:, 2:4] - observation[:, 2:4]

        observation[:, 2:4] += self.c_alpha * residuals
        observation[:, 0:2] += self.c_beta / self.c_dt * residuals

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

These functions handle the backend simulation of the toy problem and the SUT. Now we implement the ``simulate`` function, checking to be sure that the horizon wasn't reached:
::
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
        # initialize the simulation
        path_length = 0
        self.reset(s_0)
        self._info  = []

        # Take simulation steps unbtil horizon is reached
        while path_length < self.c_max_path_length:
            #get the action from the list
            self._action = actions[path_length]

            # move the peds
            self.update_peds()

            # move the car
            self._car = self.move_car(self._car, self._car_accel)

            # take new measurements and noise them
            noise = self._action.reshape((self.c_num_peds,6))[:, 2:6]
            self._measurements = self.sensors(self._car, self._peds, noise)

            # filter out the noise with an alpha-beta tracker
            self._car_obs = self.tracker(self._car_obs, self._measurements)

            # select the SUT action for the next timestep
            self._car_accel[0] = self.update_car(self._car_obs, self._car[0])

            # grab simulation state, if interactive
            self.observe()

            # record step variables
            self.log()

            # check if a crash has occurred. If so return the timestep, otherwise continue
            if self.is_goal():
                return path_length, np.array(self._info)
            path_length = path_length + 1

        # horizon reached without crash, return -1
        self._is_terminal = True
        return -1, np.array(self._info)

.. _the-step-function:

2.5 The ``step`` function:
--------------------------

If a simulation is interactive, the ``step`` function should interact with it at each timestep. The functions takes as input the current action. If the action is interactive and the simulation state is being used, return the state. Otherwise, return ``None``. If the simulation is non-interactive, other per-step actions can still be put here if neccessary - this function is called at each timestep either way. However, there is nothing to do at each step in this case, so the function will just return ``None``.
::
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

.. _the-reset-function:

2.6 The ``reset`` function:
---------------------------

The reset function should return the simulation to a state where it can accept the next sequence of actions. In some cases this may mean explcitily reseting the simulation parameters, like SUT location or simulation time. It could also mean opening and initializing a new instance of the simulator (in which case the ``simulate`` function should close the current instance). Your implementation of the ``reset`` function may be something else entirely, this is highly dependent on how your simulator functions. The method takes the initial state as an input, and returns the state of the simulator after the reset actions are taken. If the simulation state is not accessable, just return the initial condition parameters that were passed in.
::
    def reset(self, s_0):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """

        # initialize variables
        self._info = []
        self._step = 0
        self._is_terminal = False
        self.init_conditions = s_0
        self._first_step = True

        # Get v_des if it is sampled from a range
        v_des = self.init_conditions[3*self.c_num_peds]

        # initialize SUT location
        car_init_x = self.init_conditions[3*self.c_num_peds + 1]
        self._car = np.array([v_des, 0.0, car_init_x, self.c_car_init_y])

        # zero out the first SUT acceleration
        self._car_accel = np.zeros((2))

        # initialize pedestrian locations and velocities
        pos = self.init_conditions[0:2*self.c_num_peds]
        self.x = pos[0:self.c_num_peds*2:2]
        self.y = pos[1:self.c_num_peds*2:2]
        v_start = self.init_conditions[2*self.c_num_peds:3*self.c_num_peds]
        self._peds[0:self.c_num_peds, 0] = np.zeros((self.c_num_peds))
        self._peds[0:self.c_num_peds, 1] = v_start
        self._peds[0:self.c_num_peds, 2] = self.x
        self._peds[0:self.c_num_peds, 3] = self.y

        # Calculate the relative position measurements
        self._measurements = self._peds - self._car
        self._env_obs = self._measurements
        self._car_obs = self._measurements

        # return the initial simulation state
        if self.action_only:
            return self.init_conditions
        else:
            self._car = np.array([self.c_v_des, 0.0, self.c_car_init_x, self.c_car_init_y])
            self._car_accel = np.zeros((2))
            self._peds[:, 0:4] = np.array([0.0, 1.0, -0.5, -4.0])
            self._measurements = self._peds - self._car
            self._env_obs = self._measurements
            self._car_obs = self._measurements
            return np.ndarray.flatten(self._measurements)

.. _the-get-reward-info-function:

2.7 The ``get_reward_info`` function:
-------------------------------------

It is likely that your reward function (see XXX) will need some information from the simulator. The reward function will be passed whatever information is returned from this function. For the example, the reward function augments the "no crash" case with the distance between the SUT and the nearest pedestrian. To do this, both the car and pedestrian locations are returned. In addition, boolean values indicating whether a crash has been found or if the horizon has been reached are returned.
::
    def get_reward_info(self):
        """
        returns any info needed by the reward function to calculate the current reward
        """

        return {"peds": self._peds,
                "car": self._car,
                "is_goal": self.is_goal(),
                "is_terminal": self._is_terminal}

.. _the-is-goal-function:

2.8 The ``is_goal`` function:
-----------------------------

This function returns a boolean value indicating if the current state is in the goal set. In the example, this is True if the pedestrian is hit by the car. Therefore this function checks for any pedestrians in the hitbox of the SUT.
::
    def is_goal(self):
        """
        returns whether the current state is in the goal set
        :return: boolean, true if current state is in goal set.
        """
        # calculate the relative distances between the pedestrians and the car
        dist = self._peds[:, 2:4] - self._car[2:4]

        # return True if any relative distance is within the SUT's hitbox
        if (np.any(np.all(np.less_equal(abs(dist), self.c_min_dist), axis=1))):
            return True

        return False

.. _the-log-function-optional:

2.9 The ``log`` function (Optional):
------------------------------------

The log function is a way to store variables from the simulator for later access. In the example, some simulation state information is appended to a list at every timestep.
::
    def log(self):
        # Create a cache of step specific variables for post-simulation analysis
        cache = np.hstack([0.0,  # Dummy, will be filled in with trial # during post processing in save_trials.py
                           self._step,
                           np.ndarray.flatten(self._car),
                           np.ndarray.flatten(self._peds),
                           np.ndarray.flatten(self._action),
                           0.0])
        self._info.append(cache)
        self._step += 1

3 Creating a Reward Function
============================

This section explains how to create a function that dictates the reward at each timestep of a simulation. AST formulates the problem of searching the space of possible variations of a stochastic simulation as an MDP so that modern-day reinforcement learning (RL) techniques can be used. When optimizing a policy using RL, the reward function is of the utmost importance, as it determines how the agent will learn. Changing the reward function to achieve the desired policy is known as reward shaping. 

3.1 Reward Shaping
------------------


**SPOILER ALERT**: This section uses a famous summercamp game as an example. If you are planning on attending a children's summercamp in the near future I highly reccomend you skip this section, lest you ruin the counselors attempts at having fun at your expense. You have been warned.

As an example of reinforcement learning, and the importance of the reward function, consider the famous childrens game "The Hat Game." Common at summer camps, the game usually starts with a counselor holding a hat in his hands, telling the kids he is about to teach them a new game. He will say "Ok, ready everyone....? I can play the hat game," proceed to do a bunch of random things with the hat, and then say "how about you?" He will then pass the hat to a camper, who repeats almost exactly everything the counselor does, but is told "no, you didn't play the hat game." Another counselor will take the hat, say the words, do something completly different with it, and the game is on. The trick is actually the word "OK" - so long as you say that magic word, you have played the hat game, even if you have no hat.

How does this relate to reward shaping? In this case, the children are the policy. They are taking stochastic actions, trying to learn how to play the hat game. The key to the game being fun is that the children are pretrained to pay attentian to meaningless words, and to mimic the hat motions. However, after enough trials (and it can take a long time), most of them will pick up the pattern and attention will shift to "OK." In the vanilla game, there are two rewards. "Yes, you played the hat game" can be considered positive, and "No, you didn't play the hat game" can be considered negative, or just zero. By changing this reward, we could make the game difficulty radically different. Imagine if 10 kids tried the game, and all they got was a binary response on if at least one of them played the game. This would be much harder to pick up on! This is an example of a sparse reward function, or one that only rarely gives rewards, such as at the end of a trajectory. On the other hand, what if the children recieved feedback after every single word or motion on if they had played the hat game during that trial yet. The game would be much easier! These are examples of how different reward functions can make achieving the same policy easier or harder. 

3.2 Inheriting the Base Reward Function
---------------------------------------

Start by creating a file named ``example_av_reward.py`` in the ``rewards`` folder. Create a class title ``ExampleAVReward`` which inherits from ``ASTReward``:
::
	# import base class
	from mylab.rewards.ast_reward import ASTReward

	# useful packages for math and debugging
	import numpy as np
	import pdb

	# Define the class, inherit from the base
	class ExampleAVReward(ASTReward):

The base class does not take an inputs, and there is only one required function - ``give_reward``.

3.3 Initializing the Example Reward Function
--------------------------------------------

The reward function will be calculating some rewards based on the probability of certain actions. We have assumed the means action is the 0 vector, but we still need to take the following inputs:

* **num\_peds**: The number of pedestrians in the scenario
* **cov\_x**: The covariance of the gaussian distribution used to model the x-acceleration of a pedestrian
* **cov\_y**: The covariance of the gaussian distribution used to model the y-acceleration of a pedestrian
* **cov\_sensor\_noise**: The covariance of the gaussian distribution used to model the noise on a sensor measurement in both the x and y directions (assumed equal)

The code is below:
::
    def __init__(self,
                 num_peds=1,
                 cov_x=0.1,
                 cov_y=0.01,
                 cov_sensor_noise=0.1):

        self.c_num_peds = num_peds
        self.c_cov_x = cov_x
        self.c_cov_y = cov_y
        self.c_cov_sensor_noise = cov_sensor_noise
        super().__init__()

3.4 The ``give_reward`` function
--------------------------------

The give reward function takes
Our example reward function is broken down into three cases, as specified in the paper. The three cases are as follows:

1. There is a 

4 Creating a Runner
===================

5 Creating the Spaces
=====================

5.1 The Action Space
--------------------

5.2 The Observation Space
-------------------------

6 Running the Example
=====================
