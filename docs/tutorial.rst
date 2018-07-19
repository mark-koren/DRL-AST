Tutorial
******************

1. Introduction
===============

This tutorial is intended for readers to learn how to use this package with their own simulator.
Matery of the underlying theory would be helpful, but is not needed for installation. Please install 
package before proceeding.

1.1. About AST
-----------------
Adaptive Stress Testing is a way of finding flaws in an autonomous agent. For any non-trivial problem, 
searching the space of a stochastic simulation is intractable, and grid searches do not perform well.
By modeling the search as a Markov Decision Process, we can use reinforcement learning to find the
most probable failure. AST treats the simulator as a black box, and only needs access in a few specific
ways. To interface a simulator to the AST packages, a few things will be needed:

- A **Simulator** is a wraper that exposes the simulation software to this package. See the Simulator
section for details on Interactive vs. Non-Interactive Simulators
- A **Reward** function dictates the optimization goals of the algorithm. 
- A **Runner** collects all of the run options and starts the method.
- **Space** objects give information on the size and limits of a space. This will be used to
define the **Observation Space** and the **Action Space**


1.2. About this tutorial
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

2.1 Interactive Simulations
---------------------------

An Interactive Simulation is one in which control can be injected at each step during the actual simulation run. 
For example, if a simulation is run by creating a specification file, and no other control is possible, that 
simulation would not be interactive. A simulation must be interactive for simulation state to be accesable 
to the AST solver. Passing the simulation state to the solver may reduce the number of episodes needed to
converge to a solution. However, pausing the simulation at each step may introduce overhead which slows
the execution. Neither variant is inherently better, so use whatever is appropriate for your project.

2.2 Inheriting the Base Simulator
---------------------------------

Start by creating a file named ''example_av_simulator.py'' in the ''simulators'' folder. Create a class titled
''ExampleAVSimulator'', which inherits from ''Simulator''.

::

	#import base Simulator class
	from mylab.simulators.simulator import Simulator

	#Used for math and debugging
	import numpy as np
	import pdb

	#Define the class
	class ExampleAVSimulator(Simulator):

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
