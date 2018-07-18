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


