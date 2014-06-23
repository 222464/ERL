#README

## ERL - Evolved Reinforcement Learner

### SETUP:

To get started with ERL, you need your favorite compiler and CMake (http://www.cmake.org/cmake/resources/software.html).

Set CMake's source code directory to the ERL directory (the one that contains the /source folder as well as a CMakeLists.txt).

Set CMake's build directory to the /build/ directory (inside /ERL/).

Then press configure, and choose your compiler.

If the configuration did not result in errors, you can press configure again, and then hit generate. This will generate files necessary for your compiler in the /build/ directory.

You should then be able to compile and execute the program. If you are using Visual Studio, you may have to set your startup project to the ERL project.

##ERL – Evolved Reinforcement Learner – Overview

The ERL (Evolved Reinforcement Learner) project seeks to produce a reinforcement learning agent that is both general and fast. It consists of a continuous neural field whose dynamics are evolved using genetic programming techniques. The evolved continuous neural field is executed as an OpenCL kernel for maximum performance. The resulting agent will be tested on several experiments to evaluate its performance (fitness for the genetic algorithm).

The ERL project consists of the following components:
* A Continuous Neural Field (CNF)
* A CNF visualization system (OpenGL)
* An update rule generation/reproduction system
* An evolutionary algorithm
* A update rule to OpenCL kernel compiler
* A set of experiments
* A training system (running experiments and epochs in the genetic algorithm)

###The Continuous Neural Field

A continuous neural field is an abstraction of neural networks that can represent populations of  neurons as individual nodes. In this project, it may end up not doing this, since it is evolved and will do whatever is best, but it is given the option.

Continuous neural fields are a mathematical framework that treat neural networks as a continuum. Since we cannot represent continuous data, we must approximate the field with a discrete grid. The gird contains nodes with connections to other nodes within a neighborhood radius, modulated by a weight kernel (in the mathematical sense, not in the OpenCL sense).

The field can take on any dimension, but we will start with 2D since it is easiest to visualize. We may also choose to create the system such that any number of dimensions can be chosen in the future.

Individual nodes will be fed data on their positioning, at least during creation, so that different areas of the field can be specialized for different tasks. We may do this by having nodes operate on certain parameters defined by another function, the node parameter generator, based on the node position as well as possibly the other nodes in the vicinity.

###The CNF Visualization System

In order to observe the proper propagation of signals throughout the field, we need a visualization system. Fortunately, OpenGL and OpenCL have pretty decent interoperability. We should therefore be able to display the state of the nodes representing the field in a graphical manner. What data about the nodes is displayed is not yet certain, as the evolutionary system may not use data as we expect it to. Either the use of multiple colors or a node value toggle would be useful for visualizing the multiple values a node produces/holds.

###Update Rule Generation and Reproduction

ERL should work with a standard genetic programming approach to update rule generation. The terminals to the system would be the its inputs as well as its hidden state. For a node, for example, there may be input from connections, as well as a hidden state from the node itself as a form of memory. These parameters, along with the update rules, are all evolved.
	
The rules should be able to be crossed over and mutated. Mutation should not only perturb existing values, but also be able to expand the function call hierarchy of the rule.

###Evolutionary Algorithm

A standard evolutionary algorithm, preferably with speciation to preserve population variety and escape local maxima. The evolutionary algorithm should use roulette wheel selection with speciation taken into account.

###Update Rule to OpenCL Kernel Compiler

We require a way to transfer the evolved field dynamics into the OpenCL kernel language. We then need to compile the new kernel for every individual of the population prior to execution.

For this to work, we need to translate the evolved functions into the primitive C-like OpenCL kernel language.

###A Set of Experiments

We must train the agent to achieve some goal. What the fitness function is does not really matter, we must train it to be able to solve a variety of tasks given the environment and reinforcement signals. Therefore, a large variety of experiments (to evaluate the agent on in preperation for reproduction in the genetic algorithm) is necessary.

The following experiments can be used as a starting point:

* Pole balancing: A cart must be moved left/right in order to balance a pole. Can be extended to multiple dimensions.
* Mountain car: This constitutes a fairly simple delayed reward task. A car must drive up a valley, but it does not have enough power to directly drive up the hill. Instead, it must drive back and forth in the valley to gain momentum.
* Water Maze: A partially observable environment task. The simulated rat needs to escape a maze which is filled with water (so that the rat does not want to remain in the maze). The rat must use limited sensory information combined with an internal state in order to reliably escape the maze after repeated trials.

###A Training System

A system must be put in place that normalizes the rewards gained form each experiment (so that the experiments are weighted according to their true value) and feeds the result of multiple runs of multiple experiments into the genetic algorithm for rule reproduction.

This system should have an interface for adding new experiments and saving the population to a file. It should also allow you to look up the individual with the current highest fitness.