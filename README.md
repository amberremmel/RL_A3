## Reinforcement learning Assignment 3: Policy-based RL
To obtain the graphs that are presented in the report experiment.py can be called. The file experiment.py uses the two files called Helper.py, reinforce.py and actor_critic.py. Helper.py consists of the functions to make and smooth the plots. The file reinforce.py has the reinforce algorithm in it. The file actor_critic.py has the actor-critic algorithm with bootstrap and baseline subtraction in it. The file experiment.py creates two files, a data file with the data of the curves and a plot of the experimented curves.

The replot.py file can use the data files created by experiment.py to combine curves in a new plot.

Python packages numpy, matplotlib, scipy, gym and tensorflow should be installed. The results presented in the report were obtained using python version 3.7.13 with the following package versions:

- tensorflow 2.4.1
- gym 0.21.0
- numpy 1.21.5
- matplotlib 3.5.1
- scipy 1.7.3
- pybox2d 2.3.10
