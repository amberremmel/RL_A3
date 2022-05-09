#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
import time
import itertools
import sys

from reinforce import reinforce
from actor_critic import actor_critic
from Helper import LearningCurvePlot, smooth


def average_over_repetitions(algorithm = actor_critic, n_repetitions = 10,
                             n_episodes=1500, learning_rate=0.005, gamma=.8, n_nodes=[64, 128],
                             baseline_subtraction=True, bootstrapping=True, bootstrapping_depth=3,
                             render=False, print_episodes=False, smoothing_window=51, environment='CartPole-v0'):
    """Run the learning algorithm a number of repetitions, and return the mean, min, max values at each timestep
    The standard deviation is also calculated automatically. A dataframe with all traces is also returned"""
    reward_results = np.empty([n_repetitions, n_episodes])  # Result array
    now = time.time()
    for rep in range(n_repetitions):  # Loop over repetitions, and run the algorithm for each one
        print("repetition: ", rep)
        rewards = algorithm(n_episodes, learning_rate, gamma,
                            n_nodes, baseline_subtraction, bootstrapping, bootstrapping_depth, render, print_episodes, environment)
        print("average reward obtained: ", np.mean(rewards))
        reward_results[rep] = rewards
    # calculate max, min, mean and std at each timestep
    max_rewards_per_timestep = np.amax(reward_results, axis=0)
    min_rewards_per_timestep = np.amin(reward_results, axis=0)
    std_per_timestep = np.std(reward_results, axis=0)
    all_repetitions = np.array([str(algorithm), n_repetitions, n_episodes, learning_rate, gamma, n_nodes,\
                               baseline_subtraction, bootstrapping, bootstrapping_depth, reward_results], dtype=object)

    print('Running one setting takes {} minutes'.format((time.time() - now) / 60))
    learning_curve = np.mean(reward_results, axis=0)  # average over repetitions
    learning_curve = smooth(learning_curve, smoothing_window)  # additional smoothing
    max_rewards_per_timestep = smooth(max_rewards_per_timestep, smoothing_window)
    min_rewards_per_timestep = smooth(min_rewards_per_timestep, smoothing_window)
    std_per_timestep = smooth(std_per_timestep, smoothing_window)
    return learning_curve, all_repetitions, max_rewards_per_timestep, min_rewards_per_timestep, std_per_timestep

def parameter_sweep(parameters, title, filename):
    """
    algorithms is a list of functions. on each of these algorithms, a full sweep will be performed.
    parameters is a dictionary, with keys of valid parameters, and values consisting of lists of possible settings for that parameter.

    The function will run each possible combination n_repetitions, the results are averaged, smoothed, and plotted in the results window
    All possible combinations will be plotted to the same window.

    The results will also be saved as a numpy file, for simple replotting of old results.
    The plot made has the caption of 'title'
    all files will be saved with the name 'filename.*'
    standard parameters are defined in the average_over_repetitions() function
    if a different default value is desired, then either change the function mentioned above, or provide in the dict
    the name of that parameter, with the single new default value.
    """
    Plot = LearningCurvePlot(title=title)
    results = np.array([])

    names = parameters.keys()
    settings = [parameters[name] for name in names]
    useful_labels = ['n_nodes', 'learning_rate', 'gamma', 'baseline_subtraction', 'bootstrapping',
                     'bootstrapping_depth', 'smoothing_window' 'n_repetitions']
    ignore_labels = [name for name in names if len(parameters[name])==1] #dont add information to the label if it is the same for all lines in the plot


    for values in itertools.product(*settings):
        #Generate the label for the plot for the current setting
        run = dict(zip(names, values))
        label = [ name + " = " + str(run[name]) for name in names if (name in useful_labels) and (not name in ignore_labels)]
        label = ", ".join(label)
        label = label.replace("learning_rate", "$\\alpha$")
        label = label.replace("gamma", '$\gamma$')
        label = label.replace("bootstrapping_depth", "n")
        label = label.replace("bootstrapping", 'B')
        label = label.replace("baseline_subtraction", "BS")
        label = label.replace("_", " ")
        label = r'{}'.format(label)
        print(" ")
        print(label)
        # use the current settings in the average_over_repetitions function
        learning_curve, all_repetitions, max, min, std = average_over_repetitions(**run)
        results = np.append(results, all_repetitions)

        #make and save the plot after each setting has run. Also save the numpy file of the results
        Plot.add_curve(learning_curve, std, min, max, label=label)
        Plot.save(filename + '.png')
        np.save(filename, results)
    return Plot, results


def ablation():
    """run the ablation study, include the performance of the REINFORCE algorithm without attatched value head"""
    parameters = {"bootstrapping" : [True, False],
                  "baseline_subtraction" : [False],
                  "n_repetitions": [4],
                  'n_episodes': [750],
                  "print_episodes": [True]}
    Plot, results = parameter_sweep(parameters, "Ablation Study", "ablation")
    learning_curve, all_repetitions, max, min, std = average_over_repetitions(algorithm=reinforce, n_repetitions=4, n_episodes=750)
    Plot.add_curve(learning_curve, std, min, max, label="Reinforce")
    Plot.save("ablation.png")
    results = np.append(results, all_repetitions)
    np.save("ablation.npy", results)


def environments():
    """Perform the experiment comparing the performance of the algorithm on different environments"""
    environments = ['CartPole-v0', "LunarLander-v2", 'Acrobot-v1', 'MountainCar-v0']
    thresholds = [195, 200, -100, -110]
    Plot = LearningCurvePlot(title="Performance of agent on different enviroments")

    for env, threshold in zip(environments, thresholds):
        print(env)
        learning_curve, all_repetitions, max, min, std = average_over_repetitions(environment=env, print_episodes=True, render=True)
        #Rescale the learning curve, so the inital performance and the threshold at wich the environment is considered
        #solved occupy the same range on the learning curve.
        C = threshold-learning_curve[0]
        std = std/C
        max = (max-learning_curve[0])/C
        min = (min-learning_curve[0])/C
        learning_curve = learning_curve - learning_curve[0]
        learning_curve = learning_curve/C
        Plot.add_curve(learning_curve, std, min, max, label=env)
        Plot.save("environments.png")


def experiment(study):
    if study == 'ablation':
        ablation()


    elif study == 'environments':
        environments()


    elif study == 'gridsearch':
        parameters = {'learning_rate': [0.01, 0.005],
                      'gamma': [0.7, 0.8],
                      'bootstrapping_depth': [3, 5]}
        parameter_sweep(parameters, "Gridsearch", 'gridsearch')


    elif study=="optimization":
        parameters = {"learning_rate": [0.02, 0.015, 0.01, 0.005, 0.001, 0.0005, 0.0001]}
        parameter_sweep(parameters, "Varying the learning rate", 'learning_rate')

        parameters= {'gamma': [1.1, 1.0, 0.9, 0.8, 0.7, 0.6]}
        parameter_sweep(parameters, "Varying the discount factor", 'discount_factor')

        parameters = {'bootstrapping_depth': [1, 3, 5, 10, 15, 20]}
        parameter_sweep(parameters, "Varying the boostrapping depth", 'bootstrap_depth')

        parameters = {'n_nodes': [[32, 16], [16, 32, 16], [32, 16, 32, 16], [64, 128]]}
        parameter_sweep(parameters, "Varying the number of layers", 'layers')

        parameters = {'n_nodes': [[32], [64], [128], [32, 16]]}
        parameter_sweep(parameters, "Single hidden layer", 'single_layer')

        parameters = {'learning_rate': [0.01, 0.005],
                        'gamma': [0.7, 0.8],
                        'bootstrapping_depth': [3, 5]}
        parameter_sweep(parameters, "Gridsearch", 'gridsearch')

    else:
        print("Run python 'experiment.py optimization' to get the optimization study.")
        print("Run python 'experiment.py gridsearch' to get only the gridsearch from the optimization study.")
        print("Run python 'experiment.py ablation' to get the ablation study.")
        print("Run python 'experiment.py environments' to get the environments study.")



if __name__ == '__main__':

    if len(sys.argv) > 1:
        study = sys.argv[1]  # "optimization" or "ablation" or "exploration"
        if study =="all":
            for string in ["optimization", "ablation", "environments"]:
                experiment(string)
                exit()
        experiment(study)

    else:
        print("Run python 'experiment.py all' to run all studies.")
        print("Run python 'experiment.py optimization' to get the optimization study.")
        print("Run python 'experiment.py gridsearch' to get only the gridsearch from the optimization study.")
        print("Run python 'experiment.py ablation' to get the ablation study.")
        print("Run python 'experiment.py environments' to get the environments study.")
