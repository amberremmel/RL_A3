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

from reinforce import reinforce
from actor_critic import actor_critic
from Helper import LearningCurvePlot, smooth


def average_over_repetitions(algorithm = reinforce, n_repetitions = 10,
                             n_episodes=1500, learning_rate=0.001, gamma=1.0, n_nodes=[64, 128],
                             baseline_subtraction=True, bootstrapping=True, bootstrapping_depth=1,
                             render=False, print_episodes=False, smoothing_window=51):
    reward_results = np.empty([n_repetitions, n_episodes])  # Result array
    now = time.time()
    for rep in range(n_repetitions):  # Loop over repetitions
        print("repetition: ", rep)
        rewards = algorithm(n_episodes, learning_rate, gamma,
                            n_nodes, baseline_subtraction, bootstrapping, bootstrapping_depth, render, print_episodes)
        print("average reward obtained: ", np.mean(rewards))
        reward_results[rep] = rewards
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
    #algorithms is a list of functions. on each of these algorithms, a full sweep will be performed.
    #parameters is a dictionary, with keys of valid parameters, and values consisting of lists of possible settings for that parameter.

    #The function will run each possible combination n_repetitions, the results are averaged, smoothed, and plotted in the results window
    #All possible combinations will be plotted to the same window.

    #The results will also be saved as a numpy file, for simple replotting of old results.
    #The plot made has the caption of 'title'
    #all files will be saved with the name 'filename.*'
    #standard parameters are defined in the average_over_repetitions() function
    #if a different default value is desired, then either change the function mentioned above, or provide in the dict
    #the name of that parameter, with the single new default value.

    Plot = LearningCurvePlot(title=title)
    results = np.array([])

    names = parameters.keys()
    settings = [parameters[name] for name in names]
    useful_labels = ['n_nodes', 'learning_rate', 'gamma', 'baseline_subtraction', 'bootstrapping',
                     'bootstrapping_depth', 'smoothing_window' 'n_repetitions']
    ignore_labels = [name for name in names if len(parameters[name])==1] #dont add information to the label if it is the same for all lines in the plot


    for values in itertools.product(*settings):
        run = dict(zip(names, values))
        label = [ name + " = " + str(run[name]) for name in names if (name in useful_labels) and (not name in ignore_labels)]
        label = ", ".join(label)
        label = label.replace("learning_rate", "$\\alpha$")
        label = label.replace("gamma", '$\gamma$')
        label = label.replace("_", " ")
        label = r'{}'.format(label)
        print(" ")
        print(label)

        learning_curve, all_repetitions, max, min, std = average_over_repetitions(**run)
        results = np.append(results, all_repetitions)

        Plot.add_curve(learning_curve, std, min, max, label=label)
        Plot.save(filename + '.png')
        np.save(filename, results)
    return Plot, results


def ablation():
    parameters = {"bootstrapping" : [True, False],
                  "baseline_subtraction" : [True, False]}
    Plot, results = parameter_sweep(parameters, "Ablation Study", "ablation")
    learning_curve, all_repetitions, max, min, std = average_over_repetitions(algorithm=reinforce)
    Plot.add_curve(learning_curve, std, min, max, label="Reinforce")
    Plot.save("ablation.png")
    results = np.append(results, all_repetitions)
    np.save("ablation.npy", results)



def experiment():


    parameters = {'bootstrapping_depth' : [5, 20, 50, 100, 150, 200],
                  'n_episodes' : [1000],
                  'n_repetitions' : [5]}
    parameter_sweep(parameters, 'Bootstrapping depth', 'bootstrap_large')

    ablation()

    '''
    parameters = {'n_nodes' : [[32, 16],
                               [16, 32, 16],
                               [32, 16, 32, 16]]}

    parameter_sweep(parameters, 'Effect of varying hidden layers in Full Actor Critic', 'Full_AC_layers')


    parameters = {'learning_rate' : [0.03,
                                     0.025,
                                     0.02,
                                     0.015,
                                     0.01,
                                     0.005,
                                     0.001]}

    parameter_sweep(parameters, 'Effect of learning rate on Full Actor Critic', 'Full_AC_learning_rate')


    parameters = {'gamma' : [1.1,
                             1.0,
                             0.9,
                             0.8,
                             0.7,
                             0.6]}

    parameter_sweep(parameters, 'Effect of discount factor on Full Actor Critic', 'Full_AC_gamma')

    parameters = {'bootstrapping_depth': [1,
                                          2,
                                          3,
                                          4,
                                          5]}

    parameter_sweep(parameters, 'Effect of bootstrapping depth', 'bootstrapping_depth')


    parameters = {'learning_rate' : [0.005,
                                     0.001],
                  'gamma' : [1.0,
                             0.9]}

    parameter_sweep(parameters, "Gridsearch learning rate and discount factor", 'alpha_gamma')

    parameters = {'learning_rate' : [0.005,
                                     0.001],
                  'bootstrapping_depth' : [1,
                                           3,
                                           5]}

    parameter_sweep(parameters, 'Gridsearch learning rate and bootstrapping depth', 'alpha_depth')

    parameters = {'gamma' : [1.0,
                            0.9],
                  'bootstrapping_depth' : [1,
                                           3,
                                           5]}

    parameter_sweep(parameters, 'Gridsearch discount factor and bootstrapping depth', 'gamma_depth')
    ####### Experiments
    '''


    '''
    # compare the performance of the different algorithms
    Plot = LearningCurvePlot(title = 'Comparing algorithms')
    algorithm = reinforce
    learning_curve = average_over_repetitions(algorithm, n_repetitions, n_episodes,
               learning_rate, gamma, n_nodes, baseline_subtraction, bootstrapping, bootstrapping_depth, render, smoothing_window)
    Plot.add_curve(learning_curve,label="Reinforce")

    algorithm = actor_critic
    learning_curve = average_over_repetitions(algorithm, n_repetitions, n_episodes,
               learning_rate, gamma, n_nodes, baseline_subtraction, bootstrapping, bootstrapping_depth, render, smoothing_window)
    Plot.add_curve(learning_curve,label="Actor critic, no bootstrapping, no baseline subtraction")

    baseline_subtraction = True
    learning_curve = average_over_repetitions(algorithm, n_repetitions, n_episodes,
               learning_rate, gamma, n_nodes, baseline_subtraction, bootstrapping, bootstrapping_depth, render, smoothing_window)
    Plot.add_curve(learning_curve,label="Actor critic, no bootstrapping, with baseline subtraction")

    baseline_subtraction = False
    bootstrapping=True
    learning_curve = average_over_repetitions(algorithm, n_repetitions, n_episodes,
               learning_rate, gamma, n_nodes, baseline_subtraction, bootstrapping, bootstrapping_depth, render, smoothing_window)
    Plot.add_curve(learning_curve,label="Actor critic, with bootstrapping, no baseline subtraction")

    bootstrapping = True
    baseline_subtraction=True
    learning_curve = average_over_repetitions(algorithm, n_repetitions, n_episodes,
               learning_rate, gamma, n_nodes, baseline_subtraction, bootstrapping, bootstrapping_depth, render, smoothing_window)
    Plot.add_curve(learning_curve,label="Actor critic, with bootstrapping and baseline subtraction")

    Plot.save('comparing_algorithms.png')

    # Varying the bootstrapping depth
    Plot = LearningCurvePlot(title='Bootstrapping depths')
    # learning_rates = [0.1,0.05,0.01,0.005, 0.001]
    depths = [1, 2, 3, 4, 5]
    for depth in depths:
        bootstrapping_depth = depth
        learning_curve = average_over_repetitions(algorithm, n_repetitions, n_episodes,
                                                  learning_rate, gamma, n_nodes, baseline_subtraction, bootstrapping,
                                                  bootstrapping_depth, render, smoothing_window)
        Plot.add_curve(learning_curve, label=r'depth = {} '.format(bootstrapping_depth))
    Plot.save('bootstrap_depth={}.png'.format(depths))
    bootstrapping_depth = 1

    # varying Gamma:
    Plot = LearningCurvePlot(title='Discount factor')

    gammas = [1.1, 1.0, 0.9, 0.8, 0.7, 0.6]
    for g in gammas:
        gamma = g
        learning_curve = average_over_repetitions(algorithm, n_repetitions, n_episodes,
                                                  learning_rate, gamma, n_nodes, baseline_subtraction, bootstrapping,
                                                  bootstrapping_depth, render, smoothing_window)
        Plot.add_curve(learning_curve, label=r'$\gamma = {} '.format(gamma))
    Plot.save('gamma={}.png'.format(depths))
    gamma = 0.8
    '''
    '''
    # Varying the learning_rates
    Plot = LearningCurvePlot(title = 'Complete actor critic learning rates')
    results = np.array([])
    # learning_rates = [0.1,0.05,0.01,0.005, 0.001]
    learning_rates = [0.03, 0.025, 0.02, 0.015, 0.01, 0.005, 0.001]
    title = 'dqn_result_alpha={}'.format(learning_rates)
    for learning_rate in learning_rates:
        learning_curve, all_repetitions = average_over_repetitions(algorithm, n_repetitions, n_episodes,
               learning_rate, gamma, n_nodes, baseline_subtraction, bootstrapping, bootstrapping_depth, render, print_episodes, smoothing_window)
        results = np.append(results, all_repetitions)
        Plot.add_curve(learning_curve,label=r'$\alpha$ = {} '.format(learning_rate))
        Plot.save(title + '.png')
        np.save(title, results)
    '''


if __name__ == '__main__':
    experiment()
