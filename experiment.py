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

from reinforce import reinforce
from actor_critic import actor_critic
from Helper import LearningCurvePlot, smooth


def average_over_repetitions(algorithm, n_repetitions,
                             n_episodes=500, learning_rate=0.01, gamma=0.9, n_nodes=[64, 128],
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
    all_repetitions = np.array([str(algorithm), n_repetitions, n_episodes, learning_rate, gamma, n_nodes,\
                               baseline_subtraction, bootstrapping, bootstrapping_depth, reward_results], dtype=object)

    print('Running one setting takes {} minutes'.format((time.time() - now) / 60))
    learning_curve = np.mean(reward_results, axis=0)  # average over repetitions
    learning_curve = smooth(learning_curve, smoothing_window)  # additional smoothing
    return learning_curve, all_repetitions


def experiment():
    ####### Settings
    # Experiment
    n_repetitions = 3
    smoothing_window = 51

    n_episodes = 100
    gamma = 0.8
    learning_rate = 0.01

    # Hidden layers
    n_nodes = [32, 16]

    algorithm = actor_critic

    # actor critic parameters:
    baseline_subtraction = True
    bootstrapping = True
    bootstrapping_depth = 1

    # Plotting parameters
    render = False
    print_episodes = False

    ####### Experiments

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




if __name__ == '__main__':
    experiment()
