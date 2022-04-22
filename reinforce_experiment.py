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
from Helper import LearningCurvePlot, smooth

def average_over_repetitions(n_repetitions, n_episodes=250,
               learning_rate=0.001, gamma=0.9, n_nodes=[64, 128],
               render=False, smoothing_window=51):

    reward_results = np.empty([n_repetitions, n_episodes]) # Result array
    now = time.time()
    
    for rep in range(n_repetitions): # Loop over repetitions
        rewards = reinforce(n_episodes, learning_rate, gamma, 
               n_nodes, render)
        reward_results[rep] = rewards
        
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))    
    learning_curve = np.mean(reward_results,axis=0) # average over repetitions
    learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
    learning_std = np.std(reward_results,axis=0)
    learning_min = np.min(reward_results,axis=0)
    learning_max = np.max(reward_results,axis=0)
    
    return learning_curve, learning_std, learning_min, learning_max 

def experiment():
    ####### Settings
    # Experiment    
    n_repetitions = 3
    smoothing_window = 21 #kan weg

    n_episodes = 10
    gamma = 0.8
    learning_rate = 0.005
    
    # Hidden layers
    n_nodes = [32, 16]

    # Plotting parameters
    render = False
    
    ####### Experiments
    
    Plot = LearningCurvePlot(title = 'Reinforce')
    
    learning_curve, learning_std, learning_min, learning_max = average_over_repetitions(n_repetitions, n_episodes,
               learning_rate, gamma, n_nodes, render, smoothing_window)
    print(learning_min.shape)
    Plot.add_curve(learning_curve, learning_std, learning_min, learning_max)
    Plot.save('reinforce.png')
    
    
if __name__ == '__main__':
    experiment()
