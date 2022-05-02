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

from Helper import LearningCurvePlot, smooth

def index_to_var_name(x):
    var_names = ['algorithm', 'n_repetitions',
                 'n_episodes', 'learning_rate',
                 'gamma', 'n_nodes',
                 'baseline_subtraction', 'bootstrapping',
                 'bootstrapping_depth', 'results']
    return var_names[x]

def infer_variables(experiments):
    n_experiments = int(len(experiments)/10)
    settings = []
    variables = []
    for i in range(n_experiments):
        setting = experiments[10*i: 10*i+9]
        settings.append(setting)
    for j, item in enumerate(zip(*settings)):
        if j == 5:
            item = [tuple(v) for v in item]
        if j == 0: # extract the algorithms properly
            algorithms = []
            for str in item:
                if 'reinforce' in str:
                    algorithms.append('reinforce')
                if 'actor critic' in str:
                    algorithms.append('actor critic')
            item = algorithms
        n_unique = len(set(item))
        if n_unique>1:
            variables.append(j)
    return variables

def get_curves(experiment, smoothing_window = 51):
    '''
    get the mean, min, max, and mean +- std curves
    from one single run of the algorithm with n timesteps and m repetitions.
    in other words, get the curves from one specific run of the average_over_repetitions function
    '''
    mean = np.mean(experiment, axis = 0)
    std = np.std(experiment, axis=0)
    min = np.amin(experiment, axis=0)
    max = np.amax(experiment, axis=0)

    mean =  smooth(mean, smoothing_window)
    std =  smooth(std, smoothing_window)
    min =  smooth(min, smoothing_window)
    max =  smooth(max, smoothing_window)
    return mean, std, min, max

def make_label_for_curve(variables, setting):
    names = [index_to_var_name(i) for i in variables]
    vars = [var for var in setting]
    label = [name + " = " + str(var) for name, var in zip(names, vars)]
    label = ", ".join(label)
    label = label.replace("learning_rate", "$\\alpha$")
    label = label.replace("gamma", '$\gamma$')
    label = label.replace("_", " ")
    label = r'{}'.format(label)
    return label

def replot(filename, title, smoothing_window = 51):
    if type(filename) is list:
        results_list = [np.load(file, allow_pickle=True) for file in filename]
        results = results_list.pop(0)
        for arr in results_list:
            results = np.append(results, arr)
        filename = filename[0]
    else:
        results = np.load(filename, allow_pickle=True)
    #print(results)
    variables = infer_variables(results)
    n_experiments = int(len(results)/10)
    Plot = LearningCurvePlot(title=title)
    for i in range(n_experiments):
        setting = []
        for var in variables:
            setting.append(results[i*10 + var])
        experiment = results[i*10 +9]
        mean, std, min, max = get_curves(experiment)
        label = make_label_for_curve(variables, setting)
        Plot.add_curve(mean, std, min, max, label=label)
    image_filename = filename[:-4] + ".png"
    Plot.save(image_filename)


if __name__ == '__main__':
    replot(["../results parameter tuning/bootstrapping_depth_1_5_20.npy", "../results parameter tuning/bootstrapping_depth.npy"], title='Bootstrapping depth')
    '''
    replot('alpha_gamma.npy', title = 'Gridsearch alpha and gamma')

    replot('alpha_depth.npy', title = 'Gridsearch alpha and depth')

    replot('Full_AC_layers.npy', title ='Effect of increasing number of layers')

    replot('Full_AC_learning_rate.npy', title='Effect of changing learning rate')

    replot('Full_AC_gamma.npy', title="Effect of changing discount factor")

    replot('bootstrapping_depth.npy', title='Effect of increasing bootstrapping depth')
    '''