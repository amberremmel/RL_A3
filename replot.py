#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""
import sys

import numpy as np
import time
import itertools

from Helper import LearningCurvePlot, smooth

def index_to_var_name(x):
    var_names = ['algorithm', 'n_repetitions',
                 'n_episodes', 'learning_rate',
                 'gamma', 'n_nodes',
                 'baseline_subtraction', 'bootstrapping',
                 'n', 'results']
    return var_names[x]

def infer_variables(experiments):
    """Given a numpy array with the data about some number of experiments, determine what variables change between the
    different experiments, and return a list with the index of these variables
    instead of a single npy file, experiments may consist of a list of npy files, then these will be concatenated in that order
    and a plot is made of all the learning curves contained within those files."""
    n_experiments = int(len(experiments)/10)
    settings = []
    variables = []
    for i in range(n_experiments):
        setting = experiments[10*i: 10*i+9]
        settings.append(setting)
    for j, item in enumerate(zip(*settings)):
        if j == 5:
            # The number of nodes is represented as a list, but should be converted to a tuple for equality checking
            item = [tuple(v) for v in item]
        if j == 0:
            # extract the algorithms properly, the location in memory is also stored within the dataframe, but this information is unhelpful
            algorithms = []
            for str in item:
                if 'reinforce' in str:
                    algorithms.append('reinforce')
                if 'actor critic' in str:
                    algorithms.append('actor critic')
            item = algorithms
        #make a set of the list of the settings for a given parameter, if only one unique parameter was used, i.e.
        #if the parameter was not changed, then the length of the set will be 1. if the parameter was changed, then the
        #length will be more than 1.
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
    """Automatically make the labels for the plot, based on what parameters were changed in the different runs"""
    names = [index_to_var_name(i) for i in variables]
    vars = [var for var in setting]
    label = [name + " = " + str(var) for name, var in zip(names, vars)]
    label = ", ".join(label)
    label = label.replace("learning_rate", "$\\alpha$")
    label = label.replace("gamma", '$\gamma$')
    label = label.replace("bootstrapping_depth", "n")
    label = label.replace("bootstrapping", 'B')
    label = label.replace("baseline_subtraction", "BS")
    label = label.replace("_", " ")
    label = r'{}'.format(label)
    return label

def replot(filename, title):
    """Given a npy file, make a new plot of the data contained within. this function is mostly used to change some of
    the plotting parameters"""
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
    if len(sys.argv) == 3:
        replot(sys.argv[1], sys.argv[2])

    else:
        print("Usage: python replot.py \"filename\" \"plot title\"")
