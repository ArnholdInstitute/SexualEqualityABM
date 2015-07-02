''' 
=====================================================================
TO CHANGE:
Really really ugly code: definitely try tidying up if possible
=====================================================================
'''
#####################################################################
# Name: Yash Patel                                                  #
# File: SensitivitySimulation.py                                    #
# Description: Performs the overall simulations again for SE over   #
# time, but does not display outputs for each simulation. Instead,  #
# looks at and plots sensitivity of exercise levels/SE results vs.  #
# variables, particularly focusing on the impacts of the updates    #
#####################################################################

import sys
import os
import csv
import random,itertools
from copy import deepcopy
import numpy as np


import matplotlib.pyplot as plt
from operator import itemgetter 

try:
    import networkx as nx
except ImportError:
    raise ImportError("You must install NetworkX:\
    (http://networkx.lanl.gov/) for SE simulation")

#####################################################################
# Given the parameters needed for running simulation, executes the  #
# simulation and returns an array of the final (population) mean    #
# exercise and SE levels                                            #
#####################################################################
def Sensitivity_runSimulation(networkType, timeSpan, numAgents, \
    numCoaches, timeImpact, coachImpact, pastImpact, socialImpact):
    simulationModel = SMDSimulationModel(networkType, 
        timeSpan, numAgents)
    simulationModel.SMDModel_runStreamlineSimulation()

    curTrial = []
    curTrial.append(simulationModel.network.networkBase.\
         NetworkBase_getMeanPopExercise())
    curTrial.append(simulationModel.network.networkBase.\
        NetworkBase_getMeanPopSE())

    return curTrial

#####################################################################
# Given an array formatted as [[ExerciseResults, SEResults]...],    #
# as is the case for the results for each of the sensitivity trials #
# reformats the results to be of the form:                          #
# [[Independent Variable Levels], [ExerciseResult1, 2 ...],         # 
# [SEResult1, 2, ...], [Label (text for plotting)]].                #
#####################################################################
def Sensitivity_splitResults(indVarScales, mixedArr, label):
    exArr = []
    SEArr = []

    for resultsPair in mixedArr:
        exArr.append(resultsPair[0])
        SEArr.append(resultsPair[1]) 

    finalArr = []
    finalArr.append(indVarScales)
    finalArr.append(exArr)
    finalArr.append(SEArr)
    finalArr.append(label)

    return finalArr

#####################################################################
# Note: As a general layout of functioanlity, each sensitivity model#
# takes in all the parameters aside from that being varied: it uses #
# the default values for all others                                 #
# ----------------------------------------------------------------- #
# Investigates the sensitivity of the mean population/SE caused by  #
# time decay                                                        #
#####################################################################
def Sensitivity_timeDecay(networkType, timeSpan, numAgents, numCoaches,\
         coachImpact, pastImpact, socialImpact):
    print("Performing sensitivity on time decay impact")
    timeImpactTrials = [0.00, .001, .0025, .005, .0075, .01, .0125]
    trials = []

    for timeImpact in timeImpactTrials:
        trial = Sensitivity_runSimulation(networkType, timeSpan, \
            numAgents, numCoaches, timeImpact, coachImpact, \
            pastImpact, socialImpact)
        trials.append(trial)
    return Sensitivity_splitResults(timeImpactTrials, trials, \
        "Time_Impact")

#####################################################################
# Investigates the sensitivity of the mean population/SE caused by  #
# the "effectiveness" of the coaches                                #
#####################################################################
def Sensitivity_coachEffectiveness(networkType, timeSpan, numAgents, \
        numCoaches, timeImpact, pastImpact, socialImpact):
    print("Performing sensitivity on coach effectiveness")
    coachImpactTrials = [.100, .125, .15, .175, .20, .225, .25, .275,\
        .30, .325]
    trials = []

    for coachImpact in coachImpactTrials:
        trial = Sensitivity_runSimulation(networkType, timeSpan, \
            numAgents, numCoaches, timeImpact, coachImpact, \
            pastImpact, socialImpact)
        trials.append(trial)
    return Sensitivity_splitResults(coachImpactTrials, trials, \
        "Coach_Effectiveness")

#####################################################################
# Investigates the sensitivity of the mean population/SE caused by  #
# the past behavior                                                 #
#####################################################################
def Sensitivity_pastBehavior(networkType, timeSpan, numAgents, \
        numCoaches, timeImpact, coachImpact, socialImpact):
    print("Performing sensitivity on past impact")
    pastImpactTrials = [0.0, .01, .015, .020, .025, .030, .035, .04, \
        .045, .050]
    trials = []

    for pastImpact in pastImpactTrials:
        trial = Sensitivity_runSimulation(networkType, timeSpan, \
            numAgents, numCoaches, timeImpact, coachImpact, \
            pastImpact, socialImpact)
        trials.append(trial)
    return Sensitivity_splitResults(pastImpactTrials, trials, \
    	"Past_Impact")

#####################################################################
# Investigates the sensitivity of the mean population/SE caused by  #
# the exercise present in the locally connected network             #
#####################################################################
def Sensitivity_socialNetwork(networkType, timeSpan, numAgents, \
        numCoaches, timeImpact, coachImpact, pastImpact):
    print("Performing sensitivity on social impact")
    socialImpactTrials = [0.00, .001, .005, .010, .015, .020, .025, \
        .030, .035]
    trials = []

    for socialImpact in socialImpactTrials:
        trial = Sensitivity_runSimulation(networkType, timeSpan, \
            numAgents, numCoaches, timeImpact, coachImpact, \
            pastImpact, socialImpact)
        trials.append(trial)
    return Sensitivity_splitResults(socialImpactTrials, trials, \
    	"Social_Impact")

#####################################################################
# Investigates the sensitivity of the mean population/SE caused by  #
# the number of coaches present in the network                      #
#####################################################################
def Sensitivity_maxCoachCount(networkType, timeSpan, numAgents, \
        timeImpact, coachImpact, pastImpact, socialImpact):
    print("Performing sensitivity on number of coaches")
    numCoachesTrials = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    trials = []

    for numCoaches in numCoachesTrials:
        trial = Sensitivity_runSimulation(networkType, timeSpan, \
            numAgents, numCoaches, timeImpact, coachImpact, \
            pastImpact, socialImpact)
        trials.append(trial)
    return Sensitivity_splitResults(numCoachesTrials, trials, \
    	"Coach_Count")

#####################################################################
# Investigates the sensitivity of the mean population/SE caused by  #
# the type of network employed for clustering                       #
#####################################################################
def Sensitivity_networkCluster(timeSpan, numAgents, numCoaches, \
        timeImpact, coachImpact, pastImpact, socialImpact):
    print("Performing sensitivity on clustering method")
    networkTypeTrials = ["ER", "SW", "ASF"]
    trials = []

    for networkType in networkTypeTrials:
        trial = Sensitivity_runSimulation(networkType, timeSpan, \
            numAgents, numCoaches, timeImpact, coachImpact, \
            pastImpact, socialImpact)
        trials.append(trial)
    return Sensitivity_splitResults(networkTypeTrials, trials, \
    	"Networks")

#####################################################################
# Produces graphical display for the sensitivity results of the     #
# different network types: produces single bar graphs for SE and Ex #
#####################################################################
def Sensitivity_networkGraphs(xArray, yArray, xLabel, yLabel):
    N = len(xArray)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.5        # the width of the bars
    fig, ax = plt.subplots()

    rects1 = ax.bar(ind, yArray, width, color='b')

    # add some text for labels, title and axes ticks
    ax.set_xlabel(yLabel)
    ax.set_ylabel(xLabel)
    ax.set_title("{} vs. {}".format(xLabel, yLabel))
    ax.set_xticks(ind + width/2)

    ax.set_xticklabels(xArray)
    plt.savefig("Results\\Sensitivity\\Networks\\{}vs{}.png"\
    	.format(xLabel, yLabel))
    plt.close()

#####################################################################
# Produces graphical display for the sensitivity results of all     #
# other variables aside from network type: plots line plot for each #
#####################################################################
def Sensitivity_plotGraphs(xArray, yArray, xLabel, yLabel):
    minX = min(xArray)
    maxX = max(xArray)
    
    minY = min(yArray)
    maxY = max(yArray)

    plt.plot(xArray, yArray)
    plt.axis([minX, maxX, .9 * minY, 1.25 * maxY])
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title('{} Vs. {}'.format(xLabel, yLabel))

    plt.savefig("Results\\Sensitivity\\{}\\{}vs{}.png"\
    	.format(xLabel, xLabel, yLabel))
    plt.close()

#####################################################################
# Conducts sensitivity tests for each of the paramaters of interest #
# and produces graphical displays for each (appropriately named)    #
#####################################################################
def Sensitivity_sensitivitySimulation(networkType, timeSpan,     \
        numAgents, numCoaches, timeImpact, coachImpact,          \
        pastImpact, socialImpact):
    finalResults = []

    finalResults.append(Sensitivity_timeDecay(networkType, timeSpan,\
        numAgents, numCoaches, coachImpact, pastImpact, socialImpact))

    finalResults.append(Sensitivity_socialNetwork(networkType, timeSpan, \
        numAgents, numCoaches, timeImpact, coachImpact, pastImpact))

    finalResults.append(Sensitivity_coachEffectiveness(networkType, \
        timeSpan, numAgents, numCoaches, timeImpact, pastImpact, \
        socialImpact))

    finalResults.append(Sensitivity_maxCoachCount(networkType, timeSpan, \
        numAgents, timeImpact, coachImpact, pastImpact, socialImpact))

    finalResults.append(Sensitivity_pastBehavior(networkType, timeSpan, \
       numAgents, numCoaches, timeImpact, coachImpact, socialImpact))

    networkResults = Sensitivity_networkCluster(timeSpan, numAgents, \
        numCoaches, timeImpact, coachImpact, pastImpact, socialImpact)

    for subResult in finalResults:
        Sensitivity_plotGraphs(subResult[0], subResult[1], 
            subResult[3], "Exercise")
        Sensitivity_plotGraphs(subResult[0], subResult[2], \
            subResult[3], "SE")

    Sensitivity_networkGraphs(networkResults[0], networkResults[1], \
        "Exercise", networkResults[3])
    Sensitivity_networkGraphs(networkResults[0], networkResults[2], \
        "SE", networkResults[3])