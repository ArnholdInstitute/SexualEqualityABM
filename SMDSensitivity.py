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

from SexMinDepressionSimulation import *
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
def Sensitivity_runSimulation(networkType, timeSpan, numAgents, 
        percentMinority, supportImpact, concealDiscriminateImpact,
        discriminateConcealImpact, concealDepressionImpact):
    simulationModel = SMDSimulationModel(networkType, timeSpan, numAgents, 
        percentMinority, supportImpact, concealDiscriminateImpact,
        discriminateConcealImpact, concealDepressionImpact)
    simulationModel.SMDModel_runStreamlineSimulation()

    curTrial = []

    curTrial.append(simulationModel.network.networkBase.\
        NetworkBase_findPercentAttr("depression"))
    curTrial.append(simulationModel.network.networkBase.\
        NetworkBase_findPercentAttr("concealed"))

    return curTrial

#####################################################################
# Given an array formatted as [[ExerciseResults, SEResults]...],    #
# as is the case for the results for each of the sensitivity trials #
# reformats the results to be of the form:                          #
# [[Independent Variable Levels], [ExerciseResult1, 2 ...],         # 
# [SEResult1, 2, ...], [Label (text for plotting)]].                #
#####################################################################
def Sensitivity_splitResults(indVarScales, mixedArr, label):
    depressArr = []
    concealArr = []

    for resultsPair in mixedArr:
        depressArr.append(resultsPair[0])
        concealArr.append(resultsPair[1]) 

    finalArr = []
    finalArr.append(indVarScales)
    finalArr.append(depressArr)
    finalArr.append(concealArr)
    finalArr.append(label)

    return finalArr

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
# Performs all the tests for odds ratios to check if results match  #
# empirically verified/identified values from literature            #
#####################################################################
def Sensitivity_oddRatioTests(original):
    network = original.network.networkBase

    ONLY_WANT_WITH = 2
    ONLY_WANT_WITHOUT = 1
    IRRELEVANT = 0

    labels = ["Minority_Depress", "Support_Depress", "Density_Depress"]
    minTest = [ONLY_WANT_WITH, ONLY_WANT_WITHOUT]
    supportTest = [ONLY_WANT_WITHOUT, ONLY_WANT_WITH]
    depressTest = [True, False]
    ORTests = [minTest, supportTest, depressTest]

    ORresults = []

    minDiscriminatePrevalence = network.\
        NetworkBase_findPercentAttr(attr="discrimination")
    ORresults.append(["Minority_Discrimination_Prevalence", \
            minDiscriminatePrevalence])

    # Iterates through each of the odds ratio tests and performs
    # from the above testing values
    args = [IRRELEVANT, IRRELEVANT, False]
    copy = list(args)
    for i in range (0, len(ORTests)):
        print("Performing {} odds ratio test".format(labels[i]))
        test = ORTests[i]
        originalSet = False
        for trial in test:
            args[i] = trial
            trialResult = network.NetworkBase_getDepressOdds(
                onlyMinority=args[0], withSupport=args[1], 
                checkDensity=args[2])
            if not originalSet:
                currentOR = trialResult
                originalSet = True
            else:
                currentOR /= trialResult
        ORresults.append([labels[i], currentOR])
        args = list(copy)

    # Performs numerical analysis on sensitivity trials
    resultsFile = "Results\\Sensitivity\\Sensitivity_OR.txt"
    with open(resultsFile, 'w') as f:
        writer = csv.writer(f, delimiter = ' ', quoting=csv.QUOTE_NONE, 
            quotechar='', escapechar='\\')
        for OR in ORresults:
            writer.writerow(OR)

#####################################################################
# Similarly performs correlation tests to identify value of r btween#
# the parameters and the final result (depression/concealment)      #
#####################################################################
def Sensitivity_correlationTests(networkType, timeSpan, numAgents, 
        percentMinority, supportImpact, concealDiscriminateImpact, 
        discriminateConcealImpact, concealDepressionImpact):
    finalResults = []
    params = [percentMinority, supportImpact, concealDiscriminateImpact, \
        discriminateConcealImpact, concealDepressionImpact]
    toVary = list(params)

    # Used to produce labels of the graphs
    labels = ["Minority_Percentage", "Support_Impact", \
        "ConcealDiscrimination_Impact", "DiscriminateConceal_Impact", \
        "ConcealDepression_Impact"]

    varyTrials = [.50, .75, .90, 1.0, 1.1, 1.25, 1.50]
    for i in range(0, len(params)):
        print("Performing {} sensitivity analysis".format(labels[i]))
        trials = []
        changeParams = []
        
        for trial in varyTrials: 
            toVary[i] *= trial
            changeParams.append(toVary[i])

            trial = Sensitivity_runSimulation(networkType, 
                timeSpan, numAgents, toVary[0], toVary[1], 
                toVary[2], toVary[3], toVary[4])
            trials.append(trial)
            toVary[i] = params[i]
        splitTrial = Sensitivity_splitResults(changeParams, 
            trials, labels[i])
        finalResults.append(splitTrial)
    Sensitivity_printCorrelationResults(finalResults)

#####################################################################
# Prints the results of correlation analysis to separate csv file   #
#####################################################################
def Sensitivity_printCorrelationResults(finalResults):
    # Performs numerical analysis on sensitivity trials
    resultsFile = "Results\\Sensitivity\\Sensitivity_Correlation.txt"
    with open(resultsFile, 'w') as f:
        writer = csv.writer(f, delimiter = '\n', quoting=csv.QUOTE_NONE, 
            quotechar='', escapechar='\\')
        for subResult in finalResults:
            xArr = subResult[0]
            yArr_1 = subResult[1]
            yArr_2 = subResult[2]

            yArrCorrelation_1 = np.corrcoef(xArr, yArr_1)[0][1]
            yArrCorrelation_2 = np.corrcoef(xArr, yArr_2)[0][1]

            depressCorrelate = "{} vs. Depression Correlation: {}".\
                format(subResult[3], yArrCorrelation_1)
            concealCorrelate = "{} vs. Concealment Correlation: {}".\
                format(subResult[3], yArrCorrelation_2)

            row = [depressCorrelate, concealCorrelate]
            writer.writerow(row)

            Sensitivity_plotGraphs(xArr, yArr_1, subResult[3], "Depression")
            Sensitivity_plotGraphs(xArr, yArr_2, subResult[3], "Concealment")

#####################################################################
# Conducts sensitivity tests for each of the paramaters of interest #
# and produces graphical displays for each (appropriately named).   #
# Can also use showOdd and showRegression to respectively choose    #
# to specifically perform odd ratio/regression sensitivity tests    #
#####################################################################
def Sensitivity_sensitivitySimulation(networkType, timeSpan, numAgents, 
        percentMinority, supportImpact, concealDiscriminateImpact, 
        discriminateConcealImpact, concealDepressionImpact, original,
        showOdd=True, showRegression=True):
    if showOdd:
        Sensitivity_oddRatioTests(original)
        
    if showRegression:
        Sensitivity_correlationTests(networkType, timeSpan, numAgents, 
            percentMinority, supportImpact, concealDiscriminateImpact, 
            discriminateConcealImpact, concealDepressionImpact)