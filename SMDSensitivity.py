#####################################################################
# Name: Yash Patel                                                  #
# File: SMDySimulation.py                                           #
# Description: Performs the overall simulations again for depression#
# and concealment, not displaying results of each simulation.Instead#
# looks at and plots sensitivity of these results on each of the    #
# independent parameters while also analyzing the odd ratios of     #
# particular variables to align with values from literature         #
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

import unittest

try:
    import networkx as nx
except ImportError:
    raise ImportError("You must install NetworkX:\
    (http://networkx.lanl.gov/) for SE simulation")

#####################################################################
# Defines test on which an additional test (to determine whether a  #
# given value is in a specified range) is provided                  #
#####################################################################
class RangeTest(unittest.TestCase):
    def assertInRange(self, value, lower, upper, err):
        self.assertTrue(lower < value < upper, err)

#####################################################################
# Performs tests to ensure OR values that were calculated are in    #
# the ranges specified in the literature                            #
#####################################################################
class OddRatiosTest(RangeTest):
    def __init__(self, valuesArr):
        self.ORTestVals = valuesArr

    def test_odd_ratios(self):
        # Defines ranges (from literature) of OR values
        discriminateTestRange = [.175, .259]
        minTestRange = [1.55, 2.65]
        supportTestRange = [1.5, 4.7]
        depressTestRange = [0.4, 1.2]

        ORRanges = [discriminateTestRange, minTestRange, \
            supportTestRange, depressTestRange]
        labels = ["Discrimination", "Minority", "Support", "Depression"]

        errorStr = "{} not in range"

        for i in range(0, len(ORRanges)):
            self.assertInRange(self.ORTestVals[i], curRange[i][0], 
                curRange[i][1], errorStr.format("{} OR".format(labels[i])))

#####################################################################
# Performs tests to ensure the regression values both match the     #
# values present in literature and too follows expected behavior    #
#####################################################################
class RegressionValueTest(RangeTest):
    def __init__(self, valuesArr):
        # For these variables, we adopt the naming convention of 
        # varxVary_(Depress/Conceal), where the value measures the 
        # value of regression on the variable relating var x to 
        # var y on either the depression or concealment result
        self.regressionValues = valuesArr

    def test_sanity_checks(self):
        shouldBeNegative = [0, 1, 2, 4]
        shouldBePositive = [6, 7, 8, 9, 10]

        labels = {
            0: "minPercentage_Depress",
            1: "minPercentage_Conceal",
            2: "supportDepress_Depress",
            4: "concealDiscrimination_Depress",
            6: "discriminateConceal_Depress",
            7: "discriminateConceal_Conceal",
            8: "discriminationDepression_Depress",
            9: "discriminationDepression_Conceal",
            10: "concealDepression_Depress"
        }

        posError = "{} should be > 0"
        negError = "{} should be < 0"

        for negativeVal in shouldBeNegative:
            self.assertTrue(self.regressionValues[positiveVal] < 0,
                negError.format(labels[negativeVal]))
        for positiveVal in shouldBePositive:
            self.assertTrue(self.regressionValues[positiveVal] > 0,
                posError.format(labels[positiveVal]))

    def test_numerical_values(self):
        # Defines ranges (from literature) of regression values
        supportConcealTestRange = [-.40, -.30] 
        discriminationConcealTestRange = [-.20, -.10]
        discriminationDepressTestRange = [.20, .30]
        concealDepressTestRange = [.22, .33]

        regressionRanges = [supportConcealTestRange,                        \
            discriminationConcealTestRange, discriminationDepressTestRange, \
            concealDepressTestRange]

        # Denotes where the respective variables are in regressions 
        # val array
        testIndices = [3, 5, 8, 10]
        labels = ["Support_Conceal", "Discrimination_Conceal", \
            "Discrimination_Depress", "Conceal_Depress"]

        errorStr = "{} not in range"

        for i in range(0, len(regressionRanges)):
            testIndex = testIndices[i]
            self.assertInRange(self.regressionValues[testIndex], 
                curRange[i][0], curRange[i][1], errorStr.format(
                    "{} Regression".format(labels[i])))

#####################################################################
# Given the parameters needed for running simulation, executes the  #
# simulation and returns an array of the final (population) mean    #
# exercise and SE levels                                            #
#####################################################################
def Sensitivity_runSimulation(simulationModel, percentMinority, 
    supportDepressionImpact, concealDiscriminateImpact, discriminateConcealImpact, 
    discriminateDepressionImpact, concealDepressionImpact):

    if percentMinority > 1.0:
        percentMinority = 1.0

    simulationModel.percentMinority = percentMinority
    simulationModel.supportDepressionImpact = supportDepressionImpact
    simulationModel.concealDiscriminateImpact = concealDiscriminateImpact
    simulationModel.discriminateConcealImpact = discriminateConcealImpact
    simulationModel.discriminateDepressionImpact = discriminateDepressionImpact
    simulationModel.concealDepressionImpact = concealDepressionImpact

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
# Produces graphical display for the sensitivity results of all     #
# other variables aside from network type: plots line plot for each.#
# graphType can be specified as either "regression" or "impact"     #
# (strings), which will display the graph accordingly               #
#####################################################################
def Sensitivity_plotGraphs(xArray, yArray, xLabel, yLabel, graphType):
    minX = min(xArray)
    maxX = max(xArray)
    
    minY = min(yArray)
    maxY = max(yArray)

    if graphType == "regression":
        xScale = [.75, 1.25]
        plt.scatter(xArray,yArray)
        folder = "Regression"
    else:
        xScale = [1.0, 1.0]
        plt.plot(xArray, yArray)
        folder = "Impact\\{}".format(xLabel)

    plt.axis([xScale[0] * minX, xScale[1] * maxX, .9 * minY, 1.25 * maxY])
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title('{} Vs. {}'.format(xLabel, yLabel))

    plt.savefig("Results\\{}\\{}vs{}.png".format(folder, xLabel, yLabel))
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
    values = []

    discriminateTestRange = network.\
        NetworkBase_findPercentAttr(attr="discrimination")
    ORresults.append(["Minority_Discrimination_Prevalence", \
            discriminateTestRange])
    values.append(discriminateTestRange)

    # Iterates through each of the odds ratio tests and performs
    # from the above testing values
    args = [ONLY_WANT_WITH, IRRELEVANT, False]
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
                if trialResult:
                    currentOR /= trialResult
                else:
                    currentOR = 0.0
        ORresults.append([labels[i], currentOR])
        values.append(currentOR)
        args = list(copy)

    # Performs numerical analysis on sensitivity trials
    resultsFile = "Results\\Impact\\Impact_OR.txt"
    with open(resultsFile, 'w') as f:
        writer = csv.writer(f, delimiter = ' ', quoting=csv.QUOTE_NONE, 
            quotechar='', escapechar='\\')
        for OR in ORresults:
            writer.writerow(OR)

#####################################################################
# Similarly performs correlation tests to identify value of r btween#
# the parameters and the final result (depression/concealment)      #
#####################################################################
def Sensitivity_regressionTests(original):
    UNCONCEALED_INDEX = 0
    CONCEALED_INDEX = 1
    OVERALL_INDEX = 2

    # Used to determine the names of files (length of "_vs_" string)
    SEPARATOR_LENGTH = 4

    # Each of these arrays will be used for regression analysis, but
    # as there are very distinct behaviors for concealed vs. not, we
    # consider here separating them into arrays and analyzing separate
    supportArr = [[], []]
    concealArr = [[], []]
    discriminationArr = [[], []]
    depressionArr = [[], []]

    minAgents = original.network.networkBase.NetworkBase_getMinorityNodes()
    for agent in minAgents:
        # The first entry of above arrays contains values corresponding
        # to unconcealed agents and the second for concealed
        concealIndex = int(agent.isConcealed)

        supportArr[concealIndex].append(agent.support)
        concealArr[concealIndex].append(agent.probConceal)
        discriminationArr[concealIndex].append(agent.discrimination)
        depressionArr[concealIndex].append(agent.currentDepression)

    labels = ["Support_vs_Concealment", "Concealment_vs_Discrimination", \
        "Discrimination_vs_Depression", "Concealment_vs_Depression"]

    tests = {
        1: [supportArr, concealArr],
        2: [concealArr, discriminationArr],
        3: [discriminationArr, depressionArr], 
        4: [concealArr, depressionArr]
    }

    finalResults = {}

    # Goes through tests defined in above dictionary (structured as 
    # [a, b] when testing a vs. b) and performs regression analysis
    for test in tests:
        testLabel = labels[test - 1]
        print("Performing {} regression analysis".format(testLabel))

        endIndex = testLabel.index("_vs_")
        startIndex = endIndex + SEPARATOR_LENGTH
        xLabel = testLabel[:endIndex]
        yLabel = testLabel[startIndex:]

        xArrUnconceal = tests[test][0][UNCONCEALED_INDEX]
        xArrConceal = tests[test][0][CONCEALED_INDEX]
        yArrUnconceal = tests[test][1][UNCONCEALED_INDEX] 
        yArrConceal = tests[test][1][CONCEALED_INDEX]

        xArr = xArrUnconceal + xArrConceal
        yArr = yArrUnconceal + yArrConceal

        Sensitivity_plotGraphs(xArr, yArr, xLabel, yLabel, "regression")

        # Adds both the unconcealed and concealed test results to the
        # corresponding dictionary entry
        finalResults[test] = [np.corrcoef(xArrUnconceal, 
            yArrUnconceal)[0][1]]
        finalResults[test].append(np.corrcoef(xArrConceal, 
            yArrConceal)[0][1])
        finalResults[test].append(np.corrcoef(xArr, yArr)[0][1])

    resultsFile = "Results\\Regression\\Regression_Values.txt"
    with open(resultsFile, 'w') as f:
        writer = csv.writer(f, delimiter = '\n', quoting=csv.QUOTE_NONE, 
            quotechar='', escapechar='\\')
        for result in finalResults:
            testLabel = labels[result - 1]
            currentResult = finalResults[result]
            row = [testLabel, \
                "Unconcealed: " + str(currentResult[UNCONCEALED_INDEX]), \
                "Concealed: " + str(currentResult[CONCEALED_INDEX]),     \
                "Overall: " + str(currentResult[OVERALL_INDEX])]
            writer.writerow(row)

#####################################################################
# Performs sensitivity tests to check the various parameters on how #
# the output varies per their variation                             #
#####################################################################
def Sensitivity_impactTests(original, percentMinority, 
    supportDepressionImpact,  concealDiscriminateImpact, 
    discriminateConcealImpact, discriminateDepressionImpact, 
    concealDepressionImpact):
    finalResults = []
    params = [percentMinority, supportDepressionImpact,   \
    concealDiscriminateImpact, discriminateConcealImpact, \
    discriminateDepressionImpact, concealDepressionImpact]
    toVary = list(params)

    # Used to produce labels of the graphs
    labels = ["Minority_Percentage", "SupportDepression_Impact", \
        "ConcealDiscrimination_Impact", "DiscriminateConceal_Impact", \
        "DiscriminationDepression_Impact", "ConcealDepression_Impact"]

    varyTrials = [.50, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
    for i in range(0, len(params)):
        print("Performing {} sensitivity analysis".format(labels[i]))
        trials = []
        changeParams = []
        
        for trial in varyTrials: 
            toVary[i] *= trial
            changeParams.append(toVary[i])

            # Ensures that, when sensitivity analysis is conducted, the network
            # is equivalent to the one that was originally used (keeps constant)
            curTrial = deepcopy(original)
            trialResult = Sensitivity_runSimulation(curTrial, toVary[0], 
                toVary[1], toVary[2], toVary[3], toVary[4], toVary[5])

            trials.append(trialResult)
            toVary[i] = params[i]
        splitTrial = Sensitivity_splitResults(changeParams, 
            trials, labels[i])
        finalResults.append(splitTrial)

    Sensitivity_printImpactResults(finalResults)

#####################################################################
# Prints the results of correlation analysis to separate csv file   #
#####################################################################
def Sensitivity_printImpactResults(finalResults):
    # Performs numerical analysis on sensitivity trials
    resultsFile = "Results\\Impact\\Impact_Correlation.txt"
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

            Sensitivity_plotGraphs(xArr, yArr_1, subResult[3], 
                "Depression", "impact")
            Sensitivity_plotGraphs(xArr, yArr_2, subResult[3], 
                "Concealment", "impact")

#####################################################################
# Conducts sensitivity tests for each of the paramaters of interest #
# and produces graphical displays for each (appropriately named).   #
# Can also use showOdd and showRegression to respectively choose    #
# to specifically perform odd ratio/regression sensitivity tests    #
#####################################################################
def Sensitivity_sensitivitySimulation(percentMinority, supportDepressionImpact, 
    concealDiscriminateImpact, discriminateConcealImpact, 
    concealDepressionImpact, discriminateDepressionImpact, original, 
    final, showOdd=True, showImpact=True, showRegression=True):

    if showOdd:
        Sensitivity_oddRatioTests(final)

    if showRegression:
        Sensitivity_regressionTests(final)

    if showImpact:
        Sensitivity_impactTests(original, percentMinority, 
            supportDepressionImpact, concealDiscriminateImpact, 
            discriminateConcealImpact, discriminateDepressionImpact, 
            concealDepressionImpact)