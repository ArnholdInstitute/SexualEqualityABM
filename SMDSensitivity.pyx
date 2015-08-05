#####################################################################
# Name: Yash Patel                                                  #
# File: SMDSensitivity.py                                           #
# Description: Performs the overall simulations again for depression#
# and concealment, not displaying results of each simulation.Instead#
# looks at and plots sensitivity of these results on each of the    #
# independent parameters while also analyzing the odd ratios of     #
# particular variables to align with values from literature         #
#####################################################################

import sys
import os
import shutil
import csv
import random,itertools
from copy import deepcopy
import numpy as np
from PIL import Image

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
# Given a number n, generates that particular number of empty arrays#
#####################################################################
def generateEmpty(n):
    for _ in range(n):
        yield []

#####################################################################
# Given a number n, generates that many "double empty arrays" (of   #
# the form [[], []])                                                #
#####################################################################
def generateDoubleEmpty(n):
    for _ in range(n):
        first, second = (generateEmpty(2))
        yield [first, second]

#####################################################################
# Given a number n and an array, generates that particular number of# 
# equivalent arrays. Note: simply yields pointers to the same array #
#####################################################################
def generateMultiple(n, arr):
    for _ in range(n):
        yield arr

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
        self.regressionValues = valuesArr

    #################################################################
    # Qualitative tests to ensure that the general behavior (for    #
    # relations for which we do not have specific numerical values) #
    # follows the expected/natural behaviors                        #
    #################################################################
    def test_sanity_checks(self):
        shouldBeNegative = [0, 1, 2, 4]
        shouldBePositive = [6, 7, 8, 9, 10]

        # For these variables, we adopt the naming convention of 
        # varxVary_(Depress/Conceal), where the value measures the 
        # value of regression on the variable relating var x to 
        # var y on either the depression or concealment result
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
            self.assertTrue(self.regressionValues[negativeVal] < 0,
                negError.format(labels[negativeVal]))
        for positiveVal in shouldBePositive:
            self.assertTrue(self.regressionValues[positiveVal] > 0,
                posError.format(labels[positiveVal]))

    #################################################################
    # Quantitative tests to ensure that the relations that have been#
    # previously studied/published in literature align with those in#
    # the simulation                                                #
    #################################################################
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
# simulation and returns an array of all results in the following   #
# format: [depression, concealed, discrimination, support, policy]. #
# Can also be used for running constrained simulations (if the final#
# parameters are passed in with non-None values). All constrained   #
# variables set the corresponding attribute of agents to the given  #
# value, aside from enforcedPolicy, which externally imposes certain#
# policies to be injected into the simulation at specific times     #
#####################################################################
def Sensitivity_runSimulation(simulationModel, percentMinority, 
    supportDepressionImpact, concealDiscriminateImpact, 
    discriminateConcealImpact, discriminateDepressionImpact, 
    concealDepressionImpact, attitude=None, support=None, 
    discrimination=None, conceal=None, depression=None, 
    enforcedPolicy=None):

    ATTR_POS = 0
    PERCENT_POS = 1

    if percentMinority > 1.0:
        percentMinority = 1.0

    simulationModel.percentMinority = percentMinority
    simulationModel.supportDepressionImpact = supportDepressionImpact
    simulationModel.concealDiscriminateImpact = concealDiscriminateImpact
    simulationModel.discriminateConcealImpact = discriminateConcealImpact
    simulationModel.discriminateDepressionImpact = discriminateDepressionImpact
    simulationModel.concealDepressionImpact = concealDepressionImpact

    simulationModel.SMDModel_runStreamlineSimulation(attitude, support, 
        discrimination, conceal, depression, enforcedPolicy)

    network = simulationModel.network.networkBase

    # Each entry has its second value corresponding to the "getPercentage"
    # for the findPercentAttr method
    tests = [["depression", False], ["concealed", False], \
        ["discrimination", True]]
    curTrial = list(map(lambda test : network.NetworkBase_findPercentAttr(
        test[ATTR_POS], getPercentage=test[PERCENT_POS]), tests))
    curTrial.append(network.NetworkBase_setMeanStdSupport(
        onlyMinority=False)[0])
    curTrial.append(network.policyScore)

    return curTrial

#####################################################################
# Given an array formatted as [[DepressResult, ConcealResult]...],  #
# as is the case for the results for each of the sensitivity trials #
# reformats the results to be of the form:                          #
# [[Independent Variable Levels], [DepressResult1, 2 ...],          # 
# [ConcealResult1, 2, ...], [Label (text for plotting)]].           #
#####################################################################
def Sensitivity_splitResults(indVarScales, mixedArr, label):
    depressArr, concealArr, discriminationArr, supportArr, \
        policyArr = generateEmpty(5)

    for resultsPair in mixedArr:
        depressArr.append(resultsPair[0])
        concealArr.append(resultsPair[1])
        discriminationArr.append(resultsPair[2])
        supportArr.append(resultsPair[3])
        policyArr.append(resultsPair[4]) 

    finalArr = [indVarScales, depressArr, concealArr, discriminationArr,\
        supportArr, policyArr, label]
    return finalArr

#####################################################################
# Produces graphical display for the sensitivity results of all     #
# other variables aside from network type: plots line plot for each.#
# graphType can be specified as either "regression" or "impact"     #
# (strings), which will display the graph accordingly. isCompiled   #
# indicates whether or not image is to be compiled (changes labels) #
#####################################################################
def Sensitivity_plotGraphs(xArray, yArray, xLabel, yLabel, 
    graphType, isCompiled=False):
    if graphType == "regression":
        xScale = [.75, 1.25]
        plt.scatter(xArray,yArray)
        folder = "Regression"
    else:
        xScale = [1.0, 1.0]
        plt.plot(xArray, yArray)
        if graphType == "impact":
            folder = "Impact\\{}".format(xLabel)
        else:
            folder = "Sensitivity\\{}".format(xLabel)

    if isCompiled:
        if xLabel == "Policy_Score": plt.ylabel(yLabel)
        if yLabel == "Support": plt.xlabel(xLabel)

        plt.savefig("Results\\{}\\Temp\\{}vs{}.png".format(folder, 
            xLabel, yLabel))

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

    ORresults, values = generateEmpty(2)

    discriminateTestRange = network.\
        NetworkBase_findPercentAttr(attr="discrimination", getPercentage=True)
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
                if trialResult: currentOR /= trialResult
                else: currentOR = 0.0
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

    print(original.network.networkBase.\
        NetworkBase_findPercentAttr(attr="depression", getPercentage=False))

    # Each of these arrays will be used for regression analysis, but
    # as there are very distinct behaviors for concealed vs. not, we
    # consider here separating them into arrays and analyzing separate
    supportArr, concealArr, discriminationArr, \
        depressionArr = generateDoubleEmpty(4)

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
# Performs sensitivity tests to check the various impact ratings on #
# their influence on the output of the simulation                   #
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
        trials, changeParams = generateEmpty(2)
        
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
# Performs sensitivity analyses on the different parameters of      #
# interest in the simulation (i.e. concealment, support, depression #
# policies, discrimination) on the final outcomes/results           #
#####################################################################
def Sensitivity_sensitivityTests(original):
    DEFAULT_VAL = 1.0

    attitudeRange = [-1.0, -.75, -.5, -.25, 0.0, .25, .5, .75, 1.0]
    supportRange, discriminationRange, concealRange, depressionRange,\
        minorityRange = generateMultiple(5, [0.0, .125, .25, .375, \
            .50, .675, .75, .875, 1.0])
    policyScores = [-5, -2, -1, 0, 1, 2, 5]

    sensitivityTests = {
        "Attitude": [attitudeRange, None], 
        "Support": [supportRange, None], 
        "Discrimination": [discriminationRange, None], 
        "Concealment": [concealRange, None], 
        "Depression": [depressionRange, None],
        "Minority_Percentage": [minorityRange, original.percentMinority],
        "Policy_Score": [policyScores, None]
    }

    finalResults = []
    for test in sensitivityTests:
        print("Performing {} sensitivity test".format(test))
        curRange = sensitivityTests[test]
        originalVal = curRange[1]
        trials = []

        for value in curRange[0]:
            curTrial = deepcopy(original)
            curRange[1] = value
            
            attitude = sensitivityTests["Attitude"][1]
            support = sensitivityTests["Support"][1]
            discrimination = sensitivityTests["Discrimination"][1]
            conceal = sensitivityTests["Concealment"][1]
            depression = sensitivityTests["Depression"][1]
            percentMinority = sensitivityTests["Minority_Percentage"][1]
            enforcedPolicy = sensitivityTests["Policy_Score"][1]

            trialResult = Sensitivity_runSimulation(curTrial, 
                percentMinority, curTrial.supportDepressionImpact, 
                curTrial.concealDiscriminateImpact, curTrial.discriminateConcealImpact, 
                curTrial.discriminateDepressionImpact, 
                curTrial.concealDepressionImpact, attitude, support, 
                discrimination, conceal, depression, enforcedPolicy)
            trials.append(trialResult)

        curRange[1] = originalVal
        splitTrial = Sensitivity_splitResults(curRange[0], 
            trials, test)
        finalResults.append(splitTrial)  

    Sensitivity_displaySensitivityResults(finalResults)

#####################################################################
# Prints the results of correlation analysis to separate csv file   #
#####################################################################
def Sensitivity_displaySensitivityResults(finalResults):
    for subResult in finalResults:
        plots = {
            1: "Depression", 
            2: "Concealment", 
            3: "Discrimination", 
            4: "Support", 
            5: "Policy_Score"
        }

        xArr = subResult[0]
        label = subResult[-1]

        for plot in plots:
            Sensitivity_plotGraphs(xArr, subResult[plot], label, 
                plots[plot], "sensitivity", isCompiled=True)
    Sensitivity_displayCompiledResults()

#####################################################################
# Takes the individual images from the sensitivity trials and makes #
# them into a compiled image                                        #
#####################################################################
def Sensitivity_displayCompiledResults():
    # Scaled dimensions (for the x and y) of images
    SCALED_X, SCALED_Y = 400, 300

    # Demarcates the index beyond which it is only independent variables
    IND_BOUNDARY = 5

    print("Compiling images...")
    labels = ["Policy_Score", "Concealment", "Depression", "Discrimination", \
        "Support", "Minority_Percentage", "Attitude"]

    directory = "Results\\Sensitivity\\{}\\Temp\\{}vs{}.png"
    blank_image = Image.new("RGB", (2000, 1500)) 
    maxsize = (SCALED_X, SCALED_Y)

    for i in range(len(labels)):
        for j in range(IND_BOUNDARY):
            firstLabel = labels[i]
            secondLabel = labels[j]
            curLocation = directory.format(firstLabel, 
                firstLabel, secondLabel)
            image = Image.open(curLocation)
            image.thumbnail(maxsize, Image.ANTIALIAS)
            blank_image.paste(image, (i * SCALED_X, j * SCALED_Y))

    out = "Results\\Sensitivity\\CompiledResult.png"
    blank_image.save(out)

#####################################################################
# Prints the results of correlation analysis to text file and also  #
# graphically displays the sensitivity of the results (in conceal   #
# and depression) as a function of the impact ratings               #
#####################################################################
def Sensitivity_printImpactResults(finalResults):
    # Performs numerical analysis on sensitivity trials
    resultsFile = "Results\\Impact\\Impact_Correlation.txt"
    with open(resultsFile, 'w') as f:
        writer = csv.writer(f, delimiter = '\n', quoting=csv.QUOTE_NONE, 
            quotechar='', escapechar='\\')
        for subResult in finalResults:
            plots = {
                1: "Depression", 
                2: "Concealment", 
                3: "Discrimination", 
                4: "Support", 
                5: "Policy Score"
            }

            xArr = subResult[0]
            label = subResult[-1]

            yArrCorrelation_1 = np.corrcoef(xArr, subResult[1])[0][1]
            yArrCorrelation_2 = np.corrcoef(xArr, subResult[2])[0][1]

            depressCorrelate = "{} vs. Depression Correlation: {}".\
                format(label, yArrCorrelation_1)
            concealCorrelate = "{} vs. Concealment Correlation: {}".\
                format(label, yArrCorrelation_2)

            row = [depressCorrelate, concealCorrelate]
            writer.writerow(row)

            for plot in plots:
                Sensitivity_plotGraphs(xArr, subResult[plot], label, 
                    plots[plot], "impact")

#####################################################################
# Conducts sensitivity tests for each of the paramaters of interest #
# and produces graphical displays for each (appropriately named).   #
# Can also use showOdd and showRegression to respectively choose    #
# to specifically perform odd ratio/regression sensitivity tests    #
#####################################################################
def Sensitivity_sensitivitySimulation(percentMinority, 
    supportDepressionImpact, concealDiscriminateImpact, 
    discriminateConcealImpact, discriminateDepressionImpact, 
    concealDepressionImpact, original, final, showOdd=True, 
    showImpact=True, showRegression=True, showSensitivity=True):
    if showOdd:
        Sensitivity_oddRatioTests(final)

    if showRegression:
        Sensitivity_regressionTests(final)

    if showImpact:
        Sensitivity_impactTests(original, percentMinority, 
            supportDepressionImpact, concealDiscriminateImpact, 
            discriminateConcealImpact, discriminateDepressionImpact, 
            concealDepressionImpact)

    if showSensitivity:
        Sensitivity_sensitivityTests(original)