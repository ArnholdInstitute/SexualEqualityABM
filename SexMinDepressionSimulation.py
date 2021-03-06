#####################################################################
# Name: Yash Patel                                                  #
# File: SexMinDepressionSimulation.py                               #
# Description: Contains all the methods pertinent to producing the  #
# simulation for modelling the relation between sexual minorities   #
# and depression (SMD simulation)                                   #
#####################################################################

import sys
import os
import csv
import random,itertools
from copy import deepcopy
import numpy as np

from NetworkBase import NetworkBase
from ERNetwork import ERNetwork
from ASFNetwork import ASFNetwork
from SWNetwork import SWNetwork

from SMDSensitivity import *
from Hypothetical import *
from Verification import *

import matplotlib.pyplot as plt
from operator import itemgetter 

try:
    import networkx as nx
except ImportError:
    raise ImportError("You must install NetworkX:\
    (http://networkx.lanl.gov/) for SE simulation")

class SMDSimulationModel:
    #################################################################
    # Given the type of network, the simulation time span, and count#
    # of agents in the network, a simulation is created and run for #
    # testing depression as a function of minority prevalence. Also #
    # have control on the impact ratings of each of the parameters: #
    # defaults have been provided                                   #
    #################################################################
    def __init__(self, networkType='ER', timeSpan=10, numAgents=10,
        percentMinority=.5, supportDepressionImpact=1.25,   
        concealDiscriminateImpact=5.0, discriminateConcealImpact=1.0, 
        discriminateDepressionImpact=3.0, concealDepressionImpact=2.0):

        if not self.SMDModel_verifySE(networkType, timeSpan, numAgents):
            return None

        self.networkType = networkType
        self.timeSpan = timeSpan
        self.numAgents = numAgents
        self.percentMinority = percentMinority

        self.supportDepressionImpact = supportDepressionImpact
        self.concealDiscriminateImpact = concealDiscriminateImpact
        self.discriminateConcealImpact = discriminateConcealImpact
        self.discriminateDepressionImpact = discriminateDepressionImpact
        self.concealDepressionImpact = concealDepressionImpact

        self.SMDModel_setNetwork()
        
    #################################################################
    # Based on the specified value of the network type, generates   #
    # and sets the network accordingly. Sets the initial value of   #
    # simulation to those specified in the parameters (attitude_0   #
    # corresponds to initial value of attitude, etc...)             #
    #################################################################
    def SMDModel_setNetwork(self, attitude_0=None, 
        support_0=None, discrimination_0=None, conceal_0=None, 
        depression_0=None, policyScore_0=None):
        if self.networkType == 'ER':
            self.network = ERNetwork(self.numAgents, 
                self.percentMinority, self.timeSpan, 0.50, attitude_0, 
                support_0, discrimination_0, conceal_0, 
                depression_0, policyScore_0)
        elif self.networkType == 'SW':
            self.network = SWNetwork(self.numAgents, 
                self.percentMinority, self.timeSpan, 10, 0.50, attitude_0, 
                support_0, discrimination_0, conceal_0, 
                depression_0, policyScore_0)
        else:
            self.network = ASFNetwork(self.numAgents, 
                self.percentMinority, self.timeSpan, 3, 4, attitude_0, 
                support_0, discrimination_0, conceal_0, 
                depression_0, policyScore_0)

    #################################################################
    # Given parameters for initializing the simulation, ensures they#
    # are legal                                                     # 
    #################################################################
    def SMDModel_verifySE(self, networkType, timeSpan, numAgents):
        if not Verification_verifyStr(networkType, "Network type"):
            return False

        if networkType != 'SW' and networkType != 'ASF'\
            and networkType != 'ER':
            sys.stderr.write("Network type must either SW, ASF, or ER")
            return False

        if not Verification_verifyInt(timeSpan, "Time span"):
            return False

        if not Verification_verifyInt(numAgents, "Number of agents"):
            return False

        return True

    #################################################################
    # Writes the header of the CSV file to be given as output in the#
    # specified file                                                #
    #################################################################
    def SMDModel_writeSimulationHeader(self, resultsFile):
        if resultsFile is not None:
            columns = ['time', 'agentID', 'attitude', 
            'isMinority', 'discrimination', 'support', 'probConceal',
            'isConcealed', 'currentDepression', 'isDepressed',
            'policy points']
            with open(resultsFile, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(columns)

    #################################################################
    # Writes the current data/parameters corresponding to each agent#
    # in the network at the current time step, given the current    #
    # time in the simulation and the file to be written to          #
    #################################################################
    def SMDModel_writeSimulationData(self, time, resultsFile):
        if resultsFile is not None:
            with open(resultsFile, 'a') as f:
                writer = csv.writer(f)
                Agents = self.network.networkBase.Agents
                for agent in Agents:
                    curAgent = Agents[agent]
                    row = [time, curAgent.agentID, curAgent.attitude, 
                    curAgent.isMinority, curAgent.discrimination, 
                    curAgent.support, curAgent.probConceal, 
                    curAgent.isConcealed, curAgent.currentDepression, 
                    curAgent.isDepressed, curAgent.network.policyScore]

                    writer.writerow(row)

    #################################################################
    # Creates a bar graph comparing two specified values (val1,val2)#
    # outputting result into file with fileName. Uses label, title  #
    # for producing the graph                                       #
    #################################################################
    def SMDModel_createBarResults(self, val1, val2, fileName, label, 
            title):
        N = len(val1)

        ind = np.arange(N)  # the x locations for the groups
        width = 0.25        # the width of the bars
        fig, ax = plt.subplots()

        rects1 = ax.bar(ind, val1, width, color='r')
        rects2 = ax.bar(ind+width, val2, width, color='b')

        # add some text for labels, title and axes ticks
        ax.set_ylabel("Agent ID")
        ax.set_ylabel(label)
        ax.set_title(title)
        ax.set_xticks(ind+width)

        labels = []
        for i in range(0, N):
            curWord = str(i + 1)
            labels.append(curWord)

        ax.set_xticklabels(labels)
        ax.legend( (rects1[0], rects2[0]), ('Before', 'After') )
        plt.savefig("Results\\TimeResults\\{}.png".format(fileName))
        plt.close()

    #################################################################
    # Creates a bar graph comparing two specified values (val1,val2)#
    # (for the mean value of depression before/after)               #
    #################################################################
    def SMDModel_createSingleBars(self, xArray, yArray, xLabel, yLabel):
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
        plt.savefig("Results\\TimeResults\\{}vs{}.png"\
            .format(xLabel, yLabel))
        plt.close()

    #################################################################
    # Runs simulation over the desired timespan and produces/outputs#
    # results in CSV file specified along with displaying graphics  #
    #################################################################
    def SMDModel_runSimulation(self, resultsFile):
        self.SMDModel_writeSimulationHeader(resultsFile)

        # Converts from years to "ticks" (represent 2 week span)
        numTicks = self.timeSpan * 26
        pos = nx.random_layout(self.network.G)

        beforeDepressLevels = []
        afterDepressLevels = []     

        timeLabels = ["Before", "After"]
        curNetwork = self.network.networkBase

        agents = curNetwork.NetworkBase_getMinorityNodes()
        for agent in agents:
            if agent.isMinority:
                beforeDepressLevels.append(agent.currentDepression)
        preAvgDepression = curNetwork.\
            NetworkBase_getMinorityDepressionAvg()

        for i in range(0, numTicks):
            if i % 10 == 0:
                self.SMDModel_writeSimulationData(i, resultsFile)   

                print("Plotting time step {}".format(i))
                self.network.networkBase.\
                    NetworkBase_visualizeNetwork(False, i, pos)

            # Updates the agents in the network base and copies those
            # to the network
            self.network.networkBase.NetworkBase_timeStep(i, 
                self.supportDepressionImpact, self.concealDiscriminateImpact, 
                self.discriminateConcealImpact, self.discriminateDepressionImpact, 
                self.concealDepressionImpact)
            self.network.Agents = self.network.networkBase.Agents 

        for agent in agents:
            if agent.isMinority:
                afterDepressLevels.append(agent.currentDepression)
        postAvgDepression = curNetwork.\
            NetworkBase_getMinorityDepressionAvg()

        avgDepressLevels = [preAvgDepression, postAvgDepression]

        self.SMDModel_createBarResults(beforeDepressLevels, 
            afterDepressLevels, "depressBar",
            "Depression", "Individual Depression Levels Chart")

        self.SMDModel_createSingleBars(timeLabels, avgDepressLevels, 
            "Average_Depression_Level", "Time")

    #################################################################
    # Runs simulation over the desired timespan without producing   #
    # visible output: used for sensitivity analysis. Each of the    #
    # parameters allows manual sets the corresponding attribute (for#
    # all of the agents) to the specified value. If none is given,  #
    # (should be the case if not running sensitivity/hypotheticals) #
    # agents follow given default update behavior for the attribute #
    #################################################################
    def SMDMOdel_runConstSimulation(self, attitude=None, 
        support=None, discrimination=None, conceal=None, 
        depression=None, enforcedPolicy=None):
        # Converts from years to "ticks" (represent 2 week span)    
        numTicks = self.timeSpan * 26

        for i in range(0, numTicks):
            # Updates the agents in the network base and copies those
            # to the network
            self.network.networkBase.NetworkBase_timeStep(i, 
                self.supportDepressionImpact, self.concealDiscriminateImpact, 
                self.discriminateConcealImpact, self.discriminateDepressionImpact, 
                self.concealDepressionImpact, attitude, support, 
                discrimination, conceal, depression, enforcedPolicy)
            
            self.network.Agents = self.network.networkBase.Agents
          
    #################################################################
    # Runs simulation over the desired timespan without producing   #
    # visible output: used for sensitivity analysis. Each of the    #
    # parameters allows manual sets for the initial value of the    #
    # parameter in the simulation. Simulation then runs as normal.  #
    # If none is given, agents follow given default update behavior #
    # for the attribute                                             #
    #################################################################
    def SMDModel_runStreamlineSimulation(self, attitude_0=None, 
        support_0=None, discrimination_0=None, conceal_0=None, 
        depression_0=None, policyScore_0=None):
        # Converts from years to "ticks" (represent 2 week span)    
        numTicks = self.timeSpan * 26
        self.SMDModel_setNetwork(attitude_0, support_0, discrimination_0, 
            conceal_0, depression_0, policyScore_0)

        for i in range(0, numTicks):
            # Updates the agents in the network base and copies those
            # to the network
            self.network.networkBase.NetworkBase_timeStep(i, 
                self.supportDepressionImpact, self.concealDiscriminateImpact, 
                self.discriminateConcealImpact, self.discriminateDepressionImpact, 
                self.concealDepressionImpact)
            
            self.network.Agents = self.network.networkBase.Agents

#####################################################################
# Given the paramters of the simulation (upon being prompted on)    #
# command line, runs simulation, outputting a CSV with each time    #
# step and a graphical display corresponding to the final iteration #   
#####################################################################
def main():
    # Used for performing sensitivity analyses
    checkSensitivity = True
    showOdd = False
    showImpact = False
    showRegression = False
    showSensitivity = True

    # Only runs streamlined simulation (no graphical/textual output)
    onlyStreamlined = True 

    # Conducts the hypothetical tests that are of interest in study
    performHypothetical = False

    # ER, SW, or ASF
    networkType = "ER"
    timeSpan = 5
    numAgents = 250
    percentMinority = .10   

    # The following denote "impact constants" for which we have adopted 
    # the naming convention of firstSecondImpact to denote the impact of
    # first on second
    supportDepressionImpact = 4.75
    concealDiscriminateImpact = 1.25
    discriminateConcealImpact = 1.025
    discriminateDepressionImpact = .65
    concealDepressionImpact = 1.075

    resultsFile = "Results\\TimeResults\\results.csv"
    simulationModel = SMDSimulationModel(networkType, timeSpan, numAgents, 
        percentMinority, supportDepressionImpact, concealDiscriminateImpact, 
        discriminateConcealImpact, discriminateDepressionImpact, 
        concealDepressionImpact)
    original = deepcopy(simulationModel)    
    
    if onlyStreamlined: 
        simulationModel.SMDModel_runStreamlineSimulation()
    else:
        simulationModel.SMDModel_runSimulation(resultsFile)

    if checkSensitivity:
        Sensitivity_sensitivitySimulation(percentMinority, 
            supportDepressionImpact, concealDiscriminateImpact, 
            discriminateConcealImpact, discriminateDepressionImpact, 
            concealDepressionImpact, original, simulationModel, 
            showOdd, showImpact, showRegression, showSensitivity)

    if performHypothetical:
        Hypothetical_findEffectiveness(original, simulationModel)

    print("Terminating simulation...")

if __name__ == "__main__":
    main()