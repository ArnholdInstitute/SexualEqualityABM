#####################################################################
# Name: Yash Patel                                                  #
# File: SESimulation.py                                             #
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
    # testing depression as a function of minority prevalence       #
    #################################################################
    def __init__(self, networkType='ER', timeSpan=10, numAgents=10):
        if not self.SMDModel_verifySE(networkType, timeSpan, numAgents):
            return None

        self.networkType = networkType
        self.timeSpan = timeSpan
        self.numAgents = numAgents

        self.SMDModel_setNetwork()
        
    #################################################################
    # Based on the specified value of the network type, generates   #
    # and sets the network accordingly                              #
    #################################################################
    def SMDModel_setNetwork(self):
        if self.networkType == 'ER':
            self.network = ERNetwork(self.numAgents, 10.0/self.numAgents)
        elif self.networkType == 'SW':
            self.network = SWNetwork(self.numAgents, 10, 0.0)
        else:
            self.network = ASFNetwork(self.numAgents, 9, 7)

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
        plt.savefig("Results\\TimeResults\\" + fileName + ".png")
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
        
        agents = self.network.networkBase.NetworkBase_getAgentArray()
        for agent in agents:
            beforeDepressLevels.append(agent.currentDepression)

        for i in range(0, numTicks):
            if i % 10 == 0:
                self.SMDModel_writeSimulationData(i, resultsFile)   

                print("Plotting time step " + str(i))
                self.network.networkBase.\
                    NetworkBase_visualizeNetwork(False, i, pos)

            # Updates the agents in the network base and copies those
            # to the network
            self.network.networkBase.NetworkBase_updateAgents(i)
            self.network.Agents = self.network.networkBase.Agents 

        for agent in agents:
            afterDepressLevels.append(agent.currentDepression)
        self.SMDModel_createBarResults(beforeDepressLevels, 
            afterDepressLevels, "depressBar",
            "Depression", "Indvidual Depression Levels Chart")

    #################################################################
    # Runs simulation over the desired timespan without producing   #
    # visible output: used for sensitivity analysis                 #
    #################################################################
    def SMDModel_runStreamlineSimulation(self):
        # Converts from years to "ticks" (represent 2 week span)
        numTicks = self.timeSpan * 26
        pos = nx.random_layout(self.network.G)

        for i in range(0, numTicks):
            # Updates the agents in the network base and copies those
            # to the network
            self.network.networkBase.NetworkBase_updateAgents(i)
            self.network.Agents = self.network.networkBase.Agents

#####################################################################
# Given the paramters of the simulation (upon being prompted on)    #
# command line, runs simulation, outputting a CSV with each time    #
# step and a graphical display corresponding to the final iteration #
#####################################################################
if __name__ == "__main__":
    # Get all input for initializing simulation

    # ER, SW, or ASF
    networkType = "ER"
    timeSpan = 2
    numAgents = 15

    resultsFile = "Results\\TimeResults\\results.csv"
    simulationModel = SMDSimulationModel(networkType, 
        timeSpan, numAgents)
    simulationModel.SMDModel_runSimulation(resultsFile)

    print("Terminating simulation...")