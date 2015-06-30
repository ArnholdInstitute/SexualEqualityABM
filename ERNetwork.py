#####################################################################
# Name: Yash Patel                                                  #
# File: ERNetwork.py                                                #
# Description: Contains all the methods pertinent to modelling ER   #
# network (randomized construction)                                 #
#####################################################################

import sys
import os
import random,itertools
from copy import deepcopy
from numpy import array, zeros, std, mean, sqrt

from NetworkBase import NetworkBase
from AgentFactory import AgentFactory
from Agent import MinorityAgent, NonMinorityAgent
from Verification import *

import matplotlib.pyplot as plt
from operator import itemgetter 

try:
    import networkx as nx
except ImportError:
    raise ImportError("You must install NetworkX:\
    (http://networkx.lanl.gov/) for SE simulation")

class ERNetwork:
    #################################################################
    # Given a nodeCount for the number of agents to be simulated,   #
    # number of coaches maximally present in the simulation, and the#
    # probability of attaching to other nodes (defaulted to .5)     #
    # initializes ER Network                                        #
    #################################################################
    def __init__(self, nodeCount, p = 0.5):
        if not self.ERNetwork_verifyNetwork(nodeCount, p):
            return None

        self.nodeCount = nodeCount

        self.p = p
        self.agentFactory = AgentFactory

        self.Agents = {}
        self.networkBase = NetworkBase("ERNetwork")

        self.ERNetwork_createAgents()

        # Sets the network base to have the agents just created and
        # the graph just generated
        self.networkBase.NetworkBase_setGraph(self.G)
        self.networkBase.NetworkBase_setAgents(self.Agents)
    
    #################################################################
    # Ensures that the given parameters for defining an ER network  #
    # are appropriate                                               # 
    #################################################################
    def ERNetwork_verifyNetwork(self, nodeCount, p):
        if not Verification_verifyInt(nodeCount, "Node count"):
            return False

        if nodeCount < 4:
            sys.stderr.write("Node count must be at least 4")
            return False

        if not Verification_verifyFloat(p, "p"):
            return False

        if not Verification_verifyInBounds(p, "p"):
            return False
        return True

    #################################################################
    # Creates the agents present in the simulation (ER graph)       #
    #################################################################
    def ERNetwork_createAgents(self):
        self.G = nx.generators.random_graphs.fast_gnp_random_graph(
                    n = self.nodeCount,
                    p = self.p,
                    seed = None)
        self.G.name = "erdosrenyi_graph(%s,%s)"%(self.nodeCount, self.p)

        for i in range(0, self.nodeCount):    
            curAgent=self.agentFactory.\
                AgentFactory_createAgent(self, i)
            self.Agents[curAgent.agentID] = curAgent