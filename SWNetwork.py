#####################################################################
# Name: Yash Patel                                                  #
# File: SWNetwork.py                                                #
# Description: Contains all the methods pertinent to modelling SW   #
# network (small world)                                             #
#####################################################################

import sys
import os
import random,itertools
from copy import deepcopy
from numpy import array, zeros, std, mean, sqrt

from NetworkBase import NetworkBase
from Agent import AgentFactory

import matplotlib.pyplot as plt
from operator import itemgetter 

try:
    import networkx as nx
except ImportError:
    raise ImportError("You must install NetworkX:\
    (http://networkx.lanl.gov/) for SE simulation")

class SWNetwork:
    #################################################################
    # Given a nodeCount for the number of agents to be simulated,   #
    # number of coaches maximally present in the simulation, the    #
    # probability of adding a new edge for each edge present to     #
    # other nodes (defaulted to .0), and the number of neighbors to #
    # which each node is to be connected (k) initializes SW Network #
    #################################################################
    def __init__(self, nodeCount, maxCoachCount, k=4, p = 0.0):
        if not self.SWNetwork_verifyNetwork(nodeCount, maxCoachCount,\
                k, p):
            return None

        self.nodeCount = nodeCount
        self.maxCoachCount = maxCoachCount

        self.k = k
        self.p = p
        self.agentFactory = AgentFactory

        self.Agents = {}
        self.networkBase = NetworkBase("SWNetwork", maxCoachCount)

        self.SWNetwork_createAgents()

        # Sets the network base to have the agents just created and
        # the graph just generated
        self.networkBase.NetworkBase_setGraph(self.G)
        self.networkBase.NetworkBase_setAgents(self.Agents)
    
    #################################################################
    # Ensures that the given parameters for defining an SW network  #
    # are appropriate                                               # 
    #################################################################
    def SWNetwork_verifyNetwork(self, nodeCount, maxCoachCount, k, p):
        if not isinstance(nodeCount, int):
            sys.stderr.write("Node count must be of type int")
            return False

        if nodeCount < 4:
            sys.stderr.write("Node count must be at least 4")
            return False

        if not isinstance(maxCoachCount, int):
            sys.stderr.write("Coach count must be of type int")
            return False

        if not isinstance(k, int):
            sys.stderr.write("Neighbor connections (k) must be of \
                type int")
            return False

        if not isinstance(p, float):
            sys.stderr.write("p must be of type double/float")
            return False

        if p < 0.0 or p > 1.0:
            sys.stderr.write("p must be between 0.0-1.0")
            return False

        return True

    #################################################################
    # Creates the agents present in the simulation (SW graph)       #
    #################################################################
    def SWNetwork_createAgents(self):
        self.G = nx.generators.random_graphs.watts_strogatz_graph(
                    n = self.nodeCount,
                    k = self.k,
                    p = self.p,
                    seed = None)
        self.G.name = "small_world_graph(%s,%s,%s)"%(self.nodeCount, \
            self.k, self.p)

        for i in range(0, self.nodeCount):    
            curAgent = self.agentFactory.\
                AgentFactory_createAgent(self, i)
            self.Agents[curAgent.agentID] = curAgent