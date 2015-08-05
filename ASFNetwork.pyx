#####################################################################
# Name: Yash Patel                                                  #
# File: ASFNetwork.py                                               #
# Description: Contains all the methods pertinent to modelling ASF  #
# network (small world)                                             #
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

class ASFNetwork:
    #################################################################
    # Given a nodeCount for the number of agents to be simulated,   #
    # number of coaches maximally present in the simulation, the    #
    # number of baseline nodes of the graph (m_0), and number       #
    # of edges to be added at each step of the initialization (m)   #
    # produces an ASF network                                       #
    #################################################################
    def __init__(self, nodeCount, percentMinority, timeSpan, m_0 = 4, 
            m = 4):
        if not self.ASFNetwork_verifyNetwork(nodeCount, m_0, m):
            return None

        self.nodeCount = nodeCount

        self.m_0 = m_0
        self.m = m
        self.agentFactory = AgentFactory
        self.percentMinority = percentMinority

        self.Agents = {}
        self.networkBase = NetworkBase("ASFNetwork", timeSpan)
        
        self.ASFNetwork_createAgents()

        # Sets the network base to have the agents just created and
        # the graph just generated and then choosing discriminating
        # portion of the population
        self.networkBase.NetworkBase_setAgents(self.Agents)
        self.networkBase.NetworkBase_chooseDiscriminate()
    
    #################################################################
    # Ensures that the given parameters for defining an SW network  #
    # are appropriate                                               # 
    #################################################################
    def ASFNetwork_verifyNetwork(self, nodeCount, m_0, m):
        if not Verification_verifyInt(nodeCount, "Node count"):
            return False

        if nodeCount < 4:
            sys.stderr.write("Node count must be at least 4")
            return False

        if not Verification_verifyInt(m_0, "Baseline node count"):
            return False

        if m_0 > 10:
            sys.stderr.write("Baseline node count (m_0) must < 10")
            return False

        if not Verification_verifyInt(m, "Incremental edge count (m)"):
            return False

        if m > 10:
            sys.stderr.write("Incremental edge count (m) must < 10")
            return False

        return True

    #################################################################
    # Creates the agents present in the simulation (ASF graph)      #
    #################################################################
    def ASFNetwork_createAgents(self):
        # Creates baseline nodes (from m_0 specified)
        totalConnect = self.m_0

        self.G = nx.Graph()
        self.networkBase.NetworkBase_setGraph(self.G)
        
        self.G.name = "barabasi_albert_graph(%s,%s)"\
            %(self.m,self.nodeCount)

        for i in range(0, totalConnect):
            curAgent = self.agentFactory.\
                AgentFactory_createAgent(self, i, self.percentMinority)
            self.Agents[curAgent.agentID] = curAgent
            self.G.add_node(curAgent.agentID)
        
        for i in range(0, totalConnect):
            for j in range(i, totalConnect):
                self.G.add_edge(i,j)
    
        # Creates remainder of nodes (assigning them to agents)
        for i in range(totalConnect, self.nodeCount):
            curAgent = self.agentFactory.\
                AgentFactory_createAgent(self, i, self.percentMinority)
            self.Agents[curAgent.agentID] = curAgent
            self.G.add_node(curAgent.agentID)
            curAgent.Agent_preferentiallyAttach(self, self.m)