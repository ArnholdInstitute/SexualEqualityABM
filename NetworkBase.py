#####################################################################
# Name: Yash Patel                                                  #
# File: NetworkBase.py                                              #
# Description: Contains all the methods pertinent to the network    #
# base model, used to produce the other graphs desired to simulate  #
#####################################################################

import sys
import os
import random
from numpy import array, zeros, std, mean, sqrt

from Verification import *

import matplotlib.pyplot as plt
from operator import itemgetter 

try:
    import networkx as nx
except ImportError:
    raise ImportError("You must install NetworkX:\
    (http://networkx.lanl.gov/) for SE simulation")

class NetworkBase:
    #################################################################
    # Initializes the base of the network with the type it is to be #
    # i.e. SW, ER, etc... and number of coaches                     #
    #################################################################
    def __init__(self, networkType):
        if not self.NetworkBase_verifyBase(networkType, maxCoachCount):
            return None
        self.networkType = networkType
        self.policyScore = 0
        self.policies = []

    #################################################################
    # Given parameters for initializing the network base, ensures   #
    # it is legal                                                   #  
    #################################################################
    def NetworkBase_verifyBase(self, networkType):
        if not Verification_verifyStr(networkType):
            return False
        return True

    #################################################################
    # Given a graph G, assigns it to be the graph for this network  #
    #################################################################
    def NetworkBase_setGraph(self, G):
        self.G = G

    #################################################################
    # Given dictionary of agents, assigns them for this network     #
    #################################################################
    def NetworkBase_setAgents(self, agents):
        self.Agents = agents

    #################################################################
    # Simulates updating all agents in network over a single time   #
    # step: includes updating coach presence/retention and SE. Each #
    # of the impact parameters passed in allow for sensitivity      #
    # analysis on each of the factors (i.e. coach impact helps look #
    # at, if the "effectiveness" of the coaches is different, how   #
    # would the final results vary) with default values given       #
    #################################################################
    def NetworkBase_updateAgents(self): 
        for agentID in self.Agents:
            self.Agents[agentID].Agent_timeStep()

    #################################################################
    # Given a list of nodes, adds edges between all of them         #
    #################################################################
    def addEdges(self, nodeList):
        self.G.add_edges_from(nodeList)

    #################################################################
    # Given two agents in the graph, respectively with IDs agentID1 #
    # and agentID2, removes the edge between them                   #
    #################################################################
    def NetworkBase_removeEdge(self, agentID1, agentID2):
        self.G.remove_edge(agentID1, agentID2)

    #################################################################
    # Returns all the edges present in the graph associated with the#
    # network base                                                  #
    #################################################################
    def NetworkBase_getEdges(self):
        return self.G.edges()

    #################################################################
    # Returns the agent associated with the agentID specified       #
    #################################################################
    def NetworkBase_getAgent(self, agentID):
        return self.Agents[agentID]

    #################################################################
    # Returns the total number of agents in the graph associated w/ #
    # the network base                                              #
    #################################################################
    def NetworkBase_getNumAgents(self):
        return len(self.Agents)

    #################################################################
    # Returns an array of the neighbors of a given agent in graph   #
    #################################################################
    def NetworkBase_getNeighbors(self, agent):
        agentID = agent.agentID
        return nx.neighbors(self.G, agentID)

    #################################################################
    # Given a policy, adds it to the policies present in the network#
    # and updates corresponding network score                       #
    #################################################################
    def NetworkBase_addToPolicies(self, policy):
        self.policyScore += policy.score
        self.policies.append(policy)

    #################################################################
    # Determines all the nodes in the overall network/graph that are#
    # or are not of sexual minority: distinguishes based on value of#
    # boolWantMinority. If specified as true, returns minority nodes#
    # in an array; otherwise, returns those nodes not minority      #
    #################################################################
    def NetworkBase_getMinorityNodes(self, boolWantMinority):
        collectNodes = []
        for agent in Agents:
            curAgent = Agents[agent]
            if boolWantMinority:
                if curAgent.isMinority:
                    collectNodes.append(curAgent)
            else:
                if not curAgent.isMinority:
                    collectNodes.append(curAgent)
        return collectNodes

    #################################################################
    # Finds the percentage of locally connected nodes (to some given#
    # agent) marked as of sexual minority                           #
    #################################################################
    def NetworkBase_findPercentConnectedMinority(self, agent):
        neighbors = self.NetworkBase_getNeighbors(agent)
        totalCount = 0
        minorityCount = 0

        for neighbor in neighbors:
            if not Agents[neighbor].isConcealed:
                minorityCount += 1
            totalCount += 1
        return minorityCount/totalCount

    #################################################################
    # Determines the cumulative influence, as defined by the model, #
    # namely Attitude×(SES⁄Ranking)^2                               #
    #################################################################
    def NetworkBase_getTotalInfluence(self, billRank):
        totalInfluence = 0
        for agent in Agents:
            curAgent = Agents[agent]
            totalInfluence += curAgent.Agent_getBillInfluence()
        return totalInfluence

    #################################################################
    # Determines max cumulative influence, as defined by the model, #
    # namely (SES⁄Ranking)^2                                        #
    #################################################################
    def NetworkBase_getMaxTotalInfluence(self):
        maxInfluence = 0
        for agent in Agents:
            curAgent = Agents[agent]
            maxInfluence += curAgent.currentSES ** 2
        return maxInfluence

    #################################################################
    # Assigns to each nodes the appropriate visual attributes, with #
    # those nodes with wellness coaches given a color of red and    #
    # those without blue along with an opacity corresponding to SE  #
    #################################################################
    def NetworkBase_addVisualAttributes(self):
        # Iterate through each of the nodes present in the graph and
        # finds respective agent
        for agentID in self.G.nodes():
            curAgent = self.Agents[agentID]

            exPts = curAgent.Agent_getExercisePts()
            nodeSize = int(500 * exPts/30) 
            self.G.node[agentID]['size'] = nodeSize

            if not curAgent.hasCoach:
                self.G.node[agentID]['color'] = 'red'
            else:
                self.G.node[agentID]['color'] = 'blue'
            self.G.node[agentID]['opacity'] = curAgent.SE

    #################################################################
    # Provides graphical display of the population, color coded to  #
    # illustrate who does and doesn't have the wellness coaches and #
    # sized proportional to the level of exercise. Pass in True for #
    # toShow to display directly and False to save for later view   #
    # with the fileName indicating the current timestep simulated.  #
    # pos provides the initial layout for the visual display        #
    #################################################################
    def NetworkBase_visualizeNetwork(self, toShow, time, pos):
        self.NetworkBase_addVisualAttributes()

        plt.figure(figsize=(12,12))
        for node in self.G.nodes():
            nx.draw_networkx_nodes(self.G,pos, nodelist=[node], 
                node_color=self.G.node[node]['color'],
                node_size=self.G.node[node]['size'], 
                alpha=self.G.node[node]['opacity'])
        nx.draw_networkx_edges(self.G,pos,width=1.0,alpha=.5)

        plt.title("SE Network at Time {}".format(time))
        plt.savefig("Results\\TimeResults\\timestep{}.png".format(time))
        if toShow: 
            plt.show()
        plt.close()