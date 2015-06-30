''' Fix the %connected to avoid having to add 1.0 for division by 0 ''' 

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
        if not self.NetworkBase_verifyBase(networkType):
            return None
        self.networkType = networkType

        # Ranked on a scale of 1 to 100: starts at 1 to avoid divide
        # by 0 errors
        self.policyScore = 1
        self.policies = []

        # Used for "caching": stores results after first calculation
        # since remain constant throughout simulation
        self.networkSES = 0
        self.localSES = {}

    #################################################################
    # Given parameters for initializing the network base, ensures   #
    # it is legal                                                   #  
    #################################################################
    def NetworkBase_verifyBase(self, networkType):
        if not Verification_verifyStr(networkType, "Network type"):
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
            self.Agents[agentID].Agent_updateAgent()

    #################################################################
    # Given a list of nodes, adds edges between all of them         #
    #################################################################
    def NetworkBase_addEdges(self, nodeList):
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
            if self.Agents[neighbor].isMinority and \
                not self.Agents[neighbor].isConcealed:
                minorityCount += 1
            totalCount += 1

        # Adding 1.0 avoids division by 0
        return minorityCount/totalCount + 1.0

    #################################################################
    # Determines the local average socio-economic status for a given#
    # agent (in his locally connected network); stores result in    #
    # network array                                                 #
    #################################################################
    def NetworkBase_getLocalSES(self, agent):
        neighbors = self.NetworkBase_getNeighbors(agent)
        totalCount = len(neighbors)
        SEStotal = 0

        for neighbor in neighbors:
            SEStotal += self.Agents[neighbor].currentSES

        localAvg = SEStotal/totalCount
        self.localSES[agent] = localAvg
        return localAvg

    #################################################################
    # Determines the network average for socio-economic status and  #
    # storeS as property of network (since const)                   #
    #################################################################
    def NetworkBase_getNetworkSES(self):
        SEStotal = 0
        for agent in self.Agents:
            SEStotal += self.Agents[agent].currentSES

        self.networkSES = SEStotal/len(self.Agents)
        return self.networkSES

    #################################################################
    # Determines the network average for sexual minority attitude   #
    #################################################################
    def NetworkBase_getNetworkAttitude(self):
        SEStotal = 0
        for agent in self.Agents:
            SEStotal += self.Agents[agent].attitude

        self.networkSES = SEStotal/len(self.Agents)
        return self.networkSES

    #################################################################
    # Determines the cumulative influence, as defined by the model, #
    # namely Attitude×(SES⁄Ranking)^2                               #
    #################################################################
    def NetworkBase_getTotalInfluence(self, billRank):
        totalInfluence = 0
        for agent in self.Agents:
            curAgent = self.Agents[agent]
            totalInfluence += curAgent.Agent_getBillInfluence(billRank)
        return totalInfluence

    #################################################################
    # Determines max cumulative influence, as defined by the model, #
    # namely (SES⁄Ranking)^2                                        #
    #################################################################
    def NetworkBase_getMaxTotalInfluence(self):
        maxInfluence = 0
        for agent in self.Agents:
            curAgent = self.Agents[agent]
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

            # Marks depressed agents as red nodes and blue otherwise
            self.G.node[agentID]['color'] = 'blue'
            if not curAgent.isDepressed:
                self.G.node[agentID]['color'] = 'red'

            # Displays sexual minority as different shape than others
            self.G.node[agentID]['shape'] = 'o'
            if curAgent.isMinority:
                self.G.node[agentID]['shape'] = 's'

            # Makes concealed agents less "visible" in display 
            self.G.node[agentID]['opacity'] = 1.0
            if curAgent.isConcealed:
                self.G.node[agentID]['opacity'] = .5

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
                node_size=500, alpha=self.G.node[node]['opacity'],
                node_shape=self.G.node[node]['shape'])
        nx.draw_networkx_edges(self.G,pos,width=1.0,alpha=.5)

        plt.title("Sexual Minority vs Depression at Time {}".format(time))
        plt.savefig("Results\\TimeResults\\timestep{}.png".format(time))
        if toShow: 
            plt.show()
        plt.close()