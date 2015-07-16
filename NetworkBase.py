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
from Policy import Policy

import matplotlib.pyplot as plt
from operator import itemgetter 

try:
    import networkx as nx
except ImportError:
    raise ImportError("You must install NetworkX:\
    (http://networkx.lanl.gov/) for SE simulation")


class switch(object):
    #################################################################
    # Replicates the behavior of a switch statement (for clarity)   #
    #################################################################
    def __init__(self, value):
        self.value = value
        self.fall = False

    #################################################################
    # Return the match method once, then stop                       #
    #################################################################
    def __iter__(self):
        yield self.match
        raise StopIteration
    
    #################################################################
    # Indicate whether or not to enter a case suite                 #
    #################################################################
    def match(self, *args):
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False

class NetworkBase:
    #################################################################
    # Initializes the base of the network with the type it is to be #
    # i.e. SW, ER, etc... and number of coaches                     #
    #################################################################
    def __init__(self, networkType, timeSpan):
        if not self.NetworkBase_verifyBase(networkType):
            return None
        self.networkType = networkType

        # Potential score keeps track of the maximum possible score
        # (once all the incomplete policies have matured)
        self.potentialScore = 0
        self.policyScore = 0

        # Denotes those policies whose effects have and haven't been
        # fully realized
        self.completePolicies = []        
        self.incompletePolicies = []
        
        # Used for "caching": stores results after first calculation
        # since remain constant throughout simulation
        self.networkSES = 0
        self.localSES = {}

        # Parameters to be set later: default to 0 (False) -> not set
        self.densityMean = 0 
        self.densityStd = 0

        # Marks the "max points" that can be achieved in the network
        # for the policy score
        self.policyCap = 10 * timeSpan

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
    def NetworkBase_timeStep(self, time, supportImpact, concealDiscriminateImpact, 
            discriminateConcealImpact, concealDepressionImpact): 
        newPolicy = Policy(time)
        newPolicy.Policy_considerPolicy(self, time, self.policyCap)
        self.NetworkBase_updatePolicyScore(time)
        for agentID in self.Agents:
            self.Agents[agentID].Agent_updateAgent(time, supportImpact,
                concealDiscriminateImpact, discriminateConcealImpact, 
                concealDepressionImpact)

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
    def NetworkBase_getFirstNeighbors(self, agent):
        agentID = agent.agentID
        return nx.neighbors(self.G, agentID)

    #################################################################
    # Returns an array of those in the "social network" of a given  #
    # agent, defined as being those separated by, at most, two      #
    # degrees in the graph (two connections away)                   #
    #################################################################
    def NetworkBase_getNeighbors(self, agent):
        agentID = agent.agentID
        neighbors = self.NetworkBase_getFirstNeighbors(agent)
        '''
        for neighbor in neighbors:
            curNeighbor = self.NetworkBase_getAgent(neighbor)
            secondDegree = self.\
                NetworkBase_getFirstNeighbors(curNeighbor)
            for nextNeighbor in secondDegree:
                if nextNeighbor not in neighbors:
                    neighbors.append(nextNeighbor)
        '''
        return neighbors

    #################################################################
    # Helper function converting the dictionary of agentID and agent#
    # to array of agent objects                                     #
    #################################################################
    def NetworkBase_getAgentArray(self):
        curAgents = []
        for agent in self.Agents:
            curAgents.append(self.Agents[agent])
        return curAgents

    #################################################################
    # Determines which of the agents has a tendency to decrease his #
    # attitude as there are more minority surrounding him (only in  #
    # population of lower SES)                                      #
    #################################################################
    def NetworkBase_chooseDiscriminate(self):
        PROB_DISCRIMINATORY = .25

        maxSES = self.NetworkBase_getMaxSES()
        topCap = maxSES/2
        agents = self.NetworkBase_getAgentArray()

        for agent in agents:
            if agent.currentSES < topCap:
                rand = random.random()
                if rand < PROB_DISCRIMINATORY:
                    agent.isDiscriminatory = True
                else:
                    agent.isDiscriminatory = False
            else:
                agent.isDiscriminatory = False

    #################################################################
    # Returns the maximum SES present amongst the agents in the sim #
    #################################################################
    def NetworkBase_getMaxSES(self):
        SESarr = []
        agents = self.NetworkBase_getAgentArray()
        for agent in agents:
            SESarr.append(agent.currentSES)
        return max(SESarr)

    #################################################################
    # Given a policy, adds it to the policies present in the network#
    # and updates corresponding network score                       #
    #################################################################
    def NetworkBase_addToPolicies(self, policy, time):
        self.potentialScore += policy.score
        self.incompletePolicies.append(policy)

    #################################################################
    # Goes through each of the policies that are incomplete (whose  #
    # effects have not been fully realized) and updates them to     #
    # reflect the current time                                      #
    #################################################################
    def NetworkBase_updatePolicyScore(self, time):
        for incompletePolicy in self.incompletePolicies:
            incompletePolicy.Policy_updateTimeEffect(time, self.policyCap)

            self.policyScore -= incompletePolicy.prevEffect

            if (not incompletePolicy.isDiscriminatory and \
                incompletePolicy.curEffect >= incompletePolicy.score) or\
                (incompletePolicy.isDiscriminatory and \
                incompletePolicy.curEffect <= incompletePolicy.score):

                self.policyScore += incompletePolicy.score

                self.incompletePolicies.remove(incompletePolicy)
                self.completePolicies.append(incompletePolicy)
            else:
                self.policyScore += incompletePolicy.curEffect

    #################################################################
    # Determines all the nodes in the overall network/graph that are#
    # or are not of sexual minority: distinguishes based on value of#
    # wantMinority. If specified as true, returns minority nodes    #
    # in an array; otherwise, returns those nodes not minority      #
    #################################################################
    def NetworkBase_getMinorityNodes(self, wantMinority=True):
        collectNodes = []
        agents = self.NetworkBase_getAgentArray()
        for agent in agents:
            if (wantMinority and agent.isMinority) or \
                (not wantMinority and not agent.isMinority):
                collectNodes.append(agent)
        return collectNodes

    #################################################################
    # Determines the average depression level amongst those agents  #
    # in the network who are in sexual minority                     #
    #################################################################
    def NetworkBase_getMinorityDepressionAvg(self):
        minAvg = []

        minorityAgents = self.NetworkBase_getMinorityNodes()
        for minAgent in minorityAgents:
            minAvg.append(minAgent.currentDepression)
        return mean(minAvg)

    #################################################################
    # Finds the percentage of locally connected nodes (to some given#
    # agent) marked as of sexual minority. firstDegree determines   #
    # whether you wish to only find the percent in 1st degree or 2nd#
    #################################################################
    def NetworkBase_findPercentConnectedMinority(self, agent, 
        firstDegree=False):

        if firstDegree: 
            neighbors = self.NetworkBase_getFirstNeighbors(agent)
        else: 
            neighbors = self.NetworkBase_getNeighbors(agent)

        totalCount = 0
        minorityCount = 0

        for neighbor in neighbors:
            if self.Agents[neighbor].isMinority and \
                not self.Agents[neighbor].isConcealed:
                minorityCount += 1
            totalCount += 1

        return minorityCount/totalCount

    #################################################################
    # Finds the percentage of locally connected nodes (to some given#
    # agent) that has a low tolerance for those of LGB status       #
    #################################################################
    def NetworkBase_findPercentNonAccepting(self, agent):
        neighbors = self.NetworkBase_getNeighbors(agent)
        totalCount = 0
        nonAcceptingCount = 0

        for neighbor in neighbors:
            if self.Agents[neighbor].attitude < .5:
                nonAcceptingCount += 1
            totalCount += 1

        return nonAcceptingCount/totalCount

    #################################################################
    # Determines the average value of an attribute for the entire   #
    # network (either the %depressed or %concealed from minority)   #
    #################################################################
    def NetworkBase_findPercentAttr(self, attr):
        agents = self.NetworkBase_getMinorityNodes()
        
        minCount = len(agents)
        attrTotal = 0
        
        for agent in agents:
            if  (attr == "depression" and agent.isDepressed) or \
                (attr == "concealed" and agent.isConcealed):
                attrTotal += 1
            elif attr == "discrimination":
                attrTotal += agent.discrimination

        if minCount:
            return attrTotal/minCount
        return 0.0

    #################################################################
    # Determines the local average value of an attribute for a given#
    # agent (in his locally connected network)                      #
    #################################################################
    def NetworkBase_getLocalAvg(self, agent, attribute):
        neighbors = self.NetworkBase_getNeighbors(agent)
        totalCount = len(neighbors)
        total = 0

        for neighbor in neighbors:
            if attribute == "SES":
                total += self.Agents[neighbor].currentSES
            elif attribute == "attitude":
                total += self.Agents[neighbor].attitude

        localAvg = total/totalCount
        return localAvg

    #################################################################
    # Given an agent in the network, returns an array formatted as  #
    # [positive average, negative average], where the averages are  #
    # of the attitudes in the local network                         #
    #################################################################
    def NetworkBase_getAttitudes(self, agent):
        posAttitude = []
        negAttitude = []

        neighbors = self.NetworkBase_getNeighbors(agent)
        for neighbor in neighbors:
            curAttitude = self.Agents[neighbor].attitude
            if curAttitude > 0:
                posAttitude.append(curAttitude)
            else:
                negAttitude.append(curAttitude)

        posAvg = self.NetworkBase_arrMean(posAttitude)
        negAvg = self.NetworkBase_arrMean(negAttitude)
        return [posAvg, negAvg]

    #################################################################
    # Sets the network properties of mean density and std deviation #
    # of density to the corresponding values of the network         #
    #################################################################
    def NetworkBase_setMeanStdDensity(self):
        agents = self.NetworkBase_getAgentArray()
        densityArr = []
        for agent in agents:
            densityArr.append(
                self.NetworkBase_findPercentConnectedMinority(agent, 
                    firstDegree=True))

        if not self.densityMean: 
            self.densityMean = mean(densityArr)
        if not self.densityStd: 
            self.densityStd = std(densityArr)

    #################################################################
    # Given an agent, determines his corresponding z-score for the  #
    # density of LGBs in his network                                #
    #################################################################
    def NetworkBase_getDensityZScore(self, agent):
        # Sets the densities only if not already determined
        if not (self.densityMean and self.densityStd):
            self.NetworkBase_setMeanStdDensity()

        curVal = self.NetworkBase_findPercentConnectedMinority(agent)
        mean = self.densityMean
        std = self.densityStd
        return (curVal - mean)/std

    #################################################################
    # Determines the odds of having a particular depression in      #
    # the entire population (default), only non-minority (1),or only# 
    # minority (2), from the value of onlyMinority. withSupport     #
    # determines whether the attribute is only being checked against#
    # the supported agents (2), only non-supported (1), or any (0)  #
    #################################################################
    def NetworkBase_getDepressOdds(self, onlyMinority=0, withSupport=0,
            checkDensity=False):
        # Everyone with > .50 support will be considered "supported"
        SUPPORT_CUTOFF = .0025

        # Used to calculate when the z-score is ".75" (never exact: 
        # use a bounded set to compensate)
        cutoffRange = [.90, 1.10]

        # Used for checking the parameters passed in: whether check
        # for the nodes with a property, without, or without regard
        ONLY_WANT_WITH = 2
        ONLY_WANT_WITHOUT = 1
        IRRELEVANT = 0

        # Determines which agents to check based on parameter
        for case in switch(onlyMinority):
            if case(ONLY_WANT_WITH):
                agents = self.NetworkBase_getMinorityNodes()
                break
            if case(ONLY_WANT_WITHOUT):
                agents = self.NetworkBase_getMinorityNodes(
                    wantMinority=False)
                break
            if case(IRRELEVANT):
                agents = self.NetworkBase_getAgentArray()
                break
            if case():
                sys.stderr.write("Minority bool must be 0, 1, 2")
                return False

        # Agent count: defaulted to number of agents being analyzed
        count = len(agents)

        totalDepression = 0
        for case in switch(withSupport):
            # Gets total depression of those with support
            if case(ONLY_WANT_WITH):
                for agent in agents:
                    if agent.support >= SUPPORT_CUTOFF:
                        totalDepression += agent.currentDepression
                break

            # Gets depression of those without support
            if case(ONLY_WANT_WITHOUT): 
                for agent in agents:
                    if agent.support < SUPPORT_CUTOFF:
                        totalDepression += agent.currentDepression
                break

            # Gets depression for all the agents in check
            if case(IRRELEVANT):
                # If calculating odds for specific level of density
                if checkDensity:
                    # Have to redo count for only agents in cutoff
                    count = 0
                    for agent in agents:
                        z = self.NetworkBase_getDensityZScore(agent)
                        if cutoffRange[0] < z:
                            totalDepression += agent.currentDepression
                            count += 1

                else:
                    for agent in agents:
                        totalDepression += agent.currentDepression
                break

            if case():
                sys.stderr.write("Support bool must be 0, 1, 2")
                return False

        if not count:
            return 0.0
        
        prob = totalDepression/count
        return prob/(1 - prob)

    #################################################################
    # Gets the average of a given array                             #
    #################################################################
    def NetworkBase_arrMean(self, array):
        if len(array) == 0:
            return 0
        return mean(array)

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
        agents = self.NetworkBase_getAgentArray()
        for agent in agents:
            totalInfluence += agent.Agent_getBillInfluence(billRank)
        return totalInfluence

    #################################################################
    # Determines max cumulative influence, as defined by the model, #
    # namely (SES⁄Ranking)^2                                        #
    #################################################################
    def NetworkBase_getMaxTotalInfluence(self):
        maxInfluence = 0
        agents = self.NetworkBase_getAgentArray()
        for agent in agents:
            maxInfluence += agent.currentSES ** 2
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
            self.G.node[agentID]['color'] = 'red'
            if not curAgent.isDepressed:
                self.G.node[agentID]['color'] = 'blue'

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
                node_size=500, node_shape=self.G.node[node]['shape'], 
                alpha=self.G.node[node]['opacity'])
        nx.draw_networkx_edges(self.G,pos,width=1.0,alpha=.5)

        plt.title("Sexual Minority vs Depression at Time {}".format(time))
        plt.savefig("Results\\TimeResults\\timestep{}.png".format(time))
        if toShow: 
            plt.show()
        plt.close()