#####################################################################
# Name: Yash Patel                                                  #
# File: Agent.py                                                    #
# Description: Object file containing all the methods modelling the #
# agents present in the simulation. Agents comprise the individuals #
# of the simulation, of which the relation between depression and   #
# prevalence of sexual minorities was the main takeaway             #
#####################################################################

import sys
import os
import random
import numpy as np

from Verification import *

import matplotlib.pyplot as plt
from operator import itemgetter 

try:
    import networkx as nx
except ImportError:
    raise ImportError("You must install NetworkX:\
    (http://networkx.lanl.gov/) for SE simulation")

#####################################################################
# Used to create several agents to produce the agents en masse for  #
# the setup of the simulation                                       #
#####################################################################
class AgentFactory(object):
    def AgentFactory_createAgent(network, agentID):
        
        return agent

#####################################################################
# An agent of the simulation: used to model the constituent people  #
# in a population                                                   #
#####################################################################
class Agent:

    #################################################################
    # Given the socio-economic status, for childhood, previous time #
    # step, and current; attitude towards sexual minorities; whether#
    # or not is a minority; amount having been discriminated (only  #
    # used if minority); support received from network/policies;    #
    # whether or not concealed (only used if minority); and levels  #
    # of depression at previous and current time steps along with if#
    # acquired depression, initializes an agent for simulation      #
    # Note that all the float values to be provided on 0.0-1.0 scale#
    #################################################################
    def __init__(self, childSES, oldSES, currentSES, minorityAttitude, 
            isMinority, discrimination, support, isConcealed, 
            oldDepression, currentDepression, isDepressed):
        if not self.Agent_verifyAgent(childSES, oldSES, currentSES, 
            minorityAttitude, isMinority, discrimination, support, 
            isConcealed, oldDepression, currentDepression, isDepressed):
            return None

        self.childSES = childSES
        self.oldSES = oldSES
        self.currentSES = currentSES

        self.minorityAttitude = minorityAttitude
        self.isMinority = isMinority

        self.discrimination = discrimination
        self.support = support

        self.isConcealed = isConcealed
        self.oldDepression = oldDepression
        self.currentDepression = currentDepression
        self.isDepressed = isDepressed    

    #################################################################
    # Checks that, given all the parameters used to initialize the  #
    # agent, the parameters are legal                               #
    #################################################################
    def Agent_verifyAgent(self, childSES, oldSES, currentSES, 
        baseAttitude, isMinority, discrimination, support, isConcealed, 
        oldDepression, currentDepression, isDepressed):
        # Contains all the float variables to be checked for bounded
        # conditions, namely between 0.0-1.0
        boundsVerficationDict = {
            childSES: "Childhood socio-economic",
            oldSES: "Old socio-economic",
            currentSES: "Current socio-economic",
            baseAttitude: "Baseline attitude",
            discrimination: "Discrimination",
            support: "Support",
            oldDepression: "Old depression level",
            currentDepression: "Current depression level"
        }

        booleanVerificationDict = {
            isMinority: "isMinority", 
            isConcealed: "isConcealed", 
            isDepressed: "isDepressed" 
        }

        for toCheck in boundsVerficationDict:
            if not Verification_verifyFloat(toCheck, 
                boundsVerficationDict[toCheck]):
                return False
            if not Verification_verifyInBounds(toCheck, 
                boundsVerficationDict[toCheck]):
                return False

        for toCheck in booleanVerificationDict:
            if not Verification_verifyBool(toCheck, 
                booleanVerificationDict[toCheck]):
                return False

        return True

    #################################################################
    # Used for initialization of ASF network: determines to which   #
    # nodes a given agent will attach (based on connection density).#
    # Please note: the following function was taken from Steve      #
    # Mooney's Obesagent ASFNetwork.py program                      #
    #################################################################
    def Agent_preferentiallyAttach(self, network, nConnections):
        candidate_nodes = network.G.nodes()
        
        # Reorder candidates to ensure randomness
        random.shuffle(candidate_nodes)
        target_nodes = []

        # Double edge count to get per-node edge count.
        edge_count = len(network.G.edges(candidate_nodes))*2

        # Pick a random number
        rand = random.random()
        p_sum = 0.0
        
        # To add edges per the B-A algorithm, we compute probabilities
        # for each node multiplying by nConnections, then partition the
        # probability space per the probabilities.  Every time we find a
        # match, we add one to the random number.  So, for example, suppose we
        # have four nodes with p1=0.5, p2=0.75, p3=0.5 and p4=0.25.  If our
        # random number is 0.38, we'll first pick node 1, since 0 < .38 < .5,
        # then skip node 2, since 1.38 > 1.25 (=0.5+0.75), then pick node 3,
        # since 1.25 < 1.38 < 1.75, then skip node 4, since 2.38 > 2.0

        # Note that because we randomized candidates above, the selection is
        # truly random.

        for i in range(len(candidate_nodes)):
            candidate_node = candidate_nodes[i]
            candidate_edges = network.G.edges(candidate_node)
            p_edge = nConnections*1.0*len(candidate_edges)/edge_count
            low = p_sum
            high = p_sum+p_edge
            test = rand + len(target_nodes)
            if (test > low and test <= high):
                target_nodes.append(candidate_node)
            p_sum += p_edge

        node_list = [self.agentID]*len(target_nodes)
        edges_to_add = zip(node_list, target_nodes)
        network.networkBase.addEdges(edges_to_add)

    def Agent_getOverallAttitude(self): 

    def Agent_getInfluence(self):

    def Agent_getBillInfluence(self):

    def Agent_updateSupport(self):

    def Agent_updateDiscrimination(self):

    def Agent_updateAttitudes(self):

    def Agent_updateConcealment(self):

    def Agent_updateDepression(self):

    def Agent_updateAgent(self):
        Policies 
        Support 
        Discrimination 
        Attitudes 
        Concealment 
        Depression 