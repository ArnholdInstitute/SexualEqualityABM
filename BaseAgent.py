''' Some really trash code in here: lots of "magic constants" used
to scale from 1.0 to actual thing (in depression model): CHECK. '''

#####################################################################
# Name: Yash Patel                                                  #
# File: BaseAgent.py                                                #
# Description: Object file containing all the methods modelling a   #
# generic agent in the simulation. Agents comprise the individuals  #
# of the simulation, of which the relation between depression and   #
# prevalence of sexual minorities was the main takeaway             #
#####################################################################

import sys
import os
import math
import random
import numpy as np

from Verification import *
from NetworkBase import NetworkBase
from Policy import Policy

import matplotlib.pyplot as plt
from operator import itemgetter 

try:
    import networkx as nx
except ImportError:
    raise ImportError("You must install NetworkX:\
    (http://networkx.lanl.gov/) for SE simulation")

#####################################################################
# A generic base model for agents of the simulation: used to model  #
# the constituent people in a population                            #
#####################################################################
class BaseAgent:

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
            oldDepression, currentDepression, isDepressed, network,
            agentID):
        PROB_MINORITY = .30

        if not self.Agent_verifyAgent(childSES, oldSES, currentSES, 
            minorityAttitude, isMinority, discrimination, support, 
            isConcealed, oldDepression, currentDepression, isDepressed,
            agentID):
            return None

        self.childSES = childSES
        self.oldSES = oldSES
        self.currentSES = currentSES

        self.minorityAttitude = minorityAttitude
        self.attitude = self.minorityAttitude + PROB_MINORITY
        self.isMinority = isMinority

        self.discrimination = discrimination
        self.support = support

        self.isConcealed = isConcealed
        self.oldDepression = oldDepression
        self.currentDepression = currentDepression
        self.isDepressed = isDepressed

        self.network = network.networkBase
        self.agentID = agentID

        self.deltaSES = self.currentSES - self.childSES

    #################################################################
    # Provides an output string for printing out agents             #
    #################################################################
    def __str__(self):
        return ("ID: {}, isMinority: {}, isDepressed: {}, "\
            "isConcealed: {}, SES: {}".format(self.agentID, \
            self.isMinority, self.isDepressed, self.isConcealed, \
            self.currentSES))

    #################################################################
    # Checks that, given all the parameters used to initialize the  #
    # agent, the parameters are legal                               #
    #################################################################
    def Agent_verifyAgent(self, childSES, oldSES, currentSES, 
        baseAttitude, isMinority, discrimination, support, isConcealed, 
        oldDepression, currentDepression, isDepressed, agentID):
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

        if not Verification_verifyInt(agentID, "agentID"):
            return False

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
        # match, we add one to the random number.  So, for example, 
        # suppose we have four nodes with p1=0.5, p2=0.75, p3=0.5 and 
        # p4=0.25.  If our random number is 0.38, we'll first pick node 1, 
        # since 0 < .38 < .5, then skip node 2, since 1.38 > 1.25 
        # (=0.5+0.75), then pick node 3, since 1.25 < 1.38 < 1.75, then 
        # skip node 4, since 2.38 > 2.0

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
        network.networkBase.NetworkBase_addEdges(edges_to_add)

    #################################################################
    # Determines how a given agent will have influence a bill       #
    # his socio-economic standing and attitude towards minorities   #
    #################################################################
    def Agent_getInfluence(self):
        return self.attitude * self.currentSES ** 2

    #################################################################
    # Given a bill's effectiveness, determines how much relative    #
    # impact an agent will have on its passing                      #
    #################################################################
    def Agent_getBillInfluence(self, billRank):
        influence = self.Agent_getInfluence()
        return influence/(billRank ** 2)
        

    #################################################################
    # Given an agent, updates his depression status, based on the   #
    # local and network settings. Note: If agent becomes depressed  #
    # the property remains for the duration of the simulation (can't#
    # become 'undepressed')                                         #
    #################################################################
    def Agent_updateDepression(self):
        DEPRESS_CONST = .0005
        if self.isDepressed:
            return

        self.oldDepression = self.currentDepression

        probDepress = math.exp(-self.deltaSES/(10 * self.currentSES))\
            /(100 * self.support)
        probDepress += self.oldDepression

        if self.isMinority:
            BASELINE_MULT = .001

            # If concealed, greater chance of depression
            concealment = 1.0
            if self.isConcealed:
                concealment *= 2.0

            numPolicies = self.network.policyScore
            percentConnect = self.network.\
                NetworkBase_findPercentConnectedMinority(self)

            probIncrease = BASELINE_MULT * self.discrimination * \
                concealment/(numPolicies/525 * percentConnect)
            probDepress += probIncrease

        self.currentDepression = DEPRESS_CONST * probDepress

        rand = random.random()
        self.isDepressed = (rand < probDepress)

    #################################################################
    # Given an agent, updates his attitudes towards minorities, does#
    # a simulated passing of policy, updates support, discrimination#
    # concealment, and depression for a single time step. Performs  #
    # new policy check every 10 time steps, requiring current time  #                                                   
    #################################################################
    def Agent_updateAgent(self, time):
        self.Agent_updateAttitude()

        if time % 10 == 0:
            newPolicy = Policy()
            newPolicy.Policy_considerPolicy(self.network) 

        self.Agent_updateSupport()
        self.Agent_updateDiscrimination()

        self.Agent_updateConcealment()
        self.Agent_updateDepression()