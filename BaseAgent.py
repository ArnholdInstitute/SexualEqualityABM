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
    def __init__(self, currentSES, minorityAttitude, isMinority,
        discrimination, support, isConcealed, probConceal, 
        currentDepression, isDepressed, network, agentID):
        if not self.Agent_verifyAgent(currentSES, minorityAttitude, 
            isMinority, discrimination, support, isConcealed, 
            probConceal, currentDepression, isDepressed, network, agentID):
            return None

        self.currentSES = currentSES

        self.attitude = minorityAttitude
        self.isMinority = isMinority

        self.discrimination = discrimination
        self.support = support

        self.probConceal = probConceal
        self.isConcealed = isConcealed
        
        # If initialized to concealed, "start conceal time" marked as 0
        if self.isConcealed:
            self.concealStart = 0

        # Base depression is only used for non-minority members
        self.baseDepression = currentDepression
        self.currentDepression = currentDepression
        self.isDepressed = isDepressed

        # If initialized to depressed, "start depress time" marked as 0
        if self.isDepressed:
            self.depressStart = 0

        self.network = network.networkBase
        self.agentID = agentID

        # Used to determine whether or not agents have been exposed to
        # the "discrimination decay" for extended periods of time
        self.hasMultipleStagnant = False
            
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
    def Agent_verifyAgent(self, currentSES, minorityAttitude, isMinority,
        discrimination, support, isConcealed, probConceal, 
        currentDepression, isDepressed, network, agentID):
        # Contains all the float variables to be checked for bounded
        # conditions, namely between 0.0-1.0
        boundsVerficationDict = {
            currentSES: "Current socio-economic",
            discrimination: "Discrimination",
            support: "Support",
            currentDepression: "Current depression level",
            probConceal: "Probability of concealment"
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
    # Given a parameter, normalizes to be on a logit scale      #
    #################################################################
    def Agent_getLogistic(self, param):
        return 1/(1 + math.exp(-param))

    #################################################################
    # Given a parameter, normalizes to be on a 0.0 - 1.0 scale      #
    #################################################################
    def Agent_normalizeParam(self, param):
        if param < 0.0:
            return 0.0
        elif param > 1.0:
            return 1.0
        return param

    #################################################################
    # Given a bill's effectiveness, determines how much relative    #
    # impact an agent will have on its passing                      #
    #################################################################
    def Agent_getBillInfluence(self, billRank):
        signedInfluence = 2.0 * (self.attitude - .5) 
        influence = signedInfluence * self.currentSES ** 2
        return influence/(billRank ** 2)

    #################################################################
    # Given an agent, updates his attitudes towards minorities, does#
    # a simulated passing of policy, updates support, discrimination#
    # concealment, and depression for a single time step. Performs  #
    # new policy check every 10 time steps, requiring current time. #
    # All input variables defaulted to None may be used for any     #
    # constrained trials (where a certain attribute does not update:#
    # just remains fixed at given value. Otherwise, corresponding   #
    # the attribute will be updated per usual. Note: all should be  #
    # None unless either a sensitivity analysis or hypothetical test#
    # is being performed                                            #
    #################################################################
    def Agent_updateAgent(self, time, supportDepressionImpact, 
        concealDiscriminateImpact, discriminateConcealImpact, 
        discriminateDepressionImpact, concealDepressionImpact, 
        attitude=None, support=None, discrimination=None,
        conceal=None, depression=None):
        supportConcealImpact = supportDepressionImpact

        # Refers to index in the dictionary entry arrays where the 
        # corresponding elements are found (explained below)
        BEING_RUN = 0
        RUN_FUNCTION = 1
        FUNCTION_ARGS = 2

        shouldSet = lambda attr, agent: attr is not None \
            and agent.isMinority

        setAttitude = shouldSet(attitude, self)
        setSupport = shouldSet(support, self)
        setDiscrimination = shouldSet(discrimination, self)
        setConceal = shouldSet(conceal, self)
        setDepression = shouldSet(depression, self)
        
        if setAttitude: self.attitude = attitude
        if setSupport: self.support = support
        if setDiscrimination: self.discrimination = discrimination
        if setConceal: self.probConceal = conceal
        if setDepression: self.currentDepression = depression

        # Dictionary whose entries (named correspondingly) have arrays
        # have the form [bool, function, args], where the bool determines 
        # whether the attribute in question is being updated, the function
        # is that which governs the update, and args are its arguments
        updateSteps = {
            "attitude": [not setAttitude, self.Agent_updateAttitude, ()],
            "support": [not setSupport, self.Agent_updateSupport, ()],
            "discrimination": [not setDiscrimination,                   \
                self.Agent_updateDiscrimination, (time, concealDiscriminateImpact)],
            "conceal": [not setConceal, self.Agent_updateConcealment,   \
                (discriminateConcealImpact, supportConcealImpact, time)],
            "depress": [not setDepression, self.Agent_updateDepression, \
                (concealDepressionImpact, supportDepressionImpact,      \
                    discriminateDepressionImpact, time)]
        }

        for step in updateSteps:
            curUpdate = updateSteps[step]
            if curUpdate[BEING_RUN]:
                curUpdate[RUN_FUNCTION](*curUpdate[FUNCTION_ARGS])

            # Checks if agent becomes concealed/depressed (if not default)
            elif step == "conceal": 
                self.isConcealed = random.random() < \
                    self.probConceal

            elif step == "depress":
                self.isDepressed = random.random() < \
                    self.currentDepression