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
# Used to create several agents to produce the agents en masse for  #
# the setup of the simulation                                       #
#####################################################################
class AgentFactory(object):
    def AgentFactory_createAgent(network, agentID):
        # Constant values (can change for variation in simulation)
        PROB_MINORITY = .30
        PROB_CONCEAL = .50
        BASELINE_ATTITUDE = .25

        BASELINE_DISCRIMINATION = .40
        BASELINE_SUPPORT = .50 

        CONCEAL_DEPRESS_PROB = .40
        UNCONCEAL_DEPRESS_PROB = .15
        
        CENTER_SES_RAND = 3

        rand = random.random()
        isMinority = False
        if rand <= PROB_MINORITY:
            isMinority = True
        
        childSES = np.random.poisson(CENTER_SES_RAND)/10
        oldSES = np.random.poisson(CENTER_SES_RAND)/10
        currentSES = np.random.poisson(CENTER_SES_RAND)/10
        
        minorityAttitude = random.random() * BASELINE_ATTITUDE 
 
        discrimination = random.random() * BASELINE_DISCRIMINATION
        support = random.random() * BASELINE_SUPPORT

        # For simplicity in network calculations, assumed to be false
        # if the person is not of sexual minority
        rand = random.random()
        if not isMinority:
            isConcealed = False
        elif rand < PROB_CONCEAL:
            isConcealed = True
        else: 
            isConcealed = False

        if isConcealed:
            rand = random.random()
            oldDepression = rand * CONCEAL_DEPRESS_PROB
            rand = random.random()
            currentDepression = rand * CONCEAL_DEPRESS_PROB
        else:
            rand = random.random()
            oldDepression = rand * UNCONCEAL_DEPRESS_PROB
            rand = random.random()
            currentDepression = rand * UNCONCEAL_DEPRESS_PROB

        rand = random.random()
        isDepressed = False
        if rand < currentDepression:
            isDepressed = True

        agent = Agent(childSES, oldSES, currentSES, minorityAttitude, 
            isMinority, discrimination, support, isConcealed, 
            oldDepression, currentDepression, isDepressed, network,
            agentID)
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
        network.networkBase.addEdges(edges_to_add)

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
    # Given an agent, updates his attitude towards sexual minorities#
    # based on the presence of unconcealed minorities in his network#
    #################################################################
    def Agent_updateAttitude(self): 
        percentConnect = self.network.\
            NetworkBase_findPercentConnectedMinority(self)
        self.attitude = self.minorityAttitude + .75 * percentConnect

    #################################################################
    # Given an agent, updates the support he received based on his  #
    # economic status and current attitudes towards minorities. If  #
    # not a minority, returns 1.0                                   #
    #################################################################
    def Agent_updateSupport(self):
        FULL_SUPPORT = 1.0
        if not self.isMinority:
            return FULL_SUPPORT

        # Network and local SES use "caching" to reduce number calcs
        networkSES = self.network.networkSES
        if networkSES == 0:
            networkSES = self.network.NetworkBase_getNetworkSES()

        if self not in self.network.localSES:
            self.network.NetworkBase_getLocalSES(self)
        localSES = self.network.localSES[self]
        globalSES = self.network.NetworkBase_getNetworkSES()
        
        att = self.network.NetworkBase_getNetworkAttitude()
        localConnect = self.network.\
            NetworkBase_findPercentConnectedMinority(self)

        self.support = localSES/globalSES * localConnect * att

    #################################################################
    # Given an agent, updates his discrimination, based on whether  #
    # or not he is concealed and the overall network sentiments     #
    # towards minorities, expressed through the presence of policies#
    # and attitudes                                                 #
    #################################################################
    def Agent_updateDiscrimination(self):
        if not self.isMinority:
            self.discrimination = 0.0

        numPolicies = self.network.policyScore
        totalInfluence = self.network.NetworkBase_getTotalInfluence(1)
        maxInfluence = self.network.NetworkBase_getMaxTotalInfluence()

        concealment = 1.0
        if self.isConcealed:
            concealment *= 2.0

        self.discrimination = concealment * (1 - ((numPolicies)/125 \
            + totalInfluence/maxInfluence))

    #################################################################
    # Given an agent, updates his concealment, based on the network #
    # sentiments and local support. Probabilistically determines if #
    # agent becomes concealed or not. Note: If an agent becomes     #
    # concealed in the simulation, he can later unconceal himself or#
    # vice versa                                                    #
    #################################################################
    def Agent_updateConcealment(self):
        numPolicies = self.network.policyScore
        probConceal = self.discrimination/(self.support * \
            (numPolicies)/125)

        rand = random.random()
        if rand < probConceal:
            self.isConcealed = True
        else:
            self.isConcealed = False

    #################################################################
    # Given an agent, updates his depression status, based on the   #
    # local and network settings. Note: If agent becomes depressed  #
    # the property remains for the duration of the simulation (can't#
    # become 'undepressed')                                         #
    #################################################################
    def Agent_updateDepression(self):
        if self.isDepressed:
            return

        deltaSES = self.currentSES - self.childSES
        probDepress = math.exp(deltaSES)/self.support
        probDepress += self.oldDepression

        if self.isMinority:
            # If concealed, greater chance of depression
            concealment = 1.0
            if self.isConcealed:
                concealment *= 2.0

            numPolicies = self.network.policyScore
            percentConnect = self.network.\
                NetworkBase_findPercentConnectedMinority(self)

            probIncrease = self.discrimination * concealment/\
                ((numPolicies)/125 * percentConnect)
            probDepress += probIncrease

        rand = random.random()
        if rand < probDepress:
            self.isDepressed = True
        else:
            self.isDepressed = False

    #################################################################
    # Given an agent, updates his attitudes towards minorities, does#
    # a simulated passing of policy, updates support, discrimination#
    # concealment, and depression for a single time step            #                                                   
    #################################################################
    def Agent_updateAgent(self):
        self.Agent_updateAttitude()

        newPolicy = Policy()
        newPolicy.Policy_considerPolicy(self.network) 

        self.Agent_updateSupport()
        self.Agent_updateDiscrimination()
        self.Agent_updateConcealment()
        self.Agent_updateDepression()