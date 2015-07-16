#####################################################################
# Name: Yash Patel                                                  #
# File: Agent.py                                                    #
# Description: Object file containing minority and non-minority     #
# agents (derived from BaseAgent class). Used to respectively model #
# the sexual minorities and non-minorities of population in sim     #
#####################################################################

import sys
import os
import math
import random
import numpy as np

from BaseAgent import BaseAgent

import matplotlib.pyplot as plt
from operator import itemgetter 

try:
    import networkx as nx
except ImportError:
    raise ImportError("You must install NetworkX:\
    (http://networkx.lanl.gov/) for SE simulation")

#####################################################################
# A model for agents part of sexual minority                        #
#####################################################################
class NonMinorityAgent(BaseAgent):

    #################################################################
    # Given an agent, updates his attitude towards sexual minorities#
    # based on the presence of unconcealed minorities in his network#
    #################################################################
    def Agent_updateAttitude(self): 
        percentConnect = self.network.\
            NetworkBase_findPercentConnectedMinority(self)

        # Accounts for negative sentiments diffusing in network
        percentPoorNonAccept = self.network.\
            NetworkBase_findPercentNonAccepting(self)

        # Accounts for those who reflect feeling uneasy when with 
        # those of the sexual minority community
        deltaMinority = .75 * percentConnect/self.network.policyCap
        deltaNonMinority = percentPoorNonAccept/self.network.policyCap

        if self.isDiscriminatory:
            self.attitude -= deltaMinority
        else:
            self.attitude += deltaMinority
        self.attitude -= deltaNonMinority

    #################################################################
    # As those not of minorities are assumed to have full support,  #
    # no update is needed                                           #
    #################################################################
    def Agent_updateSupport(self, supportImpact):
        return

    #################################################################
    # As those not of minorities are assumed to not be discriminated#
    # against, no update is needed for them                         #
    #################################################################
    def Agent_updateDiscrimination(self, time, discriminateImpact):
        return

    #################################################################
    # As concealment only applies to minorities, no update is needed#
    #################################################################
    def Agent_updateConcealment(self, discriminateConcealImpact):
        return

    #################################################################
    # Simply due to the results of interest for this investigation  #
    # we consider the depression of the non-minority members to     #
    # perform sensitivity analysis and contrast with literature     #
    #################################################################
    def Agent_updateDepression(self, concealImpact, time):
        DEPRESSION_THRESHOLD = .025
        SCALING_FACTOR = .0025
        TIME_DECAY = .925

        self.baseDepression *= TIME_DECAY
        if self.isDiscriminatory:
            self.currentDepression = self.baseDepression + self.network.\
                NetworkBase_findPercentConnectedMinority(self) * SCALING_FACTOR

        rand = random.random()
        self.isDepressed = (rand < self.currentDepression and \
            self.currentDepression > DEPRESSION_THRESHOLD)

#####################################################################
# A model for agents not part of sexual minority                    #
#####################################################################
class MinorityAgent(BaseAgent):

    #################################################################
    # Since minority agents are assumed to be fully accepting of one#
    # another, no update is necessary                               #
    #################################################################
    def Agent_updateAttitude(self): 
        return

    #################################################################
    # Given an agent, updates the support he received based on his  #
    # economic status and current attitudes towards minorities. If  #
    # not a minority, returns 1.0                                   #
    #################################################################
    def Agent_updateSupport(self, supportImpact):
        ADDITIONAL_BOOST = .10

        att = self.network.NetworkBase_getNetworkAttitude()
        localConnect = self.network.\
            NetworkBase_findPercentConnectedMinority(self)

        # Accounts for additional boost felt when those opposing are
        # in significant minority
        const = 0
        if att > .75:
            const = ADDITIONAL_BOOST

        support = supportImpact * localConnect * att + const
        self.support = self.Agent_normalizeParam(support)

    #################################################################
    # Given an agent, updates his discrimination, based on whether  #
    # or not he is concealed and the overall network sentiments     #
    # towards minorities, expressed through the presence of policies#
    # and attitudes                                                 #
    #################################################################
    def Agent_updateDiscrimination(self, time, concealDiscriminateImpact):
        DECAY_FACTOR = 5.0
        SCALE_FACTOR = .075

        numPolicies = self.network.policyScore
        avgAttitude = self.network.NetworkBase_getLocalAvg(self, \
            "attitude")
        
        if self.isConcealed:
            if not self.hasMultipleStagnant:
                self.hasMultipleStagnant = True

                # Used to determine the extent to which network has
                # an effect on discrimination
                self.time = time

                attitudes = self.network.NetworkBase_getAttitudes(self)
                self.initialPositive = attitudes[0]
                self.initialNegative = attitudes[1]

            deltaTime = time - self.time
            discrimination = 1 - (numPolicies/self.network.policyCap \
                + (self.initialPositive + self.initialNegative * \
                concealDiscriminateImpact ** (-deltaTime)))
            discrimination *= SCALE_FACTOR
            self.discrimination = self.Agent_normalizeParam(discrimination)
            return

        # "Resets" the clock for concealed discrimination
        self.hasMultipleStagnant = False    

        discrimination = 1 - (numPolicies/self.network.policyCap + avgAttitude)
        discrimination *= SCALE_FACTOR
        self.discrimination = self.Agent_normalizeParam(discrimination)

    #################################################################
    # Given an agent, updates his concealment, based on the network #
    # sentiments and local support. Probabilistically determines if #
    # agent becomes concealed or not. Note: If an agent becomes     #
    # concealed in the simulation, he can later unconceal himself or#
    # vice versa                                                    #
    #################################################################
    def Agent_updateConcealment(self, discriminateConcealImpact):
        SCALE_FACTOR = 2.0

        numPolicies = self.network.policyScore
        probConceal = ((self.discrimination - self.support) * \
            discriminateConcealImpact - numPolicies/self.network.policyCap) \
            * SCALE_FACTOR + self.network.NetworkBase_getNetworkAttitude()

        self.probConceal = self.Agent_getLogistic(probConceal)
        
        rand = random.random()
        self.isConcealed = (rand < self.probConceal)

    #################################################################
    # Given an agent, updates his depression status, based on the   #
    # local and network settings. Note: If agent becomes depressed  #
    # the property remains for the duration of the simulation (can't#
    # become 'undepressed')                                         #
    #################################################################
    def Agent_updateDepression(self, concealDepressionImpact, time):
        SCALING_FACTOR = .065

        # Ignores those probabilities that are sufficiently small
        DEPRESSION_THRESHOLD = .025

        # Number of time intervals before which a reversal of 
        # depressive condition can disappear
        TIME_THRESHOLD = 20

        if self.isDepressed:
            if (time - self.depressStart > TIME_THRESHOLD):
                rand = random.random()
                self.isDepressed = (rand < (1 - self.currentDepression/2))
            return

        numPolicies = self.network.policyScore
        probIncrease = self.discrimination - self.support
        probIncrease -= numPolicies/25
        probIncrease += self.network.NetworkBase_getNetworkAttitude()

        if self.isConcealed:
            probIncrease *= concealDepressionImpact

        baseProb = self.currentDepression + probIncrease

        # Uses logit scale
        self.currentDepression = self.Agent_getLogistic(baseProb) \
            * SCALING_FACTOR

        rand = random.random()
        self.isDepressed = (rand < self.currentDepression and \
            self.currentDepression > DEPRESSION_THRESHOLD)
        if self.isDepressed:
            self.depressStart = time