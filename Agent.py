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
            NetworkBase_findPercentConnectedMinority(self, allSupport=True)

        # Accounts for negative sentiments diffusing in network
        percentPoorNonAccept = self.network.\
            NetworkBase_findPercentNonAccepting(self)

        # Accounts for those who reflect feeling uneasy when with 
        # those of the sexual minority community
        deltaMinority = .75 * percentConnect/self.network.policyCap
        deltaNonMinority = .025 * percentPoorNonAccept/self.network.policyCap
        
        if self.isDiscriminatory:
            self.attitude -= deltaMinority
        else:
            self.attitude += deltaMinority
        self.attitude -= deltaNonMinority

    #################################################################
    # As those not of minorities are assumed to have full support,  #
    # no update is needed                                           #
    #################################################################
    def Agent_updateSupport(self):
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
    def Agent_updateConcealment(self, discriminateConcealImpact,
        supportConcealImpact, time):
        return

    #################################################################
    # Simply due to the results of interest for this investigation  #
    # we consider the depression of the non-minority members to     #
    # perform sensitivity analysis and contrast with literature     #
    #################################################################
    def Agent_updateDepression(self, concealImpact, supportDepressionImpact,
        discriminateDepressionImpact, time):
        DEPRESSION_THRESHOLD = .025
        SCALING_FACTOR = .025
        TIME_DECAY = .875
        FINAL_SCALE = .0075

        # Number of time intervals before which a reversal of 
        # depressive condition can disappear
        TIME_THRESHOLD = 20

        if self.isDepressed:
            if (time - self.depressStart > TIME_THRESHOLD):
                rand = random.random()
                self.isDepressed = (rand < (1 - self.currentDepression/2))
            return

        self.baseDepression *= TIME_DECAY
        baseProb = self.baseDepression
        if self.isDiscriminatory:
            baseProb = self.baseDepression + self.network.\
                NetworkBase_findPercentConnectedMinority(self) * SCALING_FACTOR

        self.currentDepression = self.Agent_getLogistic(baseProb) \
            * FINAL_SCALE

        rand = random.random()
        self.isDepressed = (rand < self.currentDepression and \
            self.currentDepression > DEPRESSION_THRESHOLD)
        if self.isDepressed:
            self.depressStart = time

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
    def Agent_updateSupport(self):
        ADDITIONAL_BOOST = .50
        BASELINE_SUPPORT = .05

        att = self.network.NetworkBase_getNetworkAttitude()
        localConnect = self.network.\
            NetworkBase_findPercentConnectedMinority(self)

        # Accounts for additional boost felt when those opposing are
        # in significant minority
        const = 1.00
        if att > .75:
            const += ADDITIONAL_BOOST

        support = BASELINE_SUPPORT

        support += localConnect ** .25 + att ** 2 * const
        self.support = self.Agent_normalizeParam(support)

    #################################################################
    # Given an agent, updates his discrimination, based on whether  #
    # or not he is concealed and the overall network sentiments     #
    # towards minorities, expressed through the presence of policies#
    # and attitudes                                                 #
    #################################################################
    def Agent_updateDiscrimination(self, time, concealDiscriminateImpact):
        SCALE_FACTOR = .125
        CONCEAL_DECAY = .60

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
                discrimination = 1 - (numPolicies/self.network.policyCap \
                    + avgAttitude)
                return

            deltaTime = time - self.time
            discrimination = 1 - (numPolicies/self.network.policyCap       \
                            + (self.initialPositive + self.initialNegative \
                            * concealDiscriminateImpact ** (-deltaTime)))  \
                            - (self.probConceal ** .75) * 5

            discrimination *= CONCEAL_DECAY ** deltaTime
            discrimination *= SCALE_FACTOR 
            
            self.discrimination = self.Agent_normalizeParam(discrimination)
            return

        # "Resets" the clock for concealed discrimination
        self.hasMultipleStagnant = False
        discrimination = 1 - (numPolicies/self.network.policyCap + avgAttitude) \
            - (self.probConceal  ** .75) * 5
            
        discrimination *= SCALE_FACTOR
        self.discrimination = self.Agent_normalizeParam(discrimination)

    #################################################################
    # Given an agent, updates his concealment, based on the network #
    # sentiments and local support. Probabilistically determines if #
    # agent becomes concealed or not. Note: If an agent becomes     #
    # concealed in the simulation, he can later unconceal himself or#
    # vice versa                                                    #
    #################################################################
    def Agent_updateConcealment(self, discriminateConcealImpact,
        supportConcealImpact, time):
        SCALE_FACTOR = .675
        FULL_DEPRESS_FACTOR = 3.0
        DEPRESS_FACTOR = 15.0 

        NETWORK_SCALE = .75
        FINAL_SCALE = .0125

        # Number of time intervals before which a reversal of 
        # depressive condition can disappear
        TIME_THRESHOLD = 5

        numPolicies = self.network.policyScore
        probConceal = (self.discrimination ** 1.5 * discriminateConcealImpact   \
            - self.support * supportConcealImpact)
        probConceal -= (numPolicies/self.network.policyCap) ** 3
        probConceal *= SCALE_FACTOR 
        probConceal -= self.network.NetworkBase_getNetworkAttitude() \
            * NETWORK_SCALE

        # Significant increase if depression has actually happened
        if self.isDepressed:
            if probConceal > 0:
                probConceal *= FULL_DEPRESS_FACTOR
        else:
            depressFactor = .50 + (DEPRESS_FACTOR * \
                self.currentDepression) ** 2
            probConceal *= depressFactor

        self.probConceal = (self.Agent_getLogistic(probConceal) ** 2)/3

        # Agents will not alternate between concealed/unconcealed rapidly
        if self.isConcealed:
            if (time - self.concealStart > TIME_THRESHOLD):
                rand = random.random()
                self.isConcealed = (rand < ((1 - self.probConceal/2) \
                    * FINAL_SCALE))
            return
        
        rand = random.random()
        self.isConcealed = (rand < self.probConceal)
        if self.isConcealed:
            self.concealStart = time

    #################################################################
    # Given an agent, updates his depression status, based on the   #
    # local and network settings. Note: If agent becomes depressed  #
    # the property remains for the duration of the simulation (can't#
    # become 'undepressed')                                         #
    #################################################################
    def Agent_updateDepression(self, concealDepressionImpact, 
        supportDepressionImpact, discriminateDepressionImpact, time):
        SCALING_FACTOR = .075
        CONCEAL_FACTOR = 1.375

        # Ignores those probabilities that are sufficiently small
        DEPRESSION_THRESHOLD = .025

        # Number of time intervals before which a reversal of 
        # depressive condition can disappear
        TIME_THRESHOLD = 20

        if self.isDepressed:
            if (time - self.depressStart > TIME_THRESHOLD):
                rand = random.random()
                self.isDepressed = (rand < ((1 - self.currentDepression/2) \
                    * SCALING_FACTOR)) 
            return

        numPolicies = self.network.policyScore

        probIncrease = (self.discrimination ** .25) * discriminateDepressionImpact
        probIncrease -= self.support * supportDepressionImpact
        probIncrease -= (numPolicies/self.network.policyCap) ** 3
        probIncrease -= self.network.NetworkBase_getNetworkAttitude()

        # Significant bump if agent is already concealed
        if self.isConcealed:
            if probIncrease > 0:
                probIncrease *= concealDepressionImpact
        else:
            concealFactor = .25 + (CONCEAL_FACTOR * self.probConceal) ** 2
            probIncrease *= concealFactor

        baseProb = self.currentDepression + probIncrease

        # Uses logit scale
        self.currentDepression = (self.Agent_getLogistic(baseProb) ** 2)/4

        rand = random.random()
        self.isDepressed = (rand < self.currentDepression and \
            self.currentDepression > DEPRESSION_THRESHOLD)
        if self.isDepressed:
            self.depressStart = time