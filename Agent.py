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
        deltaMinority = percentConnect/self.network.policyCap
        deltaNonMinority = .175 * percentPoorNonAccept/self.network.policyCap
        
        if self.isDiscriminatory: self.attitude -= deltaMinority
        else: self.attitude += deltaMinority
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
        DISCRIMINATE_SUPPORT_IMPACT = .125

        avgAttitude = self.network.NetworkBase_getNetworkAttitude()
        localConnect = self.network.\
            NetworkBase_findPercentConnectedMinority(self)
        
        numPolicies = self.network.policyScore

        # Accounts for additional boost felt when those opposing are
        # in significant minority
        supportBoost = 1.00 + int(avgAttitude > .75) * ADDITIONAL_BOOST
        
        support = numPolicies/self.network.policyCap
        support += (avgAttitude * supportBoost)
        
        self.support += (self.Agent_getLogistic(support) ** 3)/50
        self.support -= self.discrimination * DISCRIMINATE_SUPPORT_IMPACT

    #################################################################
    # Given an agent, updates his discrimination, based on whether  #
    # or not he is concealed and the overall network sentiments     #
    # towards minorities, expressed through the presence of policies#
    # and attitudes                                                 #
    #################################################################
    def Agent_updateDiscrimination(self, time, concealDiscriminateImpact):
        numPolicies = self.network.policyScore
        avgAttitude = self.network.NetworkBase_getLocalAvg(self, \
            "attitude")
        SUPPORT_DISCRIMINATE_IMPACT = 5.0
        
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

            discrimination = 1 - (numPolicies/self.network.policyCap      
                            + (self.initialPositive + self.initialNegative 
                            * concealDiscriminateImpact ** (-deltaTime))) * 10
            discrimination -= self.support * SUPPORT_DISCRIMINATE_IMPACT

            self.discrimination += self.Agent_getLogistic(discrimination)/100
            return

        # "Resets" the clock for concealed discrimination
        self.hasMultipleStagnant = False
        discrimination = 1 - (numPolicies/self.network.policyCap + avgAttitude) * 10
        discrimination -= self.support * SUPPORT_DISCRIMINATE_IMPACT
        
        self.discrimination += self.Agent_getLogistic(discrimination)/100

    #################################################################
    # Given an agent, updates his concealment, based on the network #
    # sentiments and local support. Probabilistically determines if #
    # agent becomes concealed or not. Note: If an agent becomes     #
    # concealed in the simulation, he can later unconceal himself or#
    # vice versa                                                    #
    #################################################################
    def Agent_updateConcealment(self, discriminateConcealImpact,
        supportConcealImpact, time):
        DEPRESS_FACTOR = 1.025 
        FINAL_SCALE = .0125

        NETWORK_IMPACT = .25

        # Number of time intervals before which a reversal of 
        # depressive condition can disappear
        TIME_THRESHOLD = 5
        
        rand = random.random()
        self.isConcealed = (rand < self.probConceal)

        if self.isConcealed: 
            self.concealStart = time

        # Agents will not alternate between concealed/unconcealed rapidly
        if self.isConcealed:
            if (time - self.concealStart > TIME_THRESHOLD):
                rand = random.random()
                self.isConcealed = (rand < ((1 - self.probConceal/2) \
                    * FINAL_SCALE))
            return

        numPolicies = self.network.policyScore
        probConceal = (self.discrimination * discriminateConcealImpact   
            - self.support * supportConcealImpact)
        probConceal -= numPolicies/self.network.policyCap * NETWORK_IMPACT
        probConceal -= self.network.NetworkBase_getNetworkAttitude()

        self.probConceal += (self.Agent_getLogistic(probConceal) ** 3)/100
        
        # Significant increase if depression has actually happened
        if self.isDepressed:
            self.probConceal *= DEPRESS_FACTOR

    #################################################################
    # Given an agent, updates his depression status, based on the   #
    # local and network settings. Note: If agent becomes depressed  #
    # the property remains for the duration of the simulation (can't#
    # become 'undepressed')                                         #
    #################################################################
    def Agent_updateDepression(self, concealDepressionImpact, 
        supportDepressionImpact, discriminateDepressionImpact, time):
        SCALING_FACTOR = .075
        
        # Ignores those probabilities that are sufficiently small
        DEPRESSION_THRESHOLD = .025

        # Number of time intervals before which a reversal of 
        # depressive condition can disappear
        TIME_THRESHOLD = 20
        NETWORK_IMPACT = .25

        rand = random.random()
        self.isDepressed = (rand < self.currentDepression and \
            self.currentDepression > DEPRESSION_THRESHOLD)
        
        if self.isDepressed:
            self.depressStart = time

        if self.isDepressed:
            if (time - self.depressStart > TIME_THRESHOLD):
                rand = random.random()
                self.isDepressed = (rand < ((1 - self.currentDepression/2) 
                    * SCALING_FACTOR)) 
            return

        numPolicies = self.network.policyScore

        probIncrease = self.discrimination * discriminateDepressionImpact
        probIncrease -= self.support  * supportDepressionImpact
        probIncrease -= numPolicies/self.network.policyCap * NETWORK_IMPACT
        probIncrease -= self.network.NetworkBase_getNetworkAttitude()

        # Uses logit scale
        self.currentDepression += (self.Agent_getLogistic(probIncrease) ** 3)/100000

        # Significant bump if agent is already concealed
        if self.isConcealed:
            self.currentDepression *= concealDepressionImpact