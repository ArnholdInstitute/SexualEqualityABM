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

        # Accounts for those who reflect feeling uneasy when with 
        # those of the sexual minority community
        delta = .75 * percentConnect
        if self.isDiscriminatory:
            self.attitude = self.minorityAttitude - delta
        else:
            self.attitude = self.minorityAttitude + delta

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
    def Agent_updateDiscrimination(self):
        return

    #################################################################
    # As concealment only applies to minorities, no update is needed#
    #################################################################
    def Agent_updateConcealment(self):
        return

    #################################################################
    # Simply due to the results of interest for this investigation  #
    # we do not consider the depression of those agents not present #
    # in the minority                                               #
    #################################################################
    def Agent_updateDepression(self):
        return

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
        ADDITIONAL_BOOST = .25
        att = self.network.NetworkBase_getNetworkAttitude()
        localConnect = self.network.\
            NetworkBase_findPercentConnectedMinority(self)

        # Accounts for additional boost felt when those opposing are
        # in significant minority
        const = 0
        if att > .75:
            const = ADDITIONAL_BOOST

        self.support = localConnect * att + const
        self.support = self.Agent_normalizeParam(self.support)

    #################################################################
    # Given an agent, updates his discrimination, based on whether  #
    # or not he is concealed and the overall network sentiments     #
    # towards minorities, expressed through the presence of policies#
    # and attitudes                                                 #
    #################################################################
    def Agent_updateDiscrimination(self):
        numPolicies = self.network.policyScore
        avgAttitude = self.network.NetworkBase_getLocalAvg(self, \
            "attitude")

        self.discrimination = 1 - (numPolicies/25 + avgAttitude)
        self.discrimination = \
            self.Agent_normalizeParam(self.discrimination)

    #################################################################
    # Given an agent, updates his concealment, based on the network #
    # sentiments and local support. Probabilistically determines if #
    # agent becomes concealed or not. Note: If an agent becomes     #
    # concealed in the simulation, he can later unconceal himself or#
    # vice versa                                                    #
    #################################################################
    def Agent_updateConcealment(self):
        numPolicies = self.network.policyScore
        probConceal = self.discrimination/self.support - numPolicies/25

        self.probConceal = \
            self.Agent_normalizeParam(self.probConceal)

        rand = random.random()
        self.isConcealed = False
        
        if rand < self.probConceal:
            self.isConcealed = True

    #################################################################
    # Given an agent, updates his depression status, based on the   #
    # local and network settings. Note: If agent becomes depressed  #
    # the property remains for the duration of the simulation (can't#
    # become 'undepressed')                                         #
    #################################################################
    def Agent_updateDepression(self):
        if self.isDepressed:
            return

        DEPRESS_CONST = .15
        probIncrease = DEPRESS_CONST * self.discrimination/(self.support)
        if self.isConcealed:
            probIncrease *= 5.0

        self.currentDepression += probIncrease

        rand = random.random()
        self.isDepressed = (rand < self.currentDepression)