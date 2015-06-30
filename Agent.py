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
        self.attitude = self.minorityAttitude + .75 * percentConnect

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
        numPolicies = self.network.policyScore
        totalInfluence = self.network.NetworkBase_getTotalInfluence(1)
        maxInfluence = self.network.NetworkBase_getMaxTotalInfluence()

        concealment = 1.0
        if self.isConcealed:
            concealment *= 2.0

        self.discrimination = concealment * (1 - ((numPolicies)/525 \
            + totalInfluence/maxInfluence)/2)

    #################################################################
    # Given an agent, updates his concealment, based on the network #
    # sentiments and local support. Probabilistically determines if #
    # agent becomes concealed or not. Note: If an agent becomes     #
    # concealed in the simulation, he can later unconceal himself or#
    # vice versa                                                    #
    #################################################################
    def Agent_updateConcealment(self):
        SCALE_FACTOR = 100

        numPolicies = self.network.policyScore
        probConceal = self.discrimination/(self.support * \
            numPolicies/525)/SCALE_FACTOR

        rand = random.random()

        self.isConcealed = False
        if rand < probConceal:
            self.isConcealed = True