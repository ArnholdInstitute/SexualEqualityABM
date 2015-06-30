#####################################################################
# Name: Yash Patel                                                  #
# File: Policy.py                                                   #
# Description: Object file containing all the methods modelling the #
# policies and their introduction into the network. Policies are    #
# given through time updates in the simulation and affect the global#
# sentiments toward sexual minorities                               #
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
# A policy being considered in the simulation: initialized to not   #
# be considered "passed" and given an "influence" score - determines#
# how much change it will bring if passed. The higher the score, the#
# more difficult it is for the policy to pass                       #
#####################################################################
class Policy:
    def __init__(self):
        self.score = int(np.random.normal(0, 3))
        while self.score == 0 or self.score > 10 or self.score < -10:
            self.score = int(np.random.normal(0, 3))

        self.isPassed = False

        if not self.Agent_verifyAgent(self.score, self.isPassed):
            return None

    #################################################################
    # Provides an output string for printing out policies           #
    #################################################################
    def __str__(self):
        return "Score: {}, Passed: {}".replace(self.score, self.isPassed)

    #################################################################
    # Checks that, given all the parameters used to initialize the  #
    # policy, the parameters are legal                              #
    #################################################################
    def Agent_verifyAgent(self, score, isPassed):
        if not Verification_verifyInt(score, "Score"):
            return False

        if not Verification_verifyBool(isPassed, "isPassed"):
            return False

        return True

    #################################################################
    # Determines the probability of a policy to pass, given the     #
    # network being considered, based on the "acceptance" by the    #
    # population and the bill's influence score                     #
    #################################################################
    def Policy_getProbability(self, network):
        MIN_POLICY = -500
        MAX_POLICY = 500

        finalScore = network.policyScore + self.score 
        if finalScore > MAX_POLICY:
            return 0.0
        elif finalScore < MIN_POLICY:
            return 0.0

        attitudeFor = network.\
            NetworkBase_getTotalInfluence(abs(self.score))
        possibleFor = network.NetworkBase_getMaxTotalInfluence()
        return attitudeFor/possibleFor

    #################################################################
    # Passes or rejects a policy for the network under question     #
    #################################################################
    def Policy_considerPolicy(self, network):
        probAdd = self.Policy_getProbability(network)

        rand = random.random()
        if rand < probAdd:
            self.isPassed = True
            network.NetworkBase_addToPolicies(self)