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
from Switch import switch

import matplotlib.pyplot as plt
from operator import itemgetter 
from math import exp

try:
    import networkx as nx
except ImportError:
    raise ImportError("You must install NetworkX:\
    (http://networkx.lanl.gov/) for SE simulation")

#####################################################################
# A policy being considered in the simulation: initialized to not   #
# be considered "passed" and given an "influence" score - determines#
# how much change it will bring if passed. The higher the score, the#
# more difficult it is for the policy to pass. The "time" denotes   #
# when the bill was originally proposed and (possibly) passed. A    #
# score for the policy may also be supplied (defaulted to not being #
# the case) in which case a policy with that score will be given.   #
# The biasPass allows for selective production of bills, namely with#
# 0 specifying all bills are possibly, 2 being only discriminatory  #
# and 1 being only non-discriminatory                               #
#####################################################################
class Policy:
    def __init__(self, time, score=None, biasPass=0):
        IRRELEVANT = 0
        ONLY_NON_DISCRIMINATORY = 1
        ONLY_DISCRIMINATORY = 2

        # Probabilistically finds the score: less likely to get large
        # magnitudes of scores
        if not score:
            for case in switch(biasPass):
                if case(IRRELEVANT):
                    self.score = int(np.random.normal(0, 3))
                    while (self.score == 0 or self.score > 5 or
                        self.score < -5):
                        self.score = int(np.random.normal(0, 3))
                    break

                if case(ONLY_NON_DISCRIMINATORY):
                    self.score = int(np.random.normal(2.5, 1.5))
                    while (self.score <= 0 or self.score > 5):
                        self.score = int(np.random.normal(2.5, 1.5))
                    break

                if case(ONLY_DISCRIMINATORY):
                    self.score = int(np.random.normal(-2.5, 1.5))
                    while (self.score >= 0 or self.score < -5):
                        self.score = int(np.random.normal(-2.5, 1.5))
                    break

                if case():
                    sys.stderr.write("Invalid value for bias")
                    return False

        else:
            self.score = score

        # If bill has score < 0: hurts LGB sentiments
        self.isDiscriminatory = (self.score < 0)
        self.isPassed = False

        # Used to account for the time delay of the policy effect
        self.passTime = time
        self.prevEffect = 0
        self.curEffect = 0

        if not self.Policy_verifyPolicy(self.score, self.isPassed, 
            self.passTime):
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
    def Policy_verifyPolicy(self, score, isPassed, time):
        if not Verification_verifyInt(score, "Score"):
            return False

        if not Verification_verifyBool(isPassed, "isPassed"):
            return False

        if not Verification_verifyInt(time, "Time"):
            return False

        return True

    #################################################################
    # Determines the probability of a policy to pass, given the     #
    # network being considered, based on the "acceptance" by the    #
    # population and the bill's influence score                     #
    #################################################################
    def Policy_getProbability(self, network, policyCap):
        # Ensures that the score does not exceed max/min 
        finalScore = network.potentialScore + self.score
        if finalScore > policyCap:
            return 0.0
        elif finalScore < -policyCap:
            return 0.0

        attitudeFor = network.\
            NetworkBase_getTotalInfluence(abs(self.score))
        possibleFor = network.NetworkBase_getMaxTotalInfluence()

        if self.isDiscriminatory:
            attitudeFor *= -1

        return attitudeFor/possibleFor

    #################################################################
    # Determines, based on the initial time of passing, the extent  #
    # to which a bill's "effects" have been experienced             #
    #################################################################
    def Policy_updateTimeEffect(self, time, policyCap):
        # Only resets prevEffect if curEffect has been calculated at
        # least once
        if self.curEffect:
            self.prevEffect = self.curEffect
        
        DISC_FACTOR = 1
        ADD_FACTOR = 1

        if self.isDiscriminatory: 
            DISC_FACTOR = -1
            ADD_FACTOR = 0

        deltaTime = time - self.passTime
        rating = self.score

        self.curEffect = int(rating * (1 - exp(-DISC_FACTOR * \
            (policyCap * deltaTime)/rating))) + ADD_FACTOR


    #################################################################
    # Passes or rejects a policy for the network under question     #
    #################################################################
    def Policy_considerPolicy(self, network, time, policyCap):
        probAdd = self.Policy_getProbability(network, policyCap)

        rand = random.random()
        if rand < probAdd:
            self.isPassed = True
            network.NetworkBase_addToPolicies(self, time)