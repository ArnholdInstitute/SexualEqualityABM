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
		self.score = int(np.random.normal(5, 2))
		if score < 1:
			score = 1
		elif score > 10:
			score = 10

		self.isPassed = False

	#################################################################
    # Determines the probability of a policy to pass, given the     #
    # network being considered, based on the "acceptance" by the    #
    # population and the bill's influence score                     #
    #################################################################
	def Policy_getProbability(self, network):
		attitudeFor = network.NetworkBase_getTotalInfluence()
		possibleFor = network.\
			NetworkBase_getMaxTotalInfluence(self.billRank)
		return attitudeFor/possibleFor

	#################################################################
    # Passes or rejects a policy for the network under question     #
    #################################################################
	def Policy_considerPolicy(self, network):
		probAdd = self.Policy_getProbability(network)
		rand = random.random()
		if rand < probAdd:
			policy.isPassed = True
			network.NetworkBase_addToPolicies(self)