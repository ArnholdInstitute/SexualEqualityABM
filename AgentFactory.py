#####################################################################
# Name: Yash Patel                                                  #
# File: AgentFactory.py                                             #
# Description: Object file for agent factory: used to produce the   #
# agents present in the simulation (both minority and non)          #
#####################################################################

import sys
import os
import math
import random
import numpy as np

from Agent import MinorityAgent, NonMinorityAgent

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
    def AgentFactory_createAgent(network, agentID, percentMinority,
        attitude_0=None, support_0=None, discrimination_0=None, 
        conceal_0=None, depression_0=None, policyScore_0=None):

        # Structure that has, for each key, the associated passed in 
        # initial value and the default initial value (if None passed).
        # Conceal's default value is specified as None since its value is
        # determined as a function of the other specified ones (default)
        SCALING_FACTOR = .025 * (2.0 - percentMinority)
        initialVals = {
            "attitude": [attitude_0, (random.random() - .5) * .75],
            "support": [support_0, random.random() * .75],
            "discrimination": [discrimination_0, random.random() * .025],
            "conceal": [conceal_0, lambda: 1/(1 + math.exp(discrimination 
                - support)) * SCALING_FACTOR],
            "depression": [depression_0, None],
            "policyScore": [policyScore_0, 0]
        }

        DEFAULT_INDEX = 2
        for value in initialVals:
            givenInit = initialVals[value][0]
            defaultInit = initialVals[value][1]
            if givenInit is not None:
                initialVals[value].append(givenInit)
            else: 
                initialVals[value].append(defaultInit)

        attitude = initialVals["attitude"][DEFAULT_INDEX]
        support = initialVals["support"][DEFAULT_INDEX]
        discrimination = initialVals["discrimination"][DEFAULT_INDEX]
        probConceal = initialVals["conceal"][DEFAULT_INDEX]
        # Used in the case that probConceal has default lambda behavior
        if hasattr(probConceal, '__call__'): 
            probConceal = probConceal()
        currentDepression = initialVals["depression"][DEFAULT_INDEX]
        policyScore = initialVals["policyScore"][DEFAULT_INDEX]
        
        NO_DISCRIMINATION = 0.0
        NO_CONCEALMENT = 0.0

        FULL_SUPPORT = 1.0
        FULL_ACCEPTANCE = 1.0
        
        CENTER_SES_RAND = 3
        BASELINE_SES = .1

        PROB_DEPRESS_MULTIPLIER = 3.0
        CONCEAL_DEPRESS_MULT = 2.0
        UNCONCEAL_DEPRESS_PROB = .0035

        isMinority = (random.random() < percentMinority)
        currentSES = np.random.poisson(CENTER_SES_RAND)/10 + BASELINE_SES

        # Normalizes SES to 1.0 scale
        if currentSES > 1.0: currentSES = 1.0
        elif currentSES < 0.0: currentSES = 0.0

        # No discrimination imposed upon those not of minority.
        # Attitude for those of minority is fully accepting
        # of one another and probabilistic otherwise. Support is also
        # fully present for those not of minority status
        if not isMinority:
            discrimination = NO_DISCRIMINATION
            support = FULL_SUPPORT
        else:
            attitude = FULL_ACCEPTANCE

        if not isMinority: 
            probConceal = NO_CONCEALMENT

        # For simplicity in network calculations, assumed to be false
        # if the person is not of sexual minority
        isConcealed = random.random() < probConceal and isMinority

        if not isMinority:
            probDepress = (1 - PROB_DEPRESS_MULTIPLIER * currentSES)/8
            if probDepress < 0.0: probDepress = 0.0
            currentDepression = random.random() * probDepress
        elif currentDepression is None:
            probDepress = UNCONCEAL_DEPRESS_PROB
            if isConcealed: probDepress *= CONCEAL_DEPRESS_MULT

            # More likely to start depressed if less minority
            probDepress *= (2.0 - percentMinority)
            currentDepression = random.random() * probDepress

        isDepressed = random.random() < currentDepression

        if isMinority:
            agent = MinorityAgent(currentSES, attitude, isMinority,
                discrimination, support, isConcealed, probConceal, 
                currentDepression, isDepressed, network, policyScore, agentID)
        else: 
            agent = NonMinorityAgent(currentSES, attitude, isMinority,
                discrimination, support, isConcealed, probConceal, 
                currentDepression, isDepressed, network, policyScore, agentID)
        return agent