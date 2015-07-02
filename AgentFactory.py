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
    def AgentFactory_createAgent(network, agentID):
        # Constant values (can change for variation in simulation)
        PROB_MINORITY = .25
        BASELINE_ATTITUDE = .05

        BASELINE_DISCRIMINATION = .025
        NO_DISCRIMINATION = 0.0

        BASELINE_SUPPORT = .75
        FULL_SUPPORT = 1.0

        FULL_ACCEPTANCE = 1.0

        NON_MINORITY_DEPRESS = 0.0
        UNCONCEAL_DEPRESS_PROB = .025
        CONCEAL_DEPRESS_PROB = .050
        
        CENTER_SES_RAND = 3
        BASELINE_SES = .1

        rand = random.random()
        isMinority = False
        if rand <= PROB_MINORITY:
            isMinority = True

        currentSES = np.random.poisson(CENTER_SES_RAND)/10 + BASELINE_SES

        # Normalizes SES to 1.0 scale
        if currentSES > 1.0:
            currentSES = 1.0
        elif currentSES < 0.0:
            currentSES = 0.0

        # No discrimination imposed upon those not of minority.
        # Similarly, attitude for those of minority is fully accepting
        # of one another and probabilistic otherwise. Support is also
        # fully present for those not of minority status
        if not isMinority:
            discrimination = NO_DISCRIMINATION
            minorityAttitude = random.random() * BASELINE_ATTITUDE
            support = FULL_SUPPORT
        else:
            discrimination = random.random() * BASELINE_DISCRIMINATION
            minorityAttitude = FULL_ACCEPTANCE
            support = random.random() * BASELINE_SUPPORT

        SCALING_FACTOR = .50
        if not isMinority:
            probConceal = 0
        else:
            discrepancy = discrimination - support
            probConceal = 1/(1 + math.exp(discrepancy)) * SCALING_FACTOR 

        # For simplicity in network calculations, assumed to be false
        # if the person is not of sexual minority
        rand = random.random()
        isConcealed = rand < probConceal and isMinority

        if not isMinority:
            const = NON_MINORITY_DEPRESS
        else:
            const = UNCONCEAL_DEPRESS_PROB
            if isConcealed:
                const = CONCEAL_DEPRESS_PROB

        rand = random.random()
        rand = random.random()
        currentDepression = rand * const

        rand = random.random()
        isDepressed = rand < currentDepression

        if isMinority:
            agent = MinorityAgent(currentSES, minorityAttitude, isMinority,
                discrimination, support, isConcealed, probConceal, 
                currentDepression, isDepressed, network, agentID)
        else: 
            agent = NonMinorityAgent(currentSES, minorityAttitude, isMinority,
                discrimination, support, isConcealed, probConceal, 
                currentDepression, isDepressed, network, agentID)
        return agent