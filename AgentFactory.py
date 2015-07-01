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
        PROB_MINORITY = 1.00
        PROB_CONCEAL = .10
        BASELINE_ATTITUDE = .25

        NO_DISCRIMINATION = 0.0
        BASELINE_DISCRIMINATION = .25

        FULL_SUPPORT = 1.0
        BASELINE_SUPPORT = .50 
        FULL_ACCEPTANCE = 1.0

        NON_MINORITY_DEPRESS = .010
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

        # For simplicity in network calculations, assumed to be false
        # if the person is not of sexual minority
        rand = random.random()
        if not isMinority:
            isConcealed = False
        elif rand < PROB_CONCEAL:
            isConcealed = True
        else: 
            isConcealed = False

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
        isDepressed = False
        if rand < currentDepression:
            isDepressed = True

        if isMinority:
            agent = MinorityAgent(currentSES, minorityAttitude, isMinority,
                discrimination, support, isConcealed, currentDepression, 
                isDepressed, network, agentID)
        else: 
            agent = NonMinorityAgent(currentSES, minorityAttitude, isMinority,
                discrimination, support, isConcealed, currentDepression, 
                isDepressed, network, agentID)
        return agent