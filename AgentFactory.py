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
        PROB_MINORITY = .50
        PROB_CONCEAL = .10
        BASELINE_ATTITUDE = .25

        NO_DISCRIMINATION = 0.0
        BASELINE_DISCRIMINATION = .25

        FULL_SUPPORT = 1.0
        BASELINE_SUPPORT = .50 
        FULL_ACCEPTANCE = 1.0

        NON_MINORITY_DEPRESS = .00010
        UNCONCEAL_DEPRESS_PROB = .00025
        CONCEAL_DEPRESS_PROB = .00050
        
        CENTER_SES_RAND = 3
        BASELINE_SES = .1

        rand = random.random()
        isMinority = False
        if rand <= PROB_MINORITY:
            isMinority = True

        SESarr = []

        childSES = np.random.poisson(CENTER_SES_RAND)/10 + BASELINE_SES
        oldSES = np.random.poisson(CENTER_SES_RAND)/10 + BASELINE_SES
        currentSES = np.random.poisson(CENTER_SES_RAND)/10 + BASELINE_SES

        # Normalizes SES to 1.0 scale
        SESarr.append(childSES)
        SESarr.append(oldSES)
        SESarr.append(currentSES)

        for i in range(0, len(SESarr)):
            if SESarr[i] > 1.0:
                SESarr[i] = 1.0
            elif SESarr[i] < 0.0:
                SESarr[i] = 0.0

        childSES = SESarr[0]
        oldSES = SESarr[1]
        currentSES = SESarr[2]

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
        oldDepression = rand * const
        rand = random.random()
        currentDepression = rand * const

        rand = random.random()
        isDepressed = False
        if rand < currentDepression:
            isDepressed = True

        if isMinority:
            agent = MinorityAgent(childSES, oldSES, currentSES, 
                minorityAttitude, isMinority, discrimination, support, 
                isConcealed, oldDepression, currentDepression, 
                isDepressed, network, agentID)
        else: 
            agent = NonMinorityAgent(childSES, oldSES, currentSES, 
                minorityAttitude, isMinority, discrimination, support, 
                isConcealed, oldDepression, currentDepression, 
                isDepressed, network, agentID)
        return agent