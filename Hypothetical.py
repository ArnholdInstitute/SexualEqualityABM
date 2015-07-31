#####################################################################
# Name: Yash Patel                                                  #
# File: Hypothetical.py                                             #
# Description: Contains all the hypothetical situations that were of#
# interest to the case in study. We wished to investigate the       #
# relative effectiveness of each of several routes to improve LGB   #
# treatment/situation in the community, with the specifics of each  #
# method given below                                                #
#####################################################################

import sys
import os
from copy import deepcopy

from NetworkBase import NetworkBase
from SexMinDepressionSimulation import *

#####################################################################
# Defines the generic structure of an intervention (used as template#
# for all interventions modelled below). Original is the original   #
# simulation model that was run, attr the attribute to be varied,   #
# effectiveness being how much of an impact the intervention had,   #
# and params being the parameters used for simulation, in the form  #
# of (attitude, support, discrimination, conceal, depression,       # 
# enforcedPolicy), a tuple                                          #
#####################################################################
def Hypothetical_genericTest(original, attr, effectiveness, paramsDict):
    origAttr = results[attr]
    changedAttr = origAttr * effectiveness 
    paramsDict[attr] = changedAttr

    # Converts from the given dictionary format to a tuple
    params = map(lambda v: paramsDict[v], paramsDict)

    concealTest = deepcopy(original)
    concealTest.SMDModel_runStreamlineSimulation(*list(params))
    return concealTest

#####################################################################
# Scenario 1: Intervene on LGB individuals to improve their mental  #
# health. In our model, this intervention would reduce concealment  #
# without intervening on other variables. Returns the final state of# 
# the simulation                                                    #
#####################################################################
def Hypothetical_LGB_Concealment(original, paramsDict):
    INTERVENTION_EFFECTIVESS = .75
    attr = "concealed"
    return Hypothetical_genericTest(original, attr, 
        INTERVENTION_EFFECTIVESS, paramsDict)

#####################################################################
# Scenario 2: intervene on non-LGB individuals (parents, peers) to  #
# reduce rejection and victimization - so in our model, this        #
# intervention would reduce victimization/interpersonal             #
# discrimination without intervening on other variables. Returns the#
# final state of the simulation                                     #
#####################################################################
def Hypothetical_NonLGB_Discrimination(original, paramsDict):
    INTERVENTION_EFFECTIVESS = .75
    attr = "discrimination"
    return Hypothetical_genericTest(original, attr, 
        INTERVENTION_EFFECTIVESS, paramsDict)

#####################################################################
# Scenario 3: intervene on non-LGB individuals to improve their     #
# attitudes and reduce prejudice and stigma (e.g., the "It Gets     #
# Better" media campaign) - so in our model, this intervention would#
# improve attitudes without intervening on other variables. Returns #
# the final state of the simulation                                 #
#####################################################################
def Hypothetical_NonLGB_Attitudes(original, paramsDict):
    INTERVENTION_EFFECTIVESS = 1.25
    attr = "attitude"
    return Hypothetical_genericTest(original, attr, 
        INTERVENTION_EFFECTIVESS, paramsDict)

#####################################################################
# Scenario 4: intervene on the entire population by implementing    #
# policies that provide protections to LGB populations (e.g., same- #
# sex marriage, employment non-discrimination acts, etc.)- so in our#
# model, this intervention would be on the policy level. Returns the#
# final state of the simulation                                     #
#####################################################################
def Hypothetical_Policy(original):
    INTERVENTION_EFFECTIVESS = 1.25
    attr = "policy"
    return Hypothetical_genericTest(original, attr, 
        INTERVENTION_EFFECTIVESS, paramsDict)

#####################################################################
# Performs all of the hypothetical simulations that were of interest#
# and compares their effectiveness to the baseline simulation done  #
#####################################################################
def Hypothetical_findEffectiveness(original):
    # Indicates the gaps in time between passing of enforced policies
    TIME_GAP = 5

    # Given a simulation x, finds the final % depression
    getDepress = lambda x: x.network.networkBase.\
        NetworkBase_findPercentAttr("depression")

    network = original.network.networkBase
    finalScore = network.policyScore

    timeSteps = original.timeSpan
    timeAverageScore = int(TIME_GAP * finalScore/timeSteps)

    # Dictionary defined for ease of look-up throughout hypotheticals
    results = {
        "conceal": network.NetworkBase_findPercentAttr("concealed"),
        "discrimination": network.\
            NetworkBase_findPercentAttr("discrimination"),
        "attitude": network.NetworkBase_getNetworkAttitude(),
        "policy": timeAverageScore
    }

    paramsDict = {
        "attitude": None, 
        "support": None, 
        "discrimination": None, 
        "conceal": None, 
        "depression": None, 
        "policy": None
    }

    baseline = getDepress(original)
    effects = {
        "conceal": Hypothetical_LGB_Concealment(original, results),
        "discrimination": Hypothetical_NonLGB_Discrimination(
            original, results),
        "attitude": Hypothetical_NonLGB_Attitudes(original, results),
        "policy": Hypothetical_Policy(original, results)
    }

    effectStr = "{} is {} times as effective as baseline"
    for effect in effects:
        curDepress = getDepress(effects[effect])
        print(effectStr.format(effect, curDepress/baseline))