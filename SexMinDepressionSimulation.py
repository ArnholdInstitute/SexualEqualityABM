#####################################################################
# Name: Yash Patel                                                  #
# File: SESimulation.py                                             #
# Description: Contains all the methods pertinent to producing the  #
# simulation for modelling the relation between SE and exercise     #
# levels in the form of an ABM (agent-based model)                  #
#####################################################################

import sys
import os
import csv
import random,itertools
from copy import deepcopy
import numpy as np

import matplotlib.pyplot as plt
from operator import itemgetter 

try:
    import networkx as nx
except ImportError:
    raise ImportError("You must install NetworkX:\
    (http://networkx.lanl.gov/) for SE simulation")

class SexMinDepressionSimulationModel: