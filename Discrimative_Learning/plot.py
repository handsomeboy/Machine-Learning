#!/usr/bin/python

import numpy as np
from multilayer import *
import matplotlib.pyplot as plt
import pylab
import matplotlib
from metrics import *
from get_digits_data import *


fmeasures = [0.96002214138885866, 0.95998946493702708, 0.98585453553028568, 0.98785080070302145, 0.97402132254238993, 0.97598300116227743, 0.97402132254238993, 0.97391713677304348, 0.97391713677304348]
plt.plot(range(100,1000, 100),fmeasures)
plt.xlabel("h")
plt.ylabel("F-Measure")
plt.show()