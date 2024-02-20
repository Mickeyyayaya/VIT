
import numpy as np
from torch import embedding
from SRAM_m import SRAM
from PE import PE_array
from scipy.sparse import coo_matrix
import logging
import os
import math
import argparse



bandwidth = 76.8 * 1024 * 1024 * 1024 * 8 
freq = 500*1e6
a = math.ceil(bandwidth / (freq*8))
print(a)