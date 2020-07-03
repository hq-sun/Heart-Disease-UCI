#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 11:19:16 2020

@author: Heqing Sun
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Get current working directory
os.getcwd()

# Read cleaned csv file from Step 1 - data cleaning
df = pd.read_csv("./data/clean/data_clean.csv")
# df_backup = df.copy()
## 303 obs, 20 vars