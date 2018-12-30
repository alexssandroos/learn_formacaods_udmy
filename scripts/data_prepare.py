#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 11:14:34 2018

@author: alexssandroos
"""
import pandas as pd

df = pd.read_csv('../data/credit-data.csv')
df.describe()
