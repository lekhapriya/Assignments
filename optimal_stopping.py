#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 12:58:57 2021

@author: lekhapriyadheerajkashyap
"""

import numpy as np

T = 20

df = 0.99 #discount factor

s = np.arange(2,13) #sample space
total = 36 #number of samples

d = np.array([i+j for i in range(1,7) for j in range(1,7)]).reshape(6,6)

pmf = np.array([np.count_nonzero(d==x)/total if x!= 7 else 0 for x in range(2,13)])

exp_r = [] #expected reward
eP = 0 #atterminal state

#value iteration
for i in range(T):
    a1 = (s>eP) #action to stop and cash in
    a2 = (s<eP) #action to continue rolling
    eP = sum(pmf*s*a1 + pmf*eP*a2)*df
    print("at T=",T-i," the expected reward is ", eP)
