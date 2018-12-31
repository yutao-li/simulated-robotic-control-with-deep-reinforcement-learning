#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 21:04:09 2018

@author: masterlee
"""

import numpy as np
def features(state,action):
    feature=np.zeros(36)
    dealer=state[0]
    player=state[1]
    index1=[]
    index2=[]
    if dealer<4:
        index1.append(0)
    elif dealer==4:
        index1.append(0)
        index1.append(1)
    elif dealer<7:
        index1.append(1)
    elif dealer==7:
        index1.append(1)
        index1.append(2)
    else:
        index1.append(2)
        
    if player<4:
        index2.append(0)
    elif player<=6:
        index2.append(0)
        index2.append(1)
    elif player<=9:
        index2.append(1)
        index2.append(2)
    elif player<=12:
        index2.append(2)
        index2.append(3)        
    elif player<=15:
        index2.append(3)
        index2.append(4)
    elif player<=18:
        index2.append(4)
        index2.append(5) 
    else:
        index2.append(5)
    act=0 if action=='hit' else 18
#    print(dealer,index1,index2)
    for i1 in index1:
        for i2 in index2:
            feature[act+i1*6+i2]=1
    return feature
def compute_input(qvalue:dict):
    x=np.zeros((len(qvalue),36))
    for index,key in enumerate(qvalue.keys()):
        state,action=key
        x[index,:]=features(state,action)
    y=np.array(list(qvalue.values()))
    return x,y
#%%
