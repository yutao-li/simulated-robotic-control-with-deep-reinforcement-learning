#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 13:24:59 2018

@author: masterlee
"""
import numpy as np
def draw_a_card():
    point=np.random.randint(1,11)
    color=np.random.randint(3)
    return point,color
def step(state,action):
    dealer_first=state[0]
    player_sum=state[1]
    assert action=="hit" or action=="stick","no such action"
    if action=='hit':
        point,color=draw_a_card()
        if color<=1:
            player_sum+=point
        else:
            player_sum-=point
    else:
        dealer_sum=dealer_first
        while dealer_sum<17:
            point,color=draw_a_card()
            if color<=1:
                dealer_sum+=point
            else:
                dealer_sum-=point
            if dealer_sum<1:
                break
    terminal=False
    if player_sum<1 or player_sum>21:
        reward=-1;terminal=True
    elif action=='stick':
        if dealer_sum<1 or dealer_sum>21:
            reward=1
        else:
            reward=0 if player_sum==dealer_sum else player_sum>dealer_sum
        terminal=True
    if not terminal:
        return dealer_first,player_sum
    else:
        return 'terminal',reward
#%%
from collections import defaultdict
#counts of s and s,a,should be reinitialized when using another algorithm
N=defaultdict(lambda:0)
def select_action(qvalue_dict,state,epsilon=None):
    if qvalue_dict[state,'hit']==qvalue_dict[state,'stick']:
        return 'hit' if np.random.randint(2)==0 else 'stick'
    else: 
        if epsilon==None:
            epsilon=100/(100+N[state])
        if np.random.rand()<1-epsilon:
            return 'hit' if qvalue_dict[state,'hit']>qvalue_dict[state,'stick'] else 'stick'
        else:
            return 'stick' if qvalue_dict[state,'hit']>qvalue_dict[state,'stick'] else 'hit'
 
#%%
qvalue_sarsa=defaultdict(lambda:0)
def sarsalambda(lam,gamma=1,epsilon=None,alpha=None,linear_approx=False):
    dealer_first=draw_a_card()[0]
    player_first=draw_a_card()[0]
    eligibility=defaultdict(lambda:0)
    state=dealer_first,player_first
    action=select_action(qvalue_sarsa,state)
    while True:        
        next_state=step(state,action)
        N[state]+=1;N[state,action]+=1
        reward=0 if not next_state[0] == 'terminal' else next_state[1]
        next_action=select_action(qvalue_sarsa,next_state,epsilon)
        if linear_approx==False:
            delta=reward+gamma*qvalue_sarsa[next_state,next_action]-qvalue_sarsa[state,action]
        else:
            if next_state[0]=='terminal':
                q_pred=0
            else:
                q_pred=reg.predict(features(next_state,next_action).reshape(1,-1))[0]
            delta=reward+gamma*q_pred-qvalue_sarsa[state,action]
        eligibility[state,action]+=1
        for key in eligibility.keys():
            if alpha==None:
                alpha=1/N[state,action]
            qvalue_sarsa[key]+=alpha*delta*eligibility[key]
            eligibility[key]*=gamma*lam
        state=next_state;action=next_action
        if next_state[0] == 'terminal':
            break
#%%
qvalue=defaultdict(lambda:0)
def simpleupdate(reward,trajectory):    
    for state,action,_ in trajectory:
        alpha=1/N[state,action]
        qvalue[state,action]+=alpha*(reward-qvalue[state,action])
    
def montecarlo():
    dealer_first=draw_a_card()[0]
    player_first=draw_a_card()[0]
    trajectory=[]
    state=dealer_first,player_first
    while True:        
        action=select_action(qvalue,state)
        next_state=step(state,action)
        N[state]+=1;N[state,action]+=1
        trajectory.append([state,action,next_state])
        if next_state[0]=='terminal':
            reward=next_state[1]
            simpleupdate(reward,trajectory)
            break
        state=next_state
#%%
from time import clock
iterations=0
start_time=clock()
N=defaultdict(lambda:0)
qvalue=defaultdict(lambda:0)
while iterations<1e7:
    iterations+=1
    if iterations%1000000==0:
        print(iterations,clock()-start_time)
    montecarlo()
#    sarsalambda(1)
#%%
from sklearn import linear_model
reg=linear_model.LinearRegression()
x,y=compute_input(qvalue)      
reg.fit(x,y)   
#%%
import matplotlib.pyplot as plt
plt.ion()
fig=plt.figure(figsize=(20,15))
ax=fig.add_subplot(1,1,1)
#%%
def compute_mse():
    mse=0
    for key in qvalue.keys():
        mse+=(qvalue[key]-qvalue_sarsa[key])**2
    mse/=len(qvalue)
    return mse
def comparisons(lam):
    iterations=0
    learning_curve=[]
    while iterations<1e4:
        iterations+=1
        if iterations%1000==0:
            print(iterations)
            mse=compute_mse()
            if lam==0 or lam==1:
                learning_curve.append(mse)
            print(mse)
        sarsalambda(lam,linear_approx=True)
    if lam==0 or lam==1:
        ax.plot(np.linspace(1000,10000,10),learning_curve,label='lin approx:'+str(lam))
        ax.legend(prop={'size': 26})
        plt.pause(0.001)
    return mse
mse_list=[]
for lam in np.linspace(0,1,11):
    N=defaultdict(lambda:0)
    qvalue_sarsa=defaultdict(lambda:0)
    print("lambda:",lam)
    mse_list.append(comparisons(lam))
#%%
plt.plot(np.linspace(0,1,11),mse_list)
#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.ion()
values=np.empty((10,21))
x,y=np.meshgrid(list(range(21)),list(range(10)))
#%%
def plot_values():
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(10):
        for j in range(21):
            dealer_first=i+1;player_sum=j+1
            values[i,j]=max(qvalue[(dealer_first,player_sum),'hit'], \
                  qvalue[(dealer_first,player_sum),'stick'])
    ax.plot_surface(x,y,values,cmap=cm.coolwarm, \
                 linewidth=0, antialiased=False)
    plt.pause(0.05)
#%%
import pickle
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
        
    
