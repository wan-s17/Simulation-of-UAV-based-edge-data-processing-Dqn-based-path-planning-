# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 16:29:07 2018

@author: wansh
"""

import numpy as np
import gmap as gp
from center_dqn import Center_DQN
from uav import UAV_agent
from sensor import sensor_agent
import matplotlib.pyplot as plt

Ed=10000                             #total slot
ep0=0.97
batch_size=12                 #training samples per batch
pl_step=5                    #How many steps will The system plan the next destination
T=300                          #How many steps will the epslon be reset and the trained weights will be stored
com_r=60
num1=5
num2=4
region=gp.genmap(600,400,num1,num2)
E_wait=np.ones([401,601])
P_cen=np.array([300,200])
t_bandwidth=2e6
N0=2e-20
f_max=2e9    #the max cal frequency of UAV
k=1e-26
cal_L=3000
slot=0.5
num_UAV=6
omeg=1/num_UAV
num_sensor=20000
p_max=5
alfmin=1e-3
num_region=num1*num2
C=2e3
v=8
V=10e9
v1=v*np.sin(np.pi/4)
region_obstacle=gp.gen_obs(num_region)
region_rate=np.zeros([num_region])
averate=np.random.uniform(280,300,[num_region])
p_sensor=gp.position_sensor(region,num_sensor)
vlist=[[0,0],[v,0],[v1,v1],[0,v],[-v1,v1],[-v,0],[-v1,-v1],[0,-v],[v1,-v1]]
g0=1e-4
d0=1
the=4
OUT=np.zeros([num_UAV])
reward=np.zeros([num_UAV])
reset_p_T=800

#jud=70000
gammalist=[0,0.1,0.2,0.3,0.5,0.6,0.7,0.8,0.9]
Mentrd=np.zeros([num_UAV,Ed])

#generate UAV agent
UAVlist=[]
for i in range(num_UAV):
    UAVlist.append(UAV_agent(i,com_r,region_obstacle,region,omeg,slot,t_bandwidth,cal_L,k,f_max,p_max))
    
#generate sensor agent
sensorlist=[]
for i in range(num_sensor):
    sensorlist.append(sensor_agent([p_sensor['W'][i],p_sensor['H'][i]],C,region,averate,slot))


Center=Center_DQN((84,84,1),9,num_UAV,batch_size)
#Center.load("./save/center-dqn.h5")
prebuf=np.zeros([num_UAV])
data=np.zeros([num_UAV])
#pre_data=np.zeros([num_UAV])

#define record data buf
cover=np.zeros([Ed])

#init plt
plt.close()  #clf() # 清图  cla() # 清坐标轴 close() # 关窗口
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
plt.xlim((0,600))
plt.ylim((0,400))
plt.grid(True) #添加网格
plt.ion()  #interactive mode on
X=np.zeros([num_UAV])
Y=np.zeros([num_UAV])
fg=1

for t in range(Ed):  #move first, get the data, offload collected data
    gp.gen_datarate(averate,region_rate)
#    print(t)
    if t%T==0 and t>0:
        Center.epsilon=ep0
        Center.save("./save/center-dqn.h5")

    if t%pl_step==0:
        pre_feature=[]
        aft_feature=[]
        act_note=[]
        for i in range(num_UAV): 
            pre_feature.append(UAVlist[i].map_feature(region_rate,UAVlist,E_wait))    #record former feature
            act=Center.act(pre_feature[i],fg)          # get the action V
            act_note.append(act)                  #record the taken action
    
    for i in range(num_UAV):
        OUT[i]=UAVlist[i].fresh_position(vlist[act_note[i]],region_obstacle)     #execute the action
        UAVlist[i].cal_hight()
        X[i]=UAVlist[i].position[0]
        Y[i]=UAVlist[i].position[1]
        UAVlist[i].fresh_buf()
        prebuf[i]=UAVlist[i].data_buf   #the buf after fresh by server
        
    gp.list_gama(g0,d0,the,UAVlist,P_cen)

    for i in range(num_sensor):          #fresh buf send data to UAV
        sensorlist[i].data_rate=region_rate[sensorlist[i].rNo]
        sensorlist[i].fresh_buf(UAVlist)
        cover[t]=cover[t]+sensorlist[i].wait
    cover[t]=cover[t]/num_sensor
    print(cover[t])
        
    for i in range(num_UAV):
        reward[i]=reward[i]+UAVlist[i].data_buf-prebuf[i]
        Mentrd[i,t]=reward[i]
#    if sum(OUT)>=num_UAV/2:
#        fg=0
#    if np.random.rand()>0.82 and fg==0:
#        fg=1
    
    if t%pl_step==0:    
        E_wait=gp.W_wait(600,400,sensorlist)
        rdw=sum(sum(E_wait))
        print(t)
        for i in range(num_UAV):        #calculate the reward : need the modify
#            aft_feature.append(UAVlist[i].map_feature(region_rate,UAVlist,E_wait))    #recode the current feature
            rd=reward[i]/1000
            reward[i]=0
            UAVlist[i].reward=rd

    if t>0:
        ax.clear()
    plt.xlim((0,600))
    plt.ylim((0,400))
    plt.grid(True) #添加网格

    ax.scatter(X,Y,c='b',marker='.')  #散点图
#    if t>0:
    plt.pause(0.1)

    
#np.save("record_rd3",Mentrd)
np.save("cover_hungry_10",cover)
fig=plt.figure()
plt.plot(cover)
plt.show()

