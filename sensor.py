# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 14:54:02 2018

@author: wansh
"""
from gmap import j_region
import numpy as np
import random

class sensor_agent:
    def __init__(self,position,C,region,data_rate,slot):
        self.position=position.copy()
        self.capacity=C
        self.rNo=j_region(self.position,region)
        self.databuf=0
        self.data_rate=data_rate[self.rNo]
        self.slot=slot
        self.wait=0
        
    
    def fresh_buf(self,UAVlist):  #accumulate data in the former slot, transmit to UAV
        distance=[]
        num=len(UAVlist)
        self.databuf=self.databuf+np.random.poisson(self.data_rate*self.slot)
#        print(self.databuf)
        for i in range(num):
            p1=np.array([UAVlist[i].position[0],UAVlist[i].position[1]])
            p2=np.array([self.position[0],self.position[1]])
            distance.append(np.linalg.norm(p1-p2))
        
        min_d=min(distance)
        temp=[]
        inf=1e15
        for i in range(num):
            md=min(distance)
            if md>min_d+1:
                break
            l0=distance.index(md)
            temp.append(l0)
            distance[l0]=inf
        min_idx=random.sample(temp,1)[0]
        if(min_d>UAVlist[min_idx].r):
            self.wait=self.wait+1
            return -1
        else:
            UAVlist[min_idx].data_buf=UAVlist[min_idx].data_buf+min(self.databuf,self.slot*self.capacity)
            pre_buf=self.databuf
            self.databuf=max(0,self.databuf-self.slot*self.capacity)
            self.wait=self.databuf*self.wait/pre_buf
            return min_idx
