import random
import numpy as np
from collections import deque
from tensorflow import keras
import tensorflow as tf
from gmap import find_pos,j_region

#EPISODES = 50


class Center_DQN:
    def __init__(self, state_size, action_size,num_UAV,batch_size):
        self.state_size = state_size
        self.action_size = action_size
#        self.memory = deque(maxlen=124)
        self.memory=[]
        self.gamma = 0.8    # discount rate
        self.epsilon = 0.97  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.92
        self.N=36
        self.rtz=200
        self.jr=0
        self.num=0
        self.alpha=0.1
        self.pro=np.zeros([action_size])
        self.loss=[]
#        self.learning_rate = 0.001
        self.model = self._build_model()
        self.tmodel= self._build_model()
        self.num_U=num_UAV
        for i in range(num_UAV):
            self.memory.append(deque(maxlen=batch_size+10))

    def _build_model(self): #Set network of central training
        # Neural Net for Deep-Q learning Model
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(32, (8,8), strides=4,activation='relu',input_shape = self.state_size))
#        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Conv2D(64, (4,4), strides=2,activation='relu'))
        model.add(keras.layers.Conv2D(64, (3,3), strides=1,activation='relu'))
#        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
#        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(optimizer='rmsprop',loss='mse')
        return model

    def remember(self, state, action, reward, next_state,i):
        self.memory[i].append((state, action, reward, next_state))

        

        
    def act(self, state,fg):
        nrd=np.random.rand()
        if nrd <= self.epsilon:
            return random.randrange(self.action_size)
        state=np.reshape(state,[1,self.state_size[0],self.state_size[1],self.state_size[2]])
        act_values = self.model.predict(state)
        print(np.amax(act_values[0]))
        return np.argmax(act_values[0])  # returns action

#training process
    def replay(self, batch_size, i1,t):
        self.alpha=1/np.sqrt((t+1)/5)
        if self.num==0:
            self.model.save_weights("./save/temp.h5")
            self.tmodel.load_weights("./save/temp.h5")
        minibatch = random.sample(self.memory[i1], batch_size)
        train_sp=np.zeros([batch_size,self.state_size[0],self.state_size[1],self.state_size[2]])
        tg=np.zeros([batch_size,self.action_size])
#        minibatch=self.memory[i]
        error=0
        i=0
        for state1, action, reward, next_state in minibatch:
            state=np.reshape(state1,[1,self.state_size[0],self.state_size[1],self.state_size[2]])
            next_state=np.reshape(next_state,[1,self.state_size[0],self.state_size[1],self.state_size[2]])
            pdc=self.model.predict(state)[0]
            self.pro[action]+=1
            w=sum(self.pro)/self.pro[action]
#            if reward<=0:
#                w=6
            ap=min(0.9,self.alpha*w)
#            ap=self.alpha
            target = ap*(reward + self.gamma *
                          np.amax(self.tmodel.predict(next_state)[0]))+(1-ap)*pdc[action] #第一维是属于哪个batch
            target_f = self.model.predict(state)
            target_f[0][action] = target
            tg[i]=target_f[0]
            train_sp[i]=state1
            i+=1
            error+=  abs((target-pdc[action])/ap)
#            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        self.loss.append(error/batch_size)    
        self.model.fit(train_sp, tg, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min and i1==self.num_U-1:
            self.epsilon *= self.epsilon_decay
        self.num +=1
        self.jr +=1
        if self.num==self.N:
            self.num=0
#        if self.jr==self.rtz:
#            self.jr=0
#            for i in range(self.action_size):
#                self.pro[i]=0
    
    def find_ld(self,UAVlist,alfmin):
        ld_L=1e50
        ld_U=-1e50
        num=len(UAVlist)
        for i in range(num):
            h=UAVlist[i].data_buf*UAVlist[i].bandwidth*UAVlist[i].slot
            M=UAVlist[i].gama*UAVlist[i].p_tr/(UAVlist[i].noise*UAVlist[i].bandwidth)
            ldl_t=h*np.log2(1+M/1)-h*M/(np.log(2)*(1+M))
            ldu_t=h*(np.log2(1+M/alfmin)-M/(np.log(2)*(M+alfmin)))
            if ldl_t<ld_L:
                ld_L=ldl_t
            if ldu_t>ld_U:
                ld_U=ldu_t
        return [ld_L,ld_U]
            
            
        
    
    def cal_com(self,UAVlist,alfmin,ite=20):
        [ld_L,ld_U]=self.find_ld(UAVlist,alfmin)
#        print("%f,%f"%(ld_L,ld_U))
        num=len(UAVlist)
        ite2=20
        for i in range(ite2):
            mid=(ld_L+ld_U)/2
            grad=0
            for j in range(num):
                grad=grad+UAVlist[j].cal_alpha(mid,alfmin,ite,1)
            if grad<=1 and grad>=0.8:
                break
            elif grad>1:
                ld_L=mid
            else:
                ld_U=mid
        return mid   
    
    def para_com(self,UAVlist,noise,V,p_max,alfmin):   #calculate UAV offloading
        num=len(UAVlist)
        for i in range(num):
            UAVlist[i].p_tr=p_max #cal ptr give values to noise....
            
        for j in range(2):
            self.cal_com(UAVlist,alfmin)  #cal ptr and alpha by dual decomposition
            for i in range(num):
                UAVlist[i].cal_ptr(p_max,V,noise)
        for i in range(num):
            UAVlist[i].cal_f(V)
        return j
            
        
    
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        np.save("train_loss",self.loss)

