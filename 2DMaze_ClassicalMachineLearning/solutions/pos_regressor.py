#!/usr/bin/python3
from base import Regressor
from sklearn.kernel_ridge import KernelRidge
import numpy as np

class PositionRegressor(Regressor):

    """ Implement solution for Part 1 below  """
    
    
    clf = KernelRidge(alpha=1.0)

    def train(self, data):
        #for key, val in data.items():
         #   print("This is the key")
          #  print(key)
           # print("This is the value")
            #print(val)
        obs = data["obs"]
        info = data["info"]
        pos = np.empty([500,2])
       
        for i in range(len(info)):
            pos[i][0] = info[i]["agent_pos"][0]
            pos[i][1] = info[i]["agent_pos"][1]
        
        obs = obs.reshape(len(obs),3*64*64)
        PositionRegressor.clf.fit(obs,pos)
        pass

    def predict(self, Xs):
        Xs = Xs.reshape(len(Xs),3*64*64)
        pos_pred = PositionRegressor.clf.predict(Xs)
        return pos_pred
