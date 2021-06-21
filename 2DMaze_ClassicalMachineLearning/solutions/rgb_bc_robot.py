from base import RobotPolicy
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


class RGBBCRobot(RobotPolicy):

    """ Implement solution for Part3 below """
    
    transformer = PCA(n_components=9, svd_solver='full')
    clf = LogisticRegression(C=2, solver = 'lbfgs',max_iter = 300)
    scaler = MinMaxScaler(feature_range=(0,1))

    def train(self, data):
        #for key, val in data.items():
         #   print(key)
          
          #  print(val)
        #print("Using dummy solution for RGBBCRobot")
        obs = data["obs"]
        obs = obs.reshape(400,64*64*3)
        act = data["actions"]
        RGBBCRobot.scaler.fit(obs)
        obs = RGBBCRobot.scaler.transform(obs)
        RGBBCRobot.transformer.fit(obs)
        new_obs = RGBBCRobot.transformer.transform(obs)
        RGBBCRobot.clf.fit(new_obs,act)
        
        pass

    def get_actions(self, observations):
       observations = RGBBCRobot.scaler.transform(observations)
       observations = RGBBCRobot.transformer.transform(observations)
       act_pred = RGBBCRobot.clf.predict(observations)
       return act_pred
