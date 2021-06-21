from base import RobotPolicy
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

class POSBCRobot(RobotPolicy):
    
    """ Implement solution for Part 2 below """
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', C=0.1, decision_function_shape = 'ovr', kernel = 'poly'))
    def train(self, data):
        for key, val in data.items():
            
            print(key)
            print(val)
        #print("Using dummy solution for POSBCRobot")
        obs = data["obs"]
        act = data["actions"]
        POSBCRobot.clf.fit(obs,act)
        
        pass

    def get_actions(self, observations):
        
        act_pred = POSBCRobot.clf.predict(observations)

        return act_pred
