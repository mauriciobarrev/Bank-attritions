from random import Random
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import Utils


class Models:
    
    def __init__(self):
        self.reg = {
            'RandomForest' : RandomForestClassifier(),
            'KNeighbors' : KNeighborsClassifier()
        }

        self.params = {
            'RandomForest' : {
                'n_estimators' : range(50, 150, 10),
                'max_depth' : range(4, 14, 2)
            },
            'KNeighbors': {
                'n_neighbors' : range(2,10,2),
                'weights' : ['uniform', 'distance'],
                'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']
            }

        }

    def rand_training(self, X, y):
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        best_score = 0
        best_model = None

        for name, reg in self.reg.items():
            
            rand_reg = RandomizedSearchCV(reg, 
                                          self.params[name],
                                          n_iter=5,  
                                          cv=3).fit(X_train, y_train.values.ravel())
                                          
            score = np.abs(rand_reg.best_score_)

            if (score > best_score) & (score < 1):
                best_score = score
                best_model = rand_reg.best_estimator_

            print(name, ":", score)

        utils = Utils()
        utils.model_export(best_model, best_score)


