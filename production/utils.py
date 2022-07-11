import pandas as pd
import joblib

class Utils:
    
    def load_from_csv(self, path):
        return pd.read_csv(path)

    def features_target(self, dataset, drop_cols, target_col):
        X = dataset.drop(drop_cols, axis=1)
        y = dataset[target_col]
        return X, y
    
    def model_export(self, clf, score):
        print("Best score: ", score)
        joblib.dump(clf, './models/best_model.pkl')


