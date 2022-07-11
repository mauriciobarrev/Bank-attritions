from utils import Utils
from models import Models

if __name__ == '__main__':
    
    utils = Utils()
    models = Models()

    data = utils.load_from_csv('./in/data_balanced.csv')
    X, y = utils.features_target(data, ['Target'], ['Target'])

    
    models.rand_training(X, y)