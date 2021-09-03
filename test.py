from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
# %matplotlib inline


def datasets(name):
    if name == 'Boston':
        boston = load_boston()
        boston_df = pd.DataFrame(boston['data'] )
        boston_df.columns = boston['feature_names']
        boston_df['PRICE']= boston['target']

        return boston_df
    elif name == 'Superconductivity':
        return pd.read_csv('train.csv')
    else:
        return None


def models(model, X, y, params = 'None'):
    if model == 'Linear Regression':
        lr = LinearRegression()
        lr.fit(X,y)
        return lr
    elif model == 'RandomForest':
        rf = RandomForestRegressor()
        rf.fit(X,y)
        return rf
    elif model == 'SVR':
        svr = SVR()
        svr.fit(X,y)
        return svr
    else:
        return None
