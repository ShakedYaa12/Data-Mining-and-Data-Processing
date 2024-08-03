import pickle
import pandas as pd 
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from app.car_data_prep import prepare_data

def forcast(df):
    # Separating the target variable from the features
    y = df['Price']
    X = df.drop(columns=['Price'])
    
    # Creating a pipeline for scaling and model training
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Scaling all features
        ('model', ElasticNet())
    ])
    
    # Defining the parameter grid for the search
    param_grid = {
        'model__alpha': np.logspace(-4, 1, 10),
        'model__l1_ratio': np.linspace(0, 1, 10)
    }
    
    # Performing Grid Search
    grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X, y)
    
    # Getting the best results
    best_model = grid_search.best_estimator_
    
    return best_model


df = pd.read_csv("./data/processed/part2-allCars.csv")

df = prepare_data(df)
model = forcast(df)

pickle.dump(model, open("/models/trained_model.pkl","wb"))
