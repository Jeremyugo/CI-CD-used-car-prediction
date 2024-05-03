"""
Trains ML model using training dataset, and saves trained model
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error

import mlflow
from mlflow.models import infer_signature


TARGET_COL = "price"

def main():
    ''' Read train dataset, preprocess dataset, and train model'''
    
    # read train data
    train = pd.read_csv("../data/prepared/train.csv")
    
    # preprocess data
    X_train_scaled, y_train, signature = preprocess(train)
    
    # train ML model
    vot_reg = model(X_train_scaled, y_train, signature)
    
    # make predictions using trained ML model
    pred = np.expm1(vot_reg.predict(X_train_scaled))
    
    # log model metrics
    mlflow.log_metrics({
        "mean_squared_error": mean_squared_error(np.expm1(y_train), pred),
        "mean_squared_log_error": mean_squared_log_error(np.expm1(y_train), pred),
        "R2": r2_score(np.expm1(y_train), pred)
    })
    



def preprocess(train):
    ''' preprocess the data for model training '''
    
    # ----------- Preprocess the train data ------------- #
    # --------------------------------------------------- #
    
    # delineate train and target dataframes
    X_train = train.drop(TARGET_COL, axis=1)
    y_train = train[TARGET_COL]
    
    # select numerical and categorical columns
    num_attribs = X_train.select_dtypes(include=['int','float']).columns
    cat_attribs = X_train.select_dtypes(exclude=['int','float']).columns
    
    
    # transformer class to handle dataframe selection
    class DataFrameSelector(BaseEstimator, TransformerMixin):
        def __init__(self, attrib_names):
            self.attrib_names = attrib_names
            
        def fit(self, X, y=None):
            return self
        
        def transform(self, X):
            return X[self.attrib_names]


    # numerical pipeline to handle numerical processing
    num_pipeline = Pipeline([
        ('select dataframe', DataFrameSelector(num_attribs)),
        ('standardize', StandardScaler())
    ])
    
    # categorical pipeline to select categorical attributes
    cat_pipeline = Pipeline([
        ('select dataframe', DataFrameSelector(cat_attribs))
    ])

    # full transformation pipeline
    full_pipeline = FeatureUnion(transformer_list = [
        ('numerical', num_pipeline),
        ('categorical', cat_pipeline)
    ])

    # transforming X_train
    X_train_scaled = full_pipeline.fit_transform(X_train)
    
    # transforming y_train
    y_train = np.log1p(y_train)
    
    # infering data signature
    signature = infer_signature(X_train, y_train)
    
    # saving transformation pipeline
    mlflow.sklearn.save_model(full_pipeline, path="../model/artifacts/scaler")

    return X_train_scaled, y_train, signature




def model(X_train_scaled, y_train, signature):
    ''' Train the ML model using various algorithms '''
    
    # ----------- Training the model------------- #
    # ------------------------------------------- #
    
    # defining ML models
    lin_reg = LinearRegression()
    ridge_reg = Ridge()
    rnd_reg = RandomForestRegressor()
    svm_reg = SVR()
    knn_reg = KNeighborsRegressor()
    
    models = [lin_reg, ridge_reg,  rnd_reg, svm_reg, knn_reg]
    
    # training ML models
    for idx, model in enumerate(models):
        print(f"#{idx+1}. Training {model.__class__.__name__}")
        model.fit(X_train_scaled, y_train)

    # creating an ensemble alogrithm
    vot_reg = VotingRegressor([
            ('LinearRegression', lin_reg),
            ('Knn', knn_reg),
            ('Ridge', ridge_reg),
            ('RandomForest', rnd_reg),
            ('SVR', svm_reg)
        ], weights=[0.05, 0.05, 0.05, 0.5, 0.35
    ])
    
    # fitting the ensemble algorithm
    vot_reg.fit(X_train_scaled, y_train)
    
    # saving model
    mlflow.sklearn.save_model(sk_model=vot_reg, path="../model/artifacts/model", signature=signature)
    
    return vot_reg




if __name__ == "__main__":

    mlflow.start_run()
    
    main()
    
    mlflow.end_run()