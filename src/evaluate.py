"""
Evaluates the trained ML using the test dataset and deploys flag
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error

import mlflow
from mlflow.tracking import MlflowClient

TARGET_COL = "price"
MODEL_NAME = "car-prediction"



def main():
    '''Read trained model and test dataset, evaluate model and save result'''
    
    # loading the test set
    test = pd.read_csv("data/prepared/test.csv")
    
    # processing test dataset
    X_test, X_test_scaled, y_test = preprocess(test)
    
    # loading the model
    model = mlflow.sklearn.load_model("model/artifacts/model")
    
    # making model predictions
    pred = np.expm1(model.predict(X_test_scaled))
    
    # evaluating the model performance
    score = mean_squared_error(y_test, pred)
    
    # model promotion
    predictions, deploy_flag = model_promotion(MODEL_NAME, X_test, y_test, pred, score)
    


def preprocess(test):
    ''' preprocess the data for model training '''
    
    # ----------- Preprocess the train data ------------- #
    # --------------------------------------------------- #
    
    # delineate train and target dataframes
    X_test = test.drop(TARGET_COL, axis=1)
    y_test = test[TARGET_COL]
    
    scaler = mlflow.sklearn.load_model("model/artifacts/scaler")
    
    X_test_scaled = scaler.transform(X_test)
    
    return X_test, X_test_scaled, y_test



def model_promotion(MODEL_NAME, X_test, y_test, pred, score):
    '''compares current model to currently deployed model'''
    
    scores = {}
    predictions = {}
    
    client = MlflowClient()
    
    for model_run in client.search_model_versions(f"name='{MODEL_NAME}'"):
        model_version = model_run.version
        
        mdl = mlflow.pyfunc.load_model(
            model_uri=f"models:/{MODEL_NAME}/{model_version}"
        )
        predictions[f"{MODEL_NAME}:{model_version}"] = mdl.predict(X_test)
        
        scores[f"{MODEL_NAME}:{model_version}"] = mean_squared_error(y_test, predictions[f"{MODEL_NAME}:{model_version}"])
        
    
    if scores:
<<<<<<< HEAD
        if score <= max(list(scores.values)):
=======
        if score >= max(list(scores.values())):
>>>>>>> b0570c382779332bdc7b81e2e0c56062f8ebc198
            deploy_flag = 1
            
        else:
            deploy_flag = 0
            
            
    else:
        deploy_flag = 1
        
    print(f"Deploy flag: {deploy_flag}")
    
    with open("evaluation/deploy_flag.txt", 'w') as outfile:
        outfile.write(f"{int(deploy_flag)}")
        
        
    scores["current model"] = score
    predictions["current model"] = pred
    
    perf_comparison_plot = pd.DataFrame(scores, index=["mse"]).plot(kind="bar", figsize=(15, 10))
    perf_comparison_plot.figure.savefig("perf_comparison.png")
    perf_comparison_plot.figure.savefig("evaluation/perf_comparison.png")
    
    mlflow.log_metric("deploy_flag", bool(deploy_flag))
    mlflow.log_artifact("perf_comparison.png")
    
    return predictions, deploy_flag



if __name__ == "__main__":
    
    mlflow.start_run()
    
    main()
    
    mlflow.end_run()
