"""
Registers trained ML model if deploy flag is True
"""

import json
import shutil
import numpy as np
import pandas as pd
import mlflow

MODEL_NAME = "car-prediction"
BASE_MODEL_PATH = "model/artifacts/model"
SCALER_PATH = "model/artifacts/scaler"

def main():
    '''Loads and registers model if deploy flag is True'''
    
    with open("evaluation/deploy_flag.txt", "rb") as infile:
        deploy_flag = int(infile.read())
        
    
    mlflow.log_metric("deploy_flag", int(deploy_flag))
    
    if deploy_flag==1:
        print("Registering ", MODEL_NAME)
        
        base_model = mlflow.sklearn.load_model(BASE_MODEL_PATH)
        
        scaler = mlflow.sklearn.load_model(SCALER_PATH)
        
        # log models
        mlflow.sklearn.log_model(base_model, "model/base_model")
        mlflow.sklearn.log_model(scaler, "model/scaler")
        
        # register logged model
        run_id = mlflow.active_run().info.run_id
        base_uri = f"runs:/{run_id}/base_model"
        scaler_uri = f"runs:/{run_id}/scaler"
        model_uri = f"runs:/{run_id}/{MODEL_NAME}"
        
        # custom model
        class CustomPredict(mlflow.pyfunc.PythonModel):
            def __init__(self,):
                self.full_pipeline = mlflow.sklearn.load_model(scaler_uri)
                
            def process_inference_data(self, model_input):
                model_input = self.full_pipeline.transform(model_input)
                return model_input
            
            def process_prediction(self, predictions):
                predictions = np.expm1(predictions)
                predictions = predictions.astype(int)
                return predictions
            
            def load_context(self, context=None):
                self.model = mlflow.sklearn.load_model(base_uri)
                return self.model
            
            def predict(self, context, model_input):
                model = self.load_context()
                model_input = self.process_inference_data(model_input)
                predictions = model.predict(model_input)
                return self.process_prediction(predictions)
    
    
        # saving the custom model
        mlflow.pyfunc.log_model(artifact_path="model/"+MODEL_NAME, python_model=CustomPredict())
        
        # registering models
        mlflow_model = mlflow.register_model(scaler_uri, "scaler")
        mlflow_model = mlflow.register_model(base_uri, "base_model")
        mlflow_model = mlflow.register_model(model_uri, MODEL_NAME)
        model_version = mlflow_model.version
        
        # write model info
        print("Writing Json")
        dict = {"id": f"{MODEL_NAME}:{model_version}"}
        with open("evaluation/model_info.json", "w") as of:
            json.dump(dict, fp=of)
    
    
    else:
        print("Model will not be registered!")
        
    # removing temp model sub-directories
    shutil.rmtree(BASE_MODEL_PATH)
    shutil.rmtree(SCALER_PATH)
    

if __name__ == "__main__":
    
    mlflow.start_run()
    
    main()
    
    mlflow.end_run()