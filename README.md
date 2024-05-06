# CI-CD-used-car-prediction

This repository contains code for building a simple fully automated Continuous-Integration & Continous-Delivery Pipeline locally.

## Project Oragnization

This project is organized into the following directories:

- `app`: Streamlit app
- `data`: contains dataset used for training and evaluating ML model
- `docker/python`: dockerfile
- `evaluation`: contains model evaluation information
- `mlruns`: mlflow runs
- `model/artifacts`: temporary storage location for current trained ML model during workflow
- `notebooks`: contains experimentation notebook
- `src`: python executables


## CI-CD Workflow
![Picture1](https://github.com/Jeremyugo/mlops-car-prediction/assets/36512525/80e017fe-f177-4120-a99b-259cea4c41ee)


## Model Endpoint

Current deployed model can be accessed through the URL below

| Service       | URL                          |
| ------------- | ---------------------------- |
| Streamlit app | http://199.116.235.125:8501/ |
