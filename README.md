# CI-CD-used-car-prediction

This repository contains code for building a simple fully automated Continuos Integration & Continous Delivery Pipeline locally.

## Project Oragnization

This project is organized into the following directories:

- `data`: contains dataset used for training and evaluating ML model
- `evaluation`: contains model evaluation information
- `model`: temporary storage location for current trained ML model
- `notebooks`: contains experimentation notebook
- `src`: python executables

# Getting Started

To implement reproduce this repo

1. create an ec2 ubuntu instance or WSL vm
2. git clone the repo
3. cd in cloned repo
4. run the following in the ubuntu terminal

```shell
pip install -r requirements.txt
```

5. proceed with experimentation

## Model Endpoint

Current deployed model can be accessed through the URL below

| Service       | URL                          |
| ------------- | ---------------------------- |
| Streamlit app | http://199.116.235.125:8501/ |
