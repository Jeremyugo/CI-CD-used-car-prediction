"""
Prepares raw data and provides training, and test datasets
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

COLS_TO_DROP = ["carID", "mileage"]

def main():
    '''Read, split and save datasets'''
    
    # ------------ Reading Data ---------------- #
    # ------------------------------------------ #
    
    # readind the raw dataset
    data = pd.read_csv("../data/raw/car.csv")
    
    # dropping irrelevant columns
    data = data.drop(COLS_TO_DROP, axis=1)
    
    # identifying numerical and categorical columns
    num_attribs = data.select_dtypes(include=['int','float']).columns
    cat_attribs = data.select_dtypes(exclude=['int','float']).columns
    
    # getting dummy variables
    data = pd.concat([pd.get_dummies(data[cat_attribs], prefix_sep="", prefix=""), data[num_attribs]], axis=1)
    
    # splitting raw dataset into train and test
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    
    # saving train and test datasets
    train.to_csv("../data/prepared/train.csv", index=False)
    test.to_csv("../data/prepared/test.csv", index=False)
    
    
    
if __name__ == "__main__":
    main()