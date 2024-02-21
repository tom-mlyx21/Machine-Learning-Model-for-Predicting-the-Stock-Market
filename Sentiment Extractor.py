import xgboost as xgb
import pandas as pd
import numpy as np

# Load the model into pandas
first_data = pd.read_csv('Combined_News_DJIA.csv')
print(first_data.iloc[0])


