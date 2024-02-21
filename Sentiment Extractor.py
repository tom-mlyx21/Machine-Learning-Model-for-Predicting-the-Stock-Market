import xgboost as xgb
import pandas as pd
import numpy as np
import nltk

# Load the model into pandas
first_data = pd.read_csv('Combined_News_DJIA.csv')
#print(first_data.iloc[0])

#Using a ready made sentiment extractor to test against my model
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
text = "the start and end of the book"
scores = SentimentIntensityAnalyzer().polarity_scores(text)
print(scores)