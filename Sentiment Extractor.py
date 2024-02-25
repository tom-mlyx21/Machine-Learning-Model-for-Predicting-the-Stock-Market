# import xgboost as xgb
import pandas as pd
import numpy as np
import nltk

pd.options.mode.chained_assignment = None  # default='warn'

# Load the model into pandas
first_data = pd.read_csv('Combined_News_DJIA.csv')
# print(first_data.iloc[0])

# Using a ready-made sentiment extractor to test against my model
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Use Pandas to iterate over the data entries
# Build a new pandas dataframe with the sentiment scores + length
sentiment_data = first_data.copy()
"""
counter = 0
for y in first_data.index:
    for x in first_data.columns:
        if x != 'Date' and x != 'Label':
            target = first_data[x][y]
            hold = []
            # Can't encode floats so only strings accepted (Loss of 1 in 250 ish entries)
            if type(target) == str:
                scores = SentimentIntensityAnalyzer().polarity_scores(target)
                # Migrate the vader object to an array, to better suit pd
                hold.append(len(target))
                hold.append(target.count(' ')+1)
                for i in scores.values():
                    hold.append(i)
                sentiment_data[x][y] = hold
    counter+=1
    print(counter)
"""
# Since the original dataset is too large, I will only use the first 50 entries to test the code
counter = 0
for y in range(50):
    for x in first_data.columns:
        if x != 'Date' and x != 'Label':
            target = first_data[x][y]
            hold = []
            # Can't encode floats so only strings accepted (Loss of 1 in 250 ish entries)
            if type(target) == str:
                scores = SentimentIntensityAnalyzer().polarity_scores(target)
                # Migrate the vader object to an array, to better suit pd
                hold.append(len(target))
                hold.append(target.count(' ') + 1)
                for i in scores.values():
                    hold.append(i)
                sentiment_data[x][y] = hold
    counter += 1
    print(counter)
print(sentiment_data.iloc[2:50])
