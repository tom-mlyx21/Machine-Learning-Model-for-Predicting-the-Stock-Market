# import xgboost as xgb
import pandas as pd
import numpy as np
import nltk
pd.options.mode.chained_assignment = None # default='warn'

# Load the model into pandas
first_data = pd.read_csv('Combined_News_DJIA.csv')
# print(first_data.iloc[0])

# Using a ready made sentiment extractor to test against my model
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

text = "the start and end of the book"
scores = SentimentIntensityAnalyzer().polarity_scores(text)
#print(scores)

# Use Pandas to itterate over the data entries
# Build a new pandas dataframe with the sentiment scores + length
sentiment_data = first_data.copy()
for y in first_data.index:
    for x in first_data.columns:
        #print(x)
        if x != 'Date' and x != 'Label':
            target = first_data[x][y]
            #print(target)
            hold = []
            print(target)
            if target != "nan":
                scores = SentimentIntensityAnalyzer().polarity_scores(target)
                # Migrate the vader object to an array, to better suit pd
                hold.append(len(target))
                hold.append(target.count(' ')+1)
                for i in scores.values():
                    hold.append(i)
                sentiment_data[x][y] = hold
                #print(hold)
                # print(scores)
print(sentiment_data.iloc[1])
