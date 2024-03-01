# import xgboost as xgb
import pandas as pd
import numpy as np
import nltk
from nltk.lm import NgramCounter
from nltk.util import ngrams
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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
                #Counting ngrams in text
                text_unigrams = [ngrams(sent, 1) for sent in target]
                text_bigrams = [ngrams(sent, 2) for sent in target]
                text_trigrams = [ngrams(sent, 3) for sent in target]
                count = NgramCounter(text_unigrams+text_bigrams+text_trigrams)
                print("Count is:", count.N())
                for i in scores.values():
                    hold.append(i)
                sentiment_data[x][y] = hold
                hold.append(count)
    counter += 1
    print(counter)
print(sentiment_data.iloc[2:50])


# After the Process of Sentiment Extraction, the data is ready to be used for training
# A new dataset will be built using new column headings
# Source, Length, Word Count, Positive, Negative, Neutral, Compound

clean_data = pd.DataFrame(columns=['Source', 'Length', 'Word Count', 'Positive', 'Negative', 'Neutral', 'Compound', 'Label', 'Avg WordLen', 'N-Grams', 'Polarity Shift', 'Punctuation', 'Entities'])
for x in sentiment_data.columns:
    if x != 'Date' and x != 'Label':
        for y in range(5):
            hold = sentiment_data[x][y]
            if type(hold) == list:
                set = {'Source': x[3:], 'Length': hold[0], 'Word Count': hold[1], 'Positive': hold[2], 'Negative': hold[3], 'Neutral': hold[4], 'Compound': hold[5], 'Label': sentiment_data['Label'][y], 'Avg WordLen': int(hold[0]/hold[1]), 'N-Grams': int(hold[6])}
                clean_data.loc[len(clean_data)] = set
print(clean_data.head())

# Time For Training
X = clean_data[['Source', 'Length', 'Word Count', 'Positive', 'Negative', 'Neutral', 'Compound', 'Avg WordLen', 'N-Grams']]
Y = clean_data['Label']
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, test_size=.3, random_state=0)
clf = GaussianNB()
clf.fit(Xtrain,Ytrain)
Ypred = clf.predict(Xtest)
print(classification_report(Ytest,Ypred))

print("Finished without failure")