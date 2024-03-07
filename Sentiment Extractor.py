import pandas as pd
import nltk
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')


pd.options.mode.chained_assignment = None  # default='warn'

# Load the model into pandas
first_data = pd.read_csv('Combined_News_DJIA.csv')
print(first_data)
# Using a ready-made sentiment extractor to test against my model


# Use Pandas to iterate over the data entries
# Build a new pandas dataframe with the sentiment scores + length
sentiment_data = first_data.copy()
# Since the original dataset is too large, I will only use the first 50 entries to test the code
counter = 0
print()
choice = input("Enter F for full data, or anything else for testing")
if choice == 'F':
    boundary = len(first_data)
else:
    boundary = 50
for y in range(boundary):
    print("index:",  counter)
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
                # totaling ngrams in text
                sent = target.split()
                uni, bi, tri = 0, 0, 0
                for t in sent:
                    if len(t) == 1 and t != "'" and t != '-' and t != '"':
                        uni += 1
                    if len(t) == 2 and t != "--":
                        bi += 1
                    if len(t) == 3:
                        tri += 1
                # punctuation count
                punc = 0
                for c in target:
                    if c in string.punctuation and c != '"':
                        punc += 1
                for i in scores.values():
                    hold.append(i)
                sentiment_data[x][y] = hold
                hold.append(uni)
                hold.append(bi)
                hold.append(tri)
                hold.append(punc)
                # include the entity count in array
                hold.append(len(nltk.word_tokenize(target)))
    counter += 1
print(hold)
print(sentiment_data.iloc[2:50])


# After the Process of Sentiment Extraction, the data is ready to be used for training
# A new dataset will be built using new column headings
# Source, Length, Word Count, Positive, Negative, Neutral, Compound

clean_data = pd.DataFrame(columns=['Source', 'Length', 'Word Count', 'Positive', 'Negative', 'Neutral', 'Compound', 'Label', 'Avg WordLen', 'UniGrams', 'BiGrams', 'TriGrams', 'Punctuation', 'Entities'])
for x in sentiment_data.columns:
    if x != 'Date' and x != 'Label':
        for y in range(5):
            hold = sentiment_data[x][y]
            if type(hold) == list:
                set = {'Source': x[3:], 'Length': hold[0], 'Word Count': hold[1], 'Positive': hold[2], 'Negative': hold[3], 'Neutral': hold[4], 'Compound': hold[5], 'Label': sentiment_data['Label'][y], 'Avg WordLen': int(hold[0]/hold[1]), 'UniGrams': int(hold[6]), 'BiGrams': int(hold[7]), 'TriGrams': int(hold[8]), 'Punctuation': int(hold[9]), 'Entities': int(hold[10])}
                clean_data.loc[len(clean_data)] = set
print(clean_data.head())

print("Finished without failure")