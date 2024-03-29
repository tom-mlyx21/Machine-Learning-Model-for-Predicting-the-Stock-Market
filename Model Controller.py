import xgboost as xgb
import pandas as pd
import nltk
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import mean_squared_error, log_loss

# Run on first use on new device
'''nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')'''


pd.options.mode.chained_assignment = None  # default='warn'

# Load the model into pandas
first_data = pd.read_csv('Combined_News_DJIA.csv')
print(first_data)
# Using a ready-made sentiment extractor to test against my model


# Use Pandas to iterate over the data entries
# Build a new pandas dataframe with the sentiment scores + length
sentiment_data = first_data.copy()
# Since the original dataset is too large, I will give to option to test on the first 50
def trainData():
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
            if x != 'Date':
                target = first_data[x][y]
                hold = []
                # Can't encode floats so only strings accepted (Loss of 1 in 250 ish entries)
                if type(target) == str:
                    scores = SentimentIntensityAnalyzer().polarity_scores(target)
                    # Migrate the vader object to an array, to better suit pd
                    hold.append(counter)
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
                    # Sentiment scores
                    for i in scores.values():
                        hold.append(i)
                    hold.append(uni)
                    hold.append(bi)
                    hold.append(tri)
                    hold.append(punc)
                    # include the entity count in array
                    hold.append(len(nltk.word_tokenize(target)))
                    hold.append(first_data.iloc[y]['Label'])
                    # The hold temporarily assumes all metrics taken about sentiment. Then passed on to the
                    sentiment_data.at[counter, x] = hold
        counter += 1
    return boundary

# After the Process of Sentiment Extraction, the data is ready to be used for training
# A new dataset will be built using new column headings
# Source, Length, Word Count, Positive, Negative, Neutral, Compound
def frameBuilder(boundary):
    clean_data = pd.DataFrame(columns=['Source', 'Length', 'Word Count', 'Positive', 'Negative', 'Neutral', 'Compound', 'Avg WordLen', 'UniGrams', 'BiGrams', 'TriGrams', 'Punctuation', 'Entities', 'Label'])
    for x in sentiment_data.columns:
        if x != 'Date' and x != 'Label':
            for y in range(boundary):
                hold = sentiment_data[x][y]
                if type(hold) == list:
                    set = {'Source': hold[0], 'Length': hold[1], 'Word Count': hold[2], 'Positive': hold[3], 'Negative': hold[4], 'Neutral': hold[5], 'Compound': hold[6], 'Avg WordLen': int(hold[1]/hold[2]), 'UniGrams': int(hold[7]), 'BiGrams': int(hold[8]), 'TriGrams': int(hold[9]), 'Punctuation': int(hold[10]), 'Entities': int(hold[11]), 'Label': int(hold[12])}
                    clean_data.loc[len(clean_data)] = set
    print("clean data is: ", clean_data)
    return clean_data

# Program training section: Choose a training model, returns k-fold validation method results
def modelTrainer(clean_data):
    model = input("Enter the model to be used for training (NB, XGB, RF, SVM, EN): ")
    X = clean_data[['Source', 'Length', 'Word Count', 'Positive', 'Negative', 'Neutral', 'Compound', 'Avg WordLen', 'UniGrams', 'BiGrams', 'TriGrams', 'Punctuation', 'Entities']]
    Y = clean_data['Label']
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, test_size=.3, random_state=0)
    if model == 'NB':
        clf = GaussianNB()
    elif model == 'XGB':
        clf = xgb.XGBClassifier()
    elif model == 'RF':
        clf = RandomForestClassifier()
    elif model == 'SVM':
        clf = SVC(kernel='rbf', C=1, gamma='auto')
    elif model == 'EN':
        # Ensemble classifier
        model_1 = GaussianNB()
        model_2 = xgb.XGBClassifier()
        model_3 = RandomForestClassifier()
        model_4 = SVC(kernel='rbf', C=1, gamma='auto')
        final_model = VotingClassifier(
            estimators=[('NB', model_1), ('xgb', model_2), ('rf', model_3), ('svm', model_4)], voting='hard')
        final_model.fit(Xtrain, Ytrain)
        pred_final = final_model.predict(Xtest)
        print(mean_squared_error(Ytest, pred_final))
        print(log_loss(Ytest, pred_final))

    if model == 'NB' or model == 'XGB' or model == 'RF' or model == 'SVM':
        clf.fit(Xtrain, Ytrain)
        Ypred = clf.predict(Xtest)
        print(classification_report(Ytest, Ypred))
        k_folds = KFold(n_splits=10)
        scores = cross_val_score(clf, X, Y, cv=k_folds)
        print(scores, scores.mean())

    elif model != 'EN':
        print("Enter a Valid Model")
        modelTrainer(clean_data)

def main():
    index = trainData()
    newFrame = frameBuilder(index)
    modelTrainer(newFrame)

main()

print("Finished without failure")