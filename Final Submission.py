import sklearn
import xgboost as xgb
import pandas as pd
import nltk
import spacy
import string


from spacy import displacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.inspection import permutation_importance
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.naive_bayes import GaussianNB
from xgboost import plot_tree

# Run on first use on new device
'''nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')'''
NER = spacy.load("en_core_web_sm")


pd.options.mode.chained_assignment = None  # default='warn'

# Load the model into pandas
first_data = pd.read_csv('Combined_News_DJIA.csv')
# Using a ready-made sentiment extractor to test against my model


# Use Pandas to iterate over the data entries
# Build a new pandas dataframe with the sentiment scores + length
sentiment_data = first_data.copy()
# Since the original dataset is too large, I will give to option to test on the first 50
def trainData():
    counter = 0
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
                    hold.append(counter)
                    hold.append(len(target))
                    hold.append(len(nltk.word_tokenize(target)))
                    # totaling ngrams in text
                    sent = target.split()
                    uni, bi, tri = 0, 0, 0
                    for t in sent:
                        if len(t) == 1 and t != '-':
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
                    doc = NER(target)
                    hold.append(len(doc.ents))
                    hold.append(first_data.iloc[y]['Label'])
                    # The hold temporarily assumes all metrics taken about sentiment. Then passed on to the final set
                    sentiment_data.at[counter, x] = hold
                else:
                    print("Error: ", target)
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
                    dataPoints = {'Source': hold[0], 'Length': hold[1], 'Word Count': hold[2], 'Positive': hold[3], 'Negative': hold[4], 'Neutral': hold[5], 'Compound': hold[6], 'Avg WordLen': int(hold[1] / hold[2]), 'UniGrams': int(hold[7]), 'BiGrams': int(hold[8]), 'TriGrams': int(hold[9]), 'Punctuation': int(hold[10]), 'Entities': int(hold[11]), 'Label': int(hold[12])}
                    clean_data.loc[len(clean_data)] = dataPoints
    print("result ratio: ",  clean_data['Label'].value_counts(normalize=True))
    return clean_data


def regularClassifier(clf, X, Y, Xtrain, Ytrain, Xtest, Ytest):
    clf.fit(Xtrain, Ytrain)
    Ypred = clf.predict(Xtest)
    k_folds = KFold(n_splits=10)
    scores = cross_val_score(clf, X, Y, cv=k_folds)
    pVal = sklearn.model_selection.permutation_test_score(clf, X, Y, cv=k_folds, n_permutations=100, n_jobs=2)
    #print(scores, scores.mean(), "\n standard deviation: ", scores.std(), "\n  Variance: ", scores.var(),
    #      "\n Confusion Matrix: ", metrics.confusion_matrix(Ytest, Ypred),  "\n Permutation Value: ", pVal)
    # P-Value removed for time constraints
    print(scores, scores.mean(), "\n standard deviation: ", scores.std(), "\n  Variance: ", scores.var(),
          "\n Confusion Matrix: ", metrics.confusion_matrix(Ytest, Ypred),  "\n Permutation Value: ")
    print("\n classification report: ", metrics.classification_report(Ytest, Ypred))
    # Plot the ROC curve
    ax = plt.gca()
    clf_disp = RocCurveDisplay.from_estimator(clf, Xtest, Ytest, ax=ax, alpha=0.8)
    clf_disp.plot(ax=ax, alpha=0.8)
    plt.show()
    # Plot the permutation importance
    result = permutation_importance(clf, Xtest, Ytest, n_repeats=10, random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()
    plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=Xtest.columns[sorted_idx])
    plt.show()

# Program training section: Choose a training model, returns k-fold validation method results
def modelTrainer(clean_data):
    X = clean_data[['Source', 'Length', 'Word Count', 'Positive', 'Negative', 'Neutral', 'Compound', 'Avg WordLen', 'UniGrams', 'BiGrams', 'TriGrams', 'Punctuation', 'Entities']]
    Y = clean_data['Label']
    #run the classifiers a number of times to get a better average
    for i in range(1):
        # Shuffle the data to remove any structural bias
        clean_data = clean_data.sample(frac=1).reset_index(drop=True)
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, test_size=.3, random_state=0)
        # NB first
        print("Naive Bayes Results: ")
        clf = GaussianNB()
        clf.fit(Xtrain, Ytrain)
        regularClassifier(clf, X, Y, Xtrain, Ytrain, Xtest, Ytest)
        # XGB
        print("XGBoost Results: ")
        clf = xgb.XGBClassifier()
        clf.fit(Xtrain, Ytrain)
        plot_tree(clf)
        plt.show()
        regularClassifier(clf, X, Y, Xtrain, Ytrain, Xtest, Ytest)
        # RF
        print("Random Forest Results: ")
        clf = RandomForestClassifier()
        clf.fit(Xtrain, Ytrain)
        regularClassifier(clf, X, Y, Xtrain, Ytrain, Xtest, Ytest)
        # no plot_tree method for RF due to the ensemble nature
        # SVM
        print("SVM Results: ")
        clf = SVC(kernel='rbf', C=1, gamma='auto')
        clf.fit(Xtrain, Ytrain)
        regularClassifier(clf, X, Y, Xtrain, Ytrain, Xtest, Ytest)
        # Ensemble classifier
        print("Ensemble Results: ")
        model_1 = GaussianNB()
        model_2 = xgb.XGBClassifier()
        model_3 = RandomForestClassifier()
        model_4 = SVC(kernel='rbf', C=1, gamma='auto')
        final_model = VotingClassifier(estimators=[('NB', model_1), ('xgb', model_2), ('rf', model_3), ('svm', model_4)], voting='hard')
        final_model.fit(Xtrain, Ytrain)
        k_folds = KFold(n_splits=10)
        scores = cross_val_score(final_model, X, Y, cv=k_folds)
        print(scores, scores.mean(), "\n standard deviation: ", scores.std(), "\n  Variance: ", scores.var())
        pred_final = final_model.predict(Xtest)
        print(mean_squared_error(Ytest, pred_final))
        print(mean_absolute_error(Ytest, pred_final))
        print("Confusion Matrix: \n", metrics.confusion_matrix(Ytest, pred_final))
        print("\n classification report: ", metrics.classification_report(Ytest, pred_final))
        # P-Value removed for time constraints
        #print("Permutation Value: ", sklearn.model_selection.permutation_test_score(final_model, X, Y, cv=k_folds, n_permutations=100, n_jobs=2))
        # Plot the permutation importance
        result = permutation_importance(final_model, Xtest, Ytest, n_repeats=10, random_state=42, n_jobs=2)
        sorted_idx = result.importances_mean.argsort()
        plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=Xtest.columns[sorted_idx])
        plt.show()

def main():
    index = trainData()
    newFrame = frameBuilder(index)
    modelTrainer(newFrame)

main()

print("Finished without failure")