#import packages
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
#import Classifiers
from Classifiers.numerical_features_classifier import getFeatureVector, get_classifier_numerical_LinearSVC, get_classifier_numerical_GaussianNB, \
    get_classifier_numerical_RandomForest, get_classifier_numerical_DecisionTree, get_classifier_numerical_AdaBoostClassifier, get_classifier_numerical_RidgeClassifier, \
    get_classifier_numerical_PassiveAggressiveClassifier, get_classifier_numerical_Perceptron
from Classifiers.tfidf_vectorizer_classifier import vectorize_training_tfidf, get_classifier_tfidf_LinearSVC, get_classifier_tfidf_GaussianNB, \
    get_classifier_tfidf_RandomForest, get_classifier_tfidf_DecisionTree, get_classifier_tfidf_AdaBoostClassifier, get_classifier_tfidf_RidgeClassifier, \
    get_classifier_tfidf_PassiveAggressiveClassifier, get_classifier_tfidf_Perceptron
from Classifiers.count_vectorizer_classifier import vectorize_training_count, get_classifier_count_LinearSVC, get_classifier_count_GaussianNB, \
    get_classifier_count_RandomForest, get_classifier_count_DecisionTree, get_classifier_count_AdaBoostClassifier, get_classifier_count_RidgeClassifier, \
    get_classifier_count_PassiveAggressiveClassifier, get_classifier_count_Perceptron
from Classifiers.Vader_Classifier import get_Vader_Classifications

#import data
training_tweets_excel = pd.ExcelFile('C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\Sentiment_Training_Data_Larger_Processed.xlsx')
training_tweets = training_tweets_excel.parse("Sheet1")
test_tweets_excel = pd.ExcelFile('C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\Sentiment_Test_Data_Larger_Processed.xlsx')
test_tweets = test_tweets_excel.parse("Sheet1")
pos_words = np.loadtxt('C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\\bing_positive.txt', dtype="str")
neg_words = np.loadtxt('C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\\bing_negative.txt', dtype="str")


######################################################################################################################################################################## LinearSVC
#using TFIDF Vectorizer, LinearSVC
tranformed_training, transformed_test, dicti = vectorize_training_tfidf(training_tweets, test_tweets)
tfidf_vectorizer_classifier_ = get_classifier_tfidf_LinearSVC(tranformed_training, training_tweets["Sentiment"])
prediction_tfidf = tfidf_vectorizer_classifier_.predict(transformed_test)

print("Test: TFIDF Vectorizer, LinearSVC Classifier \n", classification_report(test_tweets["Sentiment"], prediction_tfidf))

#using All Numerical Features, LinearSVC
numerical_classifer = get_classifier_numerical_LinearSVC(training_tweets, training_tweets["Sentiment"], pos_words, neg_words)
featureListTest = []
for k in range(0, len(test_tweets["Tweet"]), 1):
    featureVector = getFeatureVector(test_tweets["Tweet"][k], pos_words, neg_words)
    featureListTest.append(featureVector)
prediction_numerical = numerical_classifer.predict(featureListTest)

print("Test: All Numerical Features, LinearSVC Classifier\n", classification_report(test_tweets["Sentiment"], prediction_numerical))

#using Count Vectorizer, LinearSVC
tranformed_training, transformed_test = vectorize_training_count(training_tweets, test_tweets)
count_vectorizer_classifier_ = get_classifier_count_LinearSVC(tranformed_training, training_tweets["Sentiment"])
prediction_count = count_vectorizer_classifier_.predict(transformed_test)

print("Test: Count Vectorizer, LinearSVC Classifier\n", classification_report(test_tweets["Sentiment"], prediction_count))

######################################################################################################################################################################## GaussianNB
#using TFIDF Vectorizer, GaussianNB
tranformed_training, transformed_test, dicti = vectorize_training_tfidf(training_tweets, test_tweets)
tfidf_vectorizer_classifier_ = get_classifier_tfidf_GaussianNB(tranformed_training, training_tweets["Sentiment"])
prediction_tfidf = tfidf_vectorizer_classifier_.predict(transformed_test.toarray())

print("Test: TFIDF Vectorizer, GaussianNB Classifier \n", classification_report(test_tweets["Sentiment"], prediction_tfidf))


#using All Numerical Features, GaussianNB
numerical_classifer = get_classifier_numerical_GaussianNB(training_tweets, training_tweets["Sentiment"], pos_words, neg_words)
featureListTest = []
for k in range(0, len(test_tweets["Tweet"]), 1):
    featureVector = getFeatureVector(test_tweets["Tweet"][k], pos_words, neg_words)
    featureListTest.append(featureVector)
prediction_numerical = numerical_classifer.predict(featureListTest)

print("Test: All Numerical Features, GaussianNB Classifier\n", classification_report(test_tweets["Sentiment"], prediction_numerical))

#using Count Vectorizer, GaussianNB
tranformed_training, transformed_test = vectorize_training_count(training_tweets, test_tweets)
count_vectorizer_classifier_ = get_classifier_count_GaussianNB(tranformed_training, training_tweets["Sentiment"])
prediction_count = count_vectorizer_classifier_.predict(transformed_test.toarray())

print("Test: Count Vectorizer, GaussianNB Classifier \n", classification_report(test_tweets["Sentiment"], prediction_count))


######################################################################################################################################################################## RandomForest
#using TFIDF Vectorizer, RandomForest
tranformed_training, transformed_test, dicti = vectorize_training_tfidf(training_tweets, test_tweets)
tfidf_vectorizer_classifier_ = get_classifier_tfidf_RandomForest(tranformed_training, training_tweets["Sentiment"])
prediction_tfidf = tfidf_vectorizer_classifier_.predict(transformed_test)

print("Test: TFIDF Vectorizer, RandomForest Classifier \n", classification_report(test_tweets["Sentiment"], prediction_tfidf))

#using All Numerical Features, RandomForest
numerical_classifer = get_classifier_numerical_RandomForest(training_tweets, training_tweets["Sentiment"], pos_words, neg_words)
featureListTest = []
for k in range(0, len(test_tweets["Tweet"]), 1):
    featureVector = getFeatureVector(test_tweets["Tweet"][k], pos_words, neg_words)
    featureListTest.append(featureVector)
prediction_numerical = numerical_classifer.predict(featureListTest)

print("Test: All Numerical Features, RandomForest Classifier\n", classification_report(test_tweets["Sentiment"], prediction_numerical))

#using Count Vectorizer, RandomForest
tranformed_training, transformed_test = vectorize_training_count(training_tweets, test_tweets)
count_vectorizer_classifier_ = get_classifier_count_RandomForest(tranformed_training, training_tweets["Sentiment"])
prediction_count = count_vectorizer_classifier_.predict(transformed_test)

print("Test: Count Vectorizer, RandomForest Classifier\n", classification_report(test_tweets["Sentiment"], prediction_count))

######################################################################################################################################################################## DecisionTree
#using TFIDF Vectorizer, DecisionTree
tranformed_training, transformed_test, dicti = vectorize_training_tfidf(training_tweets, test_tweets)
tfidf_vectorizer_classifier_ = get_classifier_tfidf_DecisionTree(tranformed_training, training_tweets["Sentiment"])
prediction_tfidf = tfidf_vectorizer_classifier_.predict(transformed_test)

print("Test: TFIDF Vectorizer, DecisionTree Classifier \n", classification_report(test_tweets["Sentiment"], prediction_tfidf))

#using All Numerical Features, DecisionTree
numerical_classifer = get_classifier_numerical_DecisionTree(training_tweets, training_tweets["Sentiment"], pos_words, neg_words)
featureListTest = []
for k in range(0, len(test_tweets["Tweet"]), 1):
    featureVector = getFeatureVector(test_tweets["Tweet"][k], pos_words, neg_words)
    featureListTest.append(featureVector)
prediction_numerical = numerical_classifer.predict(featureListTest)

print("Test: All Numerical Features, DecisionTree Classifier\n", classification_report(test_tweets["Sentiment"], prediction_numerical))

#using Count Vectorizer, DecisionTree
tranformed_training, transformed_test = vectorize_training_count(training_tweets, test_tweets)
count_vectorizer_classifier_ = get_classifier_count_DecisionTree(tranformed_training, training_tweets["Sentiment"])
prediction_count = count_vectorizer_classifier_.predict(transformed_test)

print("Test: Count Vectorizer, DecisionTree Classifier\n", classification_report(test_tweets["Sentiment"], prediction_count))

######################################################################################################################################################################## Vader
#Vader Classifier
vader_predictions = get_Vader_Classifications(test_tweets["Tweet"])

print("Test: Vader Classifier,  \n", classification_report(test_tweets["Sentiment"], vader_predictions))



################################################################### AdaBoostClassifier
#using TFIDF Vectorizer, AdaBoostClassifier
tranformed_training, transformed_test, dicti = vectorize_training_tfidf(training_tweets, test_tweets)
tfidf_vectorizer_classifier_ = get_classifier_tfidf_AdaBoostClassifier(tranformed_training, training_tweets["Sentiment"])
prediction_tfidf = tfidf_vectorizer_classifier_.predict(transformed_test.toarray())

print("Test: TFIDF Vectorizer, AdaBoostClassifier \n", classification_report(test_tweets["Sentiment"], prediction_tfidf))


#using All Numerical Features, AdaBoostClassifier
numerical_classifer = get_classifier_numerical_AdaBoostClassifier(training_tweets, training_tweets["Sentiment"], pos_words, neg_words)
featureListTest = []
for k in range(0, len(test_tweets["Tweet"]), 1):
    featureVector = getFeatureVector(test_tweets["Tweet"][k], pos_words, neg_words)
    featureListTest.append(featureVector)
prediction_numerical = numerical_classifer.predict(featureListTest)

print("Test: All Numerical Features, AdaBoostClassifier\n", classification_report(test_tweets["Sentiment"], prediction_numerical))

#using Count Vectorizer, AdaBoostClassifier
tranformed_training, transformed_test = vectorize_training_count(training_tweets, test_tweets)
count_vectorizer_classifier_ = get_classifier_count_AdaBoostClassifier(tranformed_training, training_tweets["Sentiment"])
prediction_count = count_vectorizer_classifier_.predict(transformed_test.toarray())

print("Test: Count Vectorizer, AdaBoostClassifier \n", classification_report(test_tweets["Sentiment"], prediction_count))

######################################################################################################################################################################## RidgeClassifier
#using TFIDF Vectorizer, RidgeClassifier
tranformed_training, transformed_test, dicti = vectorize_training_tfidf(training_tweets, test_tweets)
tfidf_vectorizer_classifier_ = get_classifier_tfidf_RidgeClassifier(tranformed_training, training_tweets["Sentiment"])
prediction_tfidf = tfidf_vectorizer_classifier_.predict(transformed_test.toarray())

print("Test: TFIDF Vectorizer, RidgeClassifier \n", classification_report(test_tweets["Sentiment"], prediction_tfidf))


#using All Numerical Features, RidgeClassifier
numerical_classifer = get_classifier_numerical_RidgeClassifier(training_tweets, training_tweets["Sentiment"], pos_words, neg_words)
featureListTest = []
for k in range(0, len(test_tweets["Tweet"]), 1):
    featureVector = getFeatureVector(test_tweets["Tweet"][k], pos_words, neg_words)
    featureListTest.append(featureVector)
prediction_numerical = numerical_classifer.predict(featureListTest)

print("Test: All Numerical Features, RidgeClassifier\n", classification_report(test_tweets["Sentiment"], prediction_numerical))

#using Count Vectorizer, RidgeClassifier
tranformed_training, transformed_test = vectorize_training_count(training_tweets, test_tweets)
count_vectorizer_classifier_ = get_classifier_count_RidgeClassifier(tranformed_training, training_tweets["Sentiment"])
prediction_count = count_vectorizer_classifier_.predict(transformed_test.toarray())

print("Test: Count Vectorizer, RidgeClassifier \n", classification_report(test_tweets["Sentiment"], prediction_count))

######################################################################################################################################################################## RidgeClassifier
#using TFIDF Vectorizer, PassiveAggressiveClassifier
tranformed_training, transformed_test, dicti = vectorize_training_tfidf(training_tweets, test_tweets)
tfidf_vectorizer_classifier_ = get_classifier_tfidf_PassiveAggressiveClassifier(tranformed_training, training_tweets["Sentiment"])
prediction_tfidf = tfidf_vectorizer_classifier_.predict(transformed_test.toarray())

print("Test: TFIDF Vectorizer, PassiveAggressiveClassifier \n", classification_report(test_tweets["Sentiment"], prediction_tfidf))


#using All Numerical Features, PassiveAggressiveClassifier
numerical_classifer = get_classifier_numerical_PassiveAggressiveClassifier(training_tweets, training_tweets["Sentiment"], pos_words, neg_words)
featureListTest = []
for k in range(0, len(test_tweets["Tweet"]), 1):
    featureVector = getFeatureVector(test_tweets["Tweet"][k], pos_words, neg_words)
    featureListTest.append(featureVector)
prediction_numerical = numerical_classifer.predict(featureListTest)

print("Test: All Numerical Features, PassiveAggressiveClassifier\n", classification_report(test_tweets["Sentiment"], prediction_numerical))

#using Count Vectorizer, PassiveAggressiveClassifier
tranformed_training, transformed_test = vectorize_training_count(training_tweets, test_tweets)
count_vectorizer_classifier_ = get_classifier_count_PassiveAggressiveClassifier(tranformed_training, training_tweets["Sentiment"])
prediction_count = count_vectorizer_classifier_.predict(transformed_test.toarray())

print("Test: Count Vectorizer, PassiveAggressiveClassifier \n", classification_report(test_tweets["Sentiment"], prediction_count))


######################################################################################################################################################################## Perceptron
#using TFIDF Vectorizer, Perceptron
tranformed_training, transformed_test, dicti = vectorize_training_tfidf(training_tweets, test_tweets)
tfidf_vectorizer_classifier_ = get_classifier_tfidf_Perceptron(tranformed_training, training_tweets["Sentiment"])
prediction_tfidf = tfidf_vectorizer_classifier_.predict(transformed_test.toarray())

print("Test: TFIDF Vectorizer, Perceptron Classifier \n", classification_report(test_tweets["Sentiment"], prediction_tfidf))


#using All Numerical Features, Perceptron
numerical_classifer = get_classifier_numerical_Perceptron(training_tweets, training_tweets["Sentiment"], pos_words, neg_words)
featureListTest = []
for k in range(0, len(test_tweets["Tweet"]), 1):
    featureVector = getFeatureVector(test_tweets["Tweet"][k], pos_words, neg_words)
    featureListTest.append(featureVector)
prediction_numerical = numerical_classifer.predict(featureListTest)

print("Test: All Numerical Features, Perceptron Classifier\n", classification_report(test_tweets["Sentiment"], prediction_numerical))

#using Count Vectorizer, Perceptron
tranformed_training, transformed_test = vectorize_training_count(training_tweets, test_tweets)
count_vectorizer_classifier_ = get_classifier_count_Perceptron(tranformed_training, training_tweets["Sentiment"])
prediction_count = count_vectorizer_classifier_.predict(transformed_test.toarray())

print("Test: Count Vectorizer, Perceptron Classifier \n", classification_report(test_tweets["Sentiment"], prediction_count))