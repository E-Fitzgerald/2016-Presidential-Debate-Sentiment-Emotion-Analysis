#import packages
import pandas as pd
from sklearn.metrics import classification_report, recall_score, precision_score
from Classifiers.tfidf_vectorizer_classifier import vectorize_training_tfidf, get_classifier_tfidf_RandomForest
import seaborn as sns
import matplotlib.pyplot as plt
import random
from PIL import Image
import numpy as np
from os import path, getcwd
from wordcloud import WordCloud, ImageColorGenerator
import warnings
warnings.filterwarnings('ignore')


d = getcwd()

#import data
#"""
training_tweets_excel = pd.ExcelFile('C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\Sentiment_Training_Data_Larger_Processed.xlsx')
training_tweets_sentiment = training_tweets_excel.parse("Sheet1")
test_tweets_excel = pd.ExcelFile('C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\Sentiment_Test_Data_Larger_Processed.xlsx')
test_tweets_sentiment = test_tweets_excel.parse("Sheet1")

training_tweets_excel = pd.ExcelFile('C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\Joy_Training_Data_Processed.xlsx')
training_tweets_joy = training_tweets_excel.parse("Sheet1")
test_tweets_excel = pd.ExcelFile('C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\Joy_Test_Data_Processed.xlsx')
test_tweets_joy = test_tweets_excel.parse("Sheet1")

training_tweets_excel = pd.ExcelFile('C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\Anger_Training_Data_Processed.xlsx')
training_tweets_anger = training_tweets_excel.parse("Sheet1")
test_tweets_excel = pd.ExcelFile('C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\Anger_Test_Data_Processed.xlsx')
test_tweets_anger = test_tweets_excel.parse("Sheet1")

training_tweets_excel = pd.ExcelFile('C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\Support_Training_Data_Processed.xlsx')
training_tweets_support = training_tweets_excel.parse("Sheet1")
test_tweets_excel = pd.ExcelFile('C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\Support_Test_Data_Processed.xlsx')
test_tweets_support = test_tweets_excel.parse("Sheet1")


#"""



#"""
"""
training_tweets_excel = pd.ExcelFile('C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\Waste\Combo_1_2.xlsx')
training_tweets = training_tweets_excel.parse("Sheet1")
test_tweets_excel = pd.ExcelFile('C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\Waste\PreseDe3_Complete_Processed_Stemmed_no_user.xls')
test_tweets = test_tweets_excel.parse("Sheet1")
#"""

##################################################################################################################################using TFIDF Vectorizer, RandomForest, Sentiment

tranformed_training, transformed_test, dictionary_wordcloud = vectorize_training_tfidf(training_tweets_sentiment, test_tweets_sentiment)
tfidf_vectorizer_classifier_ = get_classifier_tfidf_RandomForest(tranformed_training, training_tweets_sentiment["Sentiment"])
prediction_tfidf = tfidf_vectorizer_classifier_.predict(transformed_test)

print("TFIDF Vectorizer, RandomForest Classifier, Sentiment \n", classification_report(test_tweets_sentiment["Sentiment"], prediction_tfidf))

recall_array = recall_score(test_tweets_sentiment["Sentiment"], prediction_tfidf, average=None)
precision_array = precision_score(test_tweets_sentiment["Sentiment"], prediction_tfidf, average=None)


sns.set(style="darkgrid")

fig_sentiment, (plot1, plot2) = plt.subplots(ncols=2, nrows=1, sharey="all")
recall_ax = sns.barplot(x=[-3,-2,-1,0,1,2,3], y=recall_array, ax=plot1, palette="RdYlGn")
precision_ax = sns.barplot(x=[-3,-2,-1,0,1,2,3], y=precision_array, ax=plot2, palette="RdYlGn")

recall_ax.set(xlabel = "Classifications", ylabel = "Recall Score")
recall_ax.set_title('Sentiment Recall')
precision_ax.set(xlabel = "Classifications", ylabel = "Precision Score")
precision_ax.set_title('Sentiment Precision')


def sentiment_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(%d, %d%%, %d%%)" % (random.randint(0, 95), random.randint(20, 100), random.randint(20, 70))
mask = np.array(Image.open(path.join(d, "brain-clip.png")))
wordcloud = WordCloud(background_color='white',  mask=mask).generate_from_frequencies(dictionary_wordcloud)
cloud_sentiment = plt.figure()
plt.imshow(wordcloud.recolor(color_func=sentiment_color_func), interpolation='bilinear')
plt.axis("off")




##################################################################################################################################using TFIDF Vectorizer, RandomForest, Joy
tranformed_training, transformed_test, dictionary_wordcloud = vectorize_training_tfidf(training_tweets_joy, test_tweets_joy)
tfidf_vectorizer_classifier_ = get_classifier_tfidf_RandomForest(tranformed_training, training_tweets_joy["Joy"])
prediction_tfidf = tfidf_vectorizer_classifier_.predict(transformed_test)

print("TFIDF Vectorizer, RandomForest Classifier, Joy \n", classification_report(test_tweets_joy["Joy"], prediction_tfidf))



recall_array = recall_score(test_tweets_joy["Joy"], prediction_tfidf, average=None)
precision_array = precision_score(test_tweets_joy["Joy"], prediction_tfidf, average=None)


sns.set(style="darkgrid")

fig_joy, (plot1, plot2) = plt.subplots(ncols=2, nrows=1, sharey="all")
recall_ax = sns.barplot(x=[0,1,2,3], y=recall_array, ax=plot1, palette="YlGn")
precision_ax = sns.barplot(x=[0,1,2,3], y=precision_array, ax=plot2, palette="YlGn")

recall_ax.set(xlabel = "Classifications", ylabel = "Recall Score")
recall_ax.set_title('Joy Recall')
precision_ax.set(xlabel = "Classifications", ylabel = "Precision Score")
precision_ax.set_title('Joy Precision')


def joy_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(%d, %d%%, %d%%)" % (random.randint(55, 120), random.randint(20, 70), random.randint(20, 50))
mask = np.array(Image.open(path.join(d, "joy-cloud.png")))
wordcloud = WordCloud(background_color='white',  mask=mask).generate_from_frequencies(dictionary_wordcloud)
cloud_joy = plt.figure()
plt.imshow(wordcloud.recolor(color_func=joy_color_func), interpolation='bilinear')
plt.axis("off")




##################################################################################################################################using TFIDF Vectorizer, RandomForest, Anger
tranformed_training, transformed_test, dictionary_wordcloud = vectorize_training_tfidf(training_tweets_anger, test_tweets_anger)
tfidf_vectorizer_classifier_ = get_classifier_tfidf_RandomForest(tranformed_training, training_tweets_anger["Anger"])
prediction_tfidf = tfidf_vectorizer_classifier_.predict(transformed_test)

print("TFIDF Vectorizer, RandomForest Classifier, Anger \n", classification_report(test_tweets_anger["Anger"], prediction_tfidf))


recall_array = recall_score(test_tweets_anger["Anger"], prediction_tfidf, average=None)
precision_array = precision_score(test_tweets_anger["Anger"], prediction_tfidf, average=None)


sns.set(style="darkgrid")

fig_anger, (plot1, plot2) = plt.subplots(ncols=2, nrows=1, sharey="all")
recall_ax = sns.barplot(x=[0,1,2,3], y=recall_array, ax=plot1, palette="OrRd")
precision_ax = sns.barplot(x=[0,1,2,3], y=precision_array, ax=plot2, palette="OrRd")

recall_ax.set(xlabel = "Classifications", ylabel = "Recall Score")
recall_ax.set_title('Anger Recall')
precision_ax.set(xlabel = "Classifications", ylabel = "Precision Score")
precision_ax.set_title('Anger Precision')



def anger_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(%d, %d%%, %d%%)" % (random.randint(0, 35), 100, random.randint(20, 70))
mask = np.array(Image.open(path.join(d, "anger-character.png")))
wordcloud = WordCloud(background_color='white',  mask=mask).generate_from_frequencies(dictionary_wordcloud)
image_colors = ImageColorGenerator(mask)
cloud_anger = plt.figure()
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation='bilinear')
plt.axis("off")


##################################################################################################################################using TFIDF Vectorizer, RandomForest, Support Trump
tranformed_training, transformed_test, dictionary_wordcloud = vectorize_training_tfidf(training_tweets_support, test_tweets_support)
tfidf_vectorizer_classifier_ = get_classifier_tfidf_RandomForest(tranformed_training, training_tweets_support["Support Trump"])
prediction_tfidf = tfidf_vectorizer_classifier_.predict(transformed_test)

print("TFIDF Vectorizer, RandomForest Classifier, Support Trump \n", classification_report(test_tweets_support["Support Trump"], prediction_tfidf))


recall_array = recall_score(test_tweets_support["Support Trump"], prediction_tfidf, average=None)
precision_array = precision_score(test_tweets_support["Support Trump"], prediction_tfidf, average=None)


sns.set(style="darkgrid")
two_party_scheme = ["#1024cb", "#a021d7", "#f53006"]

fig_support, (plot1, plot2) = plt.subplots(ncols=2, nrows=1, sharey="all")
recall_ax = sns.barplot(x=[-1,0,1], y=recall_array, ax=plot1, palette=two_party_scheme)
precision_ax = sns.barplot(x=[-1,0,1], y=precision_array, ax=plot2, palette=two_party_scheme)

recall_ax.set(xlabel = "Classifications", ylabel = "Recall Score")
recall_ax.set_title('Support Trump Recall')
precision_ax.set(xlabel = "Classifications", ylabel = "Precision Score")
precision_ax.set_title('Support Trump Precision')


def support_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(%d, %d%%, %d%%)" % (random.randint(245, 360), 100, random.randint(20, 70))
mask = np.array(Image.open(path.join(d, "support-us.png")))
wordcloud = WordCloud(background_color='white',  mask=mask).generate_from_frequencies(dictionary_wordcloud)
cloud_support = plt.figure()
plt.imshow(wordcloud.recolor(color_func=support_color_func), interpolation='bilinear')
plt.axis("off")


plt.show()