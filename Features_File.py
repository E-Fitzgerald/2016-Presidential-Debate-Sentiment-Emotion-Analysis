import nltk.classify
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk.sentiment
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import emoji
from nltk import ngrams

def extract_emojis(str):
  return ''.join(c for c in str if c in emoji.UNICODE_EMOJI)

def getNegatedTokens(tweet):
    negated_tokens = nltk.sentiment.util.mark_negation(tweet.split())
    return negated_tokens

def getNegationCount(tweet):
    negated_tokens = nltk.sentiment.util.mark_negation(tweet.split())
    count = 0
    for word in negated_tokens:
        if "_NEG" in word:
            count +=1
    return count

def getNGrams(tweet):
    featureVector = []
    #feature: ngrams
    token_words = nltk.word_tokenize(tweet) #or use negated tokens above^^
    n = 2
    for item in ngrams(token_words, n):
        featureVector.append(' '.join(item))
    return featureVector

def getAvgWordLen(tweet):
    #feature: average word length
    length = avg_word(tweet)
    return length

def getNumWords(tweet):
    featureVector = []
    #feature: Number of words
    num_words = str(tweet).split(" ")
    return len(num_words)

def getSingleWords(tweet):
    # feature: Words
    num_words = str(tweet).split(" ")
    return num_words

def getNumChars(tweet):
    #feature: number of characters
    num_chars = len(tweet)
    return num_chars

def getEmojis(tweet):
    featureVector = []
    #feature: emojis
    emoji_string = extract_emojis(tweet)
    return emoji_string

def getNumUpChars(tweet):
    #feature: number of UpperCase Characters
    count = 0
    for word in (nltk.word_tokenize(tweet)):
        for letter in word:
            if (letter.isupper()):
                count += 1
    return count

def getNumUpWords(tweet):
    featureVector = []
    # feature: number of UpperCase Words
    count = 0
    for word in (nltk.word_tokenize(tweet)):
        if (word[0].isupper()):
            if(word == "ATUSER"):
                count = count
            else:
                count += 1
    return count

def getPosWords(tweet, pos_words):
    featureVector = []
    # feature: number of positive words from Bing Liu's Opinion Lexicon
    count_pos = 0
    for word in (nltk.word_tokenize(tweet)):
        if (word in pos_words):
                count_pos += 1
    return count_pos

def getNegWords(tweet, neg_words):
    featureVector = []
    # feature: number of negative words from Bing Liu's Opinion Lexicon
    count_neg = 0
    for word in (nltk.word_tokenize(tweet)):
        if (word in neg_words):
            count_neg += 1
    return count_neg

def getDifPosNegWords(tweet, pos_words, neg_words):
    featureVector = []
    # feature: the difference between the number of positive and negative words from Bing Liu's Opinion Lexicon
    count_neg = 0
    count_pos = 0
    for word in (nltk.word_tokenize(tweet)):
        if (word in pos_words):
                count_pos += 1
        if (word in neg_words):
            count_neg += 1
    difference = count_pos-count_neg
    return difference

def getRatioNegPosWords(tweet, pos_words, neg_words):
    featureVector = []
    # feature: the ratio between the number of negative and positive words from Bing Liu's Opinion Lexicon
    count_neg = 0
    count_pos = 0
    for word in (nltk.word_tokenize(tweet)):
        if (word in pos_words):
                count_pos += 1
        if (word in neg_words):
            count_neg += 1
    if(count_pos ==0):
        count_pos = 1
    ratio = count_neg/count_pos
    return ratio

def getTFSum(tweet, tfidf):
    featureVector = []
    #feature: tf
    tf_total = 0
    for word in nltk.word_tokenize(tweet):
        try:
            tf = tfidf.loc[tfidf['words'] == word, "tf"].iloc[0]
        except IndexError:
            tf = 0
        tf_total += tf
    return tf_total

def getIDFSum(tweet, tfidf):
    featureVector = []
    #feature: idf
    idf_total = 0
    for word in nltk.word_tokenize(tweet):
        try:
            idf = tfidf.loc[tfidf['words'] == word, "idf"].iloc[0]
        except IndexError:
            idf = 0
        idf_total += idf
    return idf_total

def getTFIDFSum(tweet, tfidf):
    featureVector = []
    # feature: tf-idf
    tfidf_total = 0
    for word in nltk.word_tokenize(tweet):
        try:
            tf_idf = tfidf.loc[tfidf['words'] == word, "tfidf"].iloc[0]
        except IndexError:
            tf_idf = 0
        tfidf_total += tf_idf
    return tfidf_total

def getTFArray(tweet, tfidf):
    featureVector = []
    # feature: tf
    for word in nltk.word_tokenize(tweet):
        try:
            tf = tfidf.loc[tfidf['words'] == word, "tf"].iloc[0]
        except IndexError:
            tf = 0
        tf = str(tf)
        featureVector.append(tf)
    return featureVector

def getIDFArray(tweet, tfidf):
    featureVector = []
    # feature: idf
    for word in nltk.word_tokenize(tweet):
        try:
            idf = tfidf.loc[tfidf['words'] == word, "idf"].iloc[0]
        except IndexError:
            idf = 0
        idf = str(idf)
        featureVector.append(idf)
    return featureVector

def getTFIDFArray(tweet, tfidf):
    featureVector = []
    # feature: idf
    for word in nltk.word_tokenize(tweet):
        try:
            tf_idf = tfidf.loc[tfidf['words'] == word, "tfidf"].iloc[0]
        except IndexError:
            tf_idf = 0
        tf_idf = str(tf_idf)
        featureVector.append(tf_idf)
    return featureVector


def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

# end of functions


# Read the tweets one by one and process it

xl = pd.ExcelFile('C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\All_Processed_Tweets.xlsx')
inpTweets = xl.parse("Sheet1")
stopWords = stopwords.words("english")
pos_words = np.loadtxt('C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\\bing_positive.txt', dtype="str")
neg_words = np.loadtxt('C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\\bing_negative.txt', dtype="str")

columns = ["Negated Tweets", "Negation Count", "N-Grams", "Average Word Length", "Number of Words", "Single Words",
           "Number of Characters", "Emojis", "Number of Uppercase Characters", "Number of UpperCase Words",
           "Number of Positive Words from Bing Liu's Opinion Lexicon", "Number of Negative Words from Bing Liu's Opinion Lexicon",
           "Difference Between the Number of Positive and Negative Words from Bing Liu's Opinion Lexicon",
           "Ratio Between the Number of Negative and Positive Words from Bing Liu's Opinion Lexicon",
           "Term Frequency Sum", "Inverse Document Frequency Sum", "TF-IDF Sum", "TF Array"]

feature_df = pd.DataFrame(index = range(0, len(inpTweets["Tweet"])), columns=columns)
feature_df = feature_df.fillna(0)

featureList = []

print("Generating TF-IDF DataFrame...")
tfidf = (inpTweets["Tweet"]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tfidf.columns = ['words','tf']
for i,word in enumerate(tfidf['words']):
    if(i == 0):
        i = i + 1
    tfidf.loc[i, 'idf'] = np.log(inpTweets.shape[0]/(len(inpTweets[inpTweets['Tweet'].str.contains(word, regex = False)])))
tfidf['tfidf'] = tfidf['tf'] * tfidf['idf']




print("Creating the feature vector for Negated Tweets...")
for k in range(0,len(inpTweets["Tweet"]),1):
    featureVector = getNegatedTokens(inpTweets["Tweet"][k])
    str1 = ' '.join(featureVector)
    feature_df.loc[k, "Negated Tweets"] = str1

print("Creating the feature vector for Negated Token Count...")
for k in range(0,len(inpTweets["Tweet"]),1):
    featureCount = getNegationCount(inpTweets["Tweet"][k])
    feature_df.loc[k, "Negation Count"] = int(featureCount)

print("Creating the feature vector for N-Grams...")
for k in range(0,len(inpTweets["Tweet"]),1):
    featureGrams = getNGrams(inpTweets["Tweet"][k])
    str2 = ' / '.join(featureGrams)
    feature_df.loc[k, "N-Grams"] = str2

print("Creating the feature vector for Average Word Length...")
for k in range(0,len(inpTweets["Tweet"]),1):
    featureLen = getAvgWordLen(inpTweets["Tweet"][k])
    feature_df.loc[k, "Average Word Length"] = featureLen

print("Creating the feature vector for Number of Words...")
for k in range(0,len(inpTweets["Tweet"]),1):
    featureNum = getNumWords(inpTweets["Tweet"][k])
    feature_df.loc[k, "Number of Words"] = featureNum

print("Creating the feature vector for Single Words...")
for k in range(0,len(inpTweets["Tweet"]),1):
    featureWords = getSingleWords(inpTweets["Tweet"][k])
    str3 = ' / '.join(featureWords)
    feature_df.loc[k, "Single Words"] = str3

print("Creating the feature vector for Number of Characters...")
for k in range(0,len(inpTweets["Tweet"]),1):
    featureChars = getNumChars(inpTweets["Tweet"][k])
    feature_df.loc[k, "Number of Characters"] = featureChars

print("Creating the feature vector for Emojis...")
for k in range(0,len(inpTweets["Tweet"]),1):
    featureWords = getEmojis(inpTweets["Tweet"][k])
    str4 = ''.join(featureWords)
    feature_df.loc[k, "Emojis"] = str4

print("Creating the feature vector for Number of Uppercase Characters...")
for k in range(0,len(inpTweets["Tweet"]),1):
    featureCount = getNumUpChars(inpTweets["Tweet"][k])
    feature_df.loc[k, "Number of Uppercase Characters"] = featureCount

print("Creating the feature vector for Number of UpperCase Words...")
for k in range(0,len(inpTweets["Tweet"]),1):
    featureCount = getNumUpWords(inpTweets["Tweet"][k])
    feature_df.loc[k, "Number of UpperCase Words"] = featureCount

print("Creating the feature vector for Number of Positive Words from Bing Liu's Opinion Lexicon...")
for k in range(0,len(inpTweets["Tweet"]),1):
    featureCount = getPosWords(inpTweets["Tweet"][k], pos_words)
    feature_df.loc[k, "Number of Positive Words from Bing Liu's Opinion Lexicon"] = featureCount

print("Creating the feature vector for Number of Negative Words from Bing Liu's Opinion Lexicon...")
for k in range(0, len(inpTweets["Tweet"]), 1):
    featureCount = getNegWords(inpTweets["Tweet"][k], neg_words)
    feature_df.loc[k, "Number of Negative Words from Bing Liu's Opinion Lexicon"] = featureCount

print("Creating the feature vector for Difference Between the Number of Positive and Negative Words from Bing Liu's Opinion Lexicon...")
for k in range(0, len(inpTweets["Tweet"]), 1):
    featureCount = getDifPosNegWords(inpTweets["Tweet"][k], pos_words, neg_words)
    feature_df.loc[k, "Difference Between the Number of Positive and Negative Words from Bing Liu's Opinion Lexicon"] = featureCount

print("Creating the feature vector for Ratio Between the Number of Negative and Positive Words from Bing Liu's Opinion Lexicon...")
for k in range(0, len(inpTweets["Tweet"]), 1):
    featureCount = getRatioNegPosWords(inpTweets["Tweet"][k], pos_words, neg_words)
    feature_df.loc[k, "Ratio Between the Number of Negative and Positive Words from Bing Liu's Opinion Lexicon"] = featureCount

print("Creating the feature vector for Term Frequency Sum...")
for k in range(0, len(inpTweets["Tweet"]), 1):
    featureCount = getTFSum(inpTweets["Tweet"][k], tfidf)
    feature_df.loc[k, "Term Frequency Sum"] = featureCount

print("Creating the feature vector for Inverse Document Frequency Sum...")
for k in range(0, len(inpTweets["Tweet"]), 1):
    featureCount = getIDFSum(inpTweets["Tweet"][k], tfidf)
    feature_df.loc[k, "Inverse Document Frequency Sum"] = featureCount

print("Creating the feature vector for TF-IDF Sum...")
for k in range(0, len(inpTweets["Tweet"]), 1):
    featureCount = getTFIDFSum(inpTweets["Tweet"][k], tfidf)
    feature_df.loc[k, "TF-IDF Sum"] = featureCount

print("Creating the feature vector for TF Array...")
for k in range(0, len(inpTweets["Tweet"]), 1):
    featureCount = getTFArray(inpTweets["Tweet"][k], tfidf)
    str1 = ', '.join(featureCount)
    feature_df.loc[k, "TF Array"] = str1

print("Creating the feature vector for IDF Array...")
for k in range(0, len(inpTweets["Tweet"]), 1):
    featureCount = getIDFArray(inpTweets["Tweet"][k], tfidf)
    str1 = ', '.join(featureCount)
    feature_df.loc[k, "IDF Array"] = str1

print("Creating the feature vector for TF-IDF Array...")
for k in range(0, len(inpTweets["Tweet"]), 1):
    featureCount = getTFIDFArray(inpTweets["Tweet"][k], tfidf)
    str1 = ', '.join(featureCount)
    feature_df.loc[k, "TFIDF Array"] = str1



writer = pd.ExcelWriter('C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\All_Features_Tweets.xls')
feature_df.to_excel(writer,'Sheet1')
writer.save()
