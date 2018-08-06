import emoji
import pandas as pd
from textblob import TextBlob
from textblob import Word
from nltk.corpus import stopwords
# import regex
import re
from nltk import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


def space_emojis(tweet):
    output_string = ""
    for c in tweet:
        if( c in emoji.UNICODE_EMOJI):
            output_string = output_string + " " + c
        else:
            output_string = output_string + c

    return(output_string)


# start replaceTwoOrMore
#this is normalization
def replaceTwoOrMore(word):
    # look for 2 or more repetitions of character
    pattern = re.sub(r"(.)\1{1,}", r"\1\1", word)
    return pattern

# start process_tweet
def processTweet(tweet, lowercase = True):
    # process the tweets
    special_chars = []
    # Convert to lower case
    tweet = tweet.lower()
    # Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
    # Convert @username to username
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet) #leave usernames in?
    # Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    # Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) #consider the importance of this
    #remove numbers
    tweet = re.sub("\s?[0-9]+\.?[0-9]*", ' ', tweet)
    #fix the ampersands to 'and'
    tweet = re.sub('(&amp;)', 'and', tweet)
    # fix the 'u' to 'you'
    tweet = re.sub('( u )', 'you', tweet)
    # remove the @
    tweet = re.sub('(@)', '', tweet)
    # fix the 'r' to 'are'
    tweet = re.sub('( r )', 'are', tweet)
    #remove special characters
    tweet = re.sub('(,)|(:)|(%)|(=)|(;)|($)|(^)|(\")|(\*)|(\()|(\))|(\{)|(})|(\[)|(])|(\|)|(/)|(>)|(<)|(-)|(!)|(\?)|(\.)|(_)|(\')', "", tweet) #consider the different characters and their significance

    # trim
    tweet = tweet.strip('\'"')
    return tweet


#tokenization and stemming
def tokenize_stem_tweets(tweet):
    words = word_tokenize(tweet)
    token_tweet = ""
    for word in words:
        token_tweet = token_tweet + lm.lemmatize(word) + " "
        #or
        #token_tweet = token_tweet + sb.stem(word) + " "
    return token_tweet

def remove_stopwords(tweet):
    words = word_tokenize(tweet, language="english")
    rebuilt_tweet = ""
    for word in words:
        # replace two or more with two occurrences
        word = replaceTwoOrMore(word)
        # strip punctuation
        # word = word.strip('\'"?,.')
        # ignore if it is a stopWord
        if (word in stopWords):
            rebuilt_tweet = rebuilt_tweet + ""
        else:
            rebuilt_tweet = rebuilt_tweet + word + " "
    return rebuilt_tweet

# end of functions

input_files = ['C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\Sentiment_Training_Data_Larger.xlsx', 'C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\Sentiment_Test_Data_Larger.xlsx',
               'C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\Joy_Training_Data.xlsx', 'C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\Joy_Test_Data.xlsx',
               'C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\Anger_Training_Data.xlsx', 'C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\Anger_Test_Data.xlsx',
               'C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\Support_Training_Data.xlsx', 'C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\Support_Test_Data.xlsx']

output_files = ['C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\Sentiment_Training_Data_Larger_Processed.xlsx', 'C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\Sentiment_Test_Data_Larger_Processed.xlsx',
               'C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\Joy_Training_Data_Processed.xlsx', 'C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\Joy_Test_Data_Processed.xlsx',
               'C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\Anger_Training_Data_Processed.xlsx', 'C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\Anger_Test_Data_Processed.xlsx',
               'C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\Support_Training_Data_Processed.xlsx', 'C:\\Users\FitzL\Desktop\Work\Sentiment Analysis\Data\Support_Test_Data_Processed.xlsx']

for j in range(0, len(input_files)):
    #import the spreadsheets of tweets and convert excel to pd df
    xl = pd.read_excel(input_files[j], encoding = "latin-1")
    inpTweets = pd.DataFrame(xl)
    #import the stopwords
    stopWords = stopwords.words("english")

    sb = SnowballStemmer("english")
    lm = WordNetLemmatizer()

    #process the tweets
    featureList = []
    tweets = []
    outTweets = pd.DataFrame([])
    print("Cleaning...")
    for k in range(0,len(inpTweets["Tweet"]),1):
        processedTweet = processTweet(inpTweets["Tweet"][k], lowercase=False)
        inpTweets.set_value(k, "Tweet", processedTweet)

    print("Stemming...")
    for k in range(0,len(inpTweets["Tweet"]),1):
        stemmedTweet = tokenize_stem_tweets(inpTweets["Tweet"][k])
        next_tweet = remove_stopwords(stemmedTweet)
        inpTweets.set_value(k, "Tweet", next_tweet)

    print("Spacing emojis...")
    for k in range(0,len(inpTweets["Tweet"]),1):
        emoji_tweet = space_emojis(inpTweets["Tweet"][k])
        inpTweets.set_value(k, "Tweet", emoji_tweet)

    """#not sure how effective this really is, but it's here now
    print("Spelling...")
    county = 0
    for k in range(0,len(inpTweets[0]),1):
        spelling = TextBlob(inpTweets[0][k])
        fixed_tweet = spelling.correct()
        inpTweets.set_value(k, 0, fixed_tweet)
        if(county % 1 == 0):
            print(county)
        county += 1
    """

    #for whatever reason, I needed this line in order to correctly output the tweets back to an excel spreadsheet
    modified_tweets = inpTweets

    #identify the path of a new excel spreadsheet
    writer = pd.ExcelWriter(output_files[j])
    modified_tweets.to_excel(writer,'Sheet1')
    writer.save()
