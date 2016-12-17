"""
classify.py
"""
import pickle
import re
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter, defaultdict, deque
from scipy.sparse import csr_matrix

import sys
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report

def read_friend_and_follower_TWEETS():
    pkl_friend_and_follower_TWEETS = open('Collected_Data/friends_and_followers_TWEETS.pkl','rb')
    return pickle.load(pkl_friend_and_follower_TWEETS)
    
def read_top_user_TWEETS():
    pkl_top_user_TWEETS = open('Collected_Data/top_user_TWEETS.pkl','rb')
    return pickle.load(pkl_top_user_TWEETS)
    
def read_cluster_data():
    pkl_cluster_data = open('Cluster_Data/dict_clusters.pkl','rb')
    return pickle.load(pkl_cluster_data)
    
def read_friends_and_followers_IDs():
    pkl_friend_and_follower_IDs = open('Collected_Data/friends_and_follower_IDs.pkl','rb')
    return pickle.load(pkl_friend_and_follower_IDs)
    
def read_training_data():
    tweets = pd.read_csv('Training_Data/training_data.csv')
    #print(np.array(tweets['polarity']))
    #print(len(tweets))
    return tweets
    
def top_user_names_to_tweets(dict_user_to_tweets, u_tweets):
    for k,v in u_tweets.items():
        dict_user_to_tweets[k] = v
    return dict_user_to_tweets
    
def friends_and_follower_ids_to_tweets(dict_user_to_tweets, ff_tweets):
    for eachDict in ff_tweets:
        for k,v in eachDict.items():
            dict_user_to_tweets[k] = v
    return dict_user_to_tweets
    
def community_members(dict_user_to_tweets, community):
    cluster_keys = list(community.keys())
    
    for eachKey in cluster_keys:
        for eachUser in community[eachKey]:
            if eachUser in list(dict_user_to_tweets.keys()):
                print(len(dict_user_to_tweets[eachUser]))
            else:
                print("False")
                
def all_tweets_in_list(dict_user_to_tweets):
    all_tweets = []
    
    for k, v in dict_user_to_tweets.items():
        for eachTweet in v:
            all_tweets.append(eachTweet)
            
    return all_tweets   
    

def tokenize_string(my_string):
    return re.findall('[\w]+', my_string.lower())
    
  
def build_vocab_training_data(tweets_text):
    vocab = []
    
    for eachTweet in tweets_text:
        if type(eachTweet) != float:
        
            for eachWord in tokenize_string(eachTweet):
                if eachWord not in vocab:
                    vocab.append(eachWord)
    vocab = sorted(vocab)
    
    dictionary = {}
    index = 0
    for eachWord in vocab:
        dictionary[eachWord] = index
        index += 1
        
    return dictionary
   
    
    
def create_sparse_matrix_for_user(dict_user_to_tweets, friend_and_follower_IDs):
    # We have a dict mapping user to its friends and followers. We have a dictionary mapping each user to their tweets.
    List_of_sparse_matrices = [] # For each user we have a sparse matrix, for each of the friends and followers
    
    for k, v in dict_user_to_tweets.items():
        if type(k) != int:
            print(k)

    print()
    for k,v in friend_and_follower_IDs.items():
        print(k)
        
#----------------------CLASSIFICATION SECTION-------------------------------------
        
def create_a_model():
    return LogisticRegression()
    
def get_data_to_train_model():
    tweets = read_training_data()
    tweets_labels = tweets['polarity']
    tweets_text = tweets['text']
    
    vectorizer = CountVectorizer(min_df=0, token_pattern=r"\b\w+\b")
    X = vectorizer.fit_transform(tweets_text.values.astype('U'))
    print('Vectorized %d tweets. Found %d terms.' % (X.shape[0], X.shape[1]))
    
    return (X, tweets_labels)

def train_the_model(model, X, y): # X is the csr-matrix of training data. y is true labels.    
    model.fit(X,y)
    return(model)

def accuracy(truth, predicted):
    return len(np.where(truth==predicted)[0]) / len(truth)
    
def accuracy_on_training_set(model, X, y):
    predicted = model.predict(X)
    print('accuracy on training data=%.3f' % accuracy(y, predicted))
    
#---------------------------------------------------------------------------------------------------------------------
    
def tokenize(doc, keep_internal_punct=False):
    doc = doc.lower()
    
    if(keep_internal_punct):
        return np.array(re.findall(r"\S+\w", doc))
    else:
        return np.array(re.sub(r"\W", " ", doc).split())
        
def token_features(tokens, feats):
    for token in tokens:
        feats['token='+token] = 0
        for each in tokens:
            if each == token:
                feats['token='+token] += 1
                
def featurize(tokens, feature_fns):
    feats = defaultdict(lambda: 0)
    for fns in feature_fns:
        fns(tokens, feats)
    return sorted(feats.items(), key = lambda x:x[0])

def vectorize(tokens_list, feature_fns,dictionary, min_freq):
    keepCount = Counter()
    final = []
    anotherList = []
    for token in tokens_list:
        docDic = featurize(np.array(token),feature_fns)
        final += docDic
    for each in final:
        if each[1] != 0:
            freq = each[0]
            keepCount[freq] += 1
    for k, v in keepCount.items():
        if v >= min_freq:
            anotherList.append(k)
    anotherList.sort()
    #vocabulary = defaultdict(int)
    #for index,val in enumerate(anotherList):
        #vocabulary[val] = index
    rowData = []
    colIdx = []
    data = []
    for index,token in enumerate(tokens_list):
        docDic = featurize(token,feature_fns)
        for word in docDic:
            if word[0] in dictionary.keys() :
                rowData.append(index)
                colIdx.append(dictionary[word[0]])
                data.append(word[1])
    return (csr_matrix((np.array(data), (np.array(rowData), np.array(colIdx))), shape = (len(tokens_list),len(dictionary)), dtype=np.int64),dictionary)
    
def predict_on_test_data(model, X):
    predicted = model.predict(X)
    print(len(predicted))

    
#-------------------------------------------------------
def train(text_train):
    vectorizer = TfidfVectorizer(min_df=5,max_df = 0.8, sublinear_tf=True,use_idf=True)
    train_vectors = vectorizer.fit_transform(text_train.values.astype('U'))
    return (vectorizer, train_vectors)                     
                             
def build_model(train_vectors, y):
    classifier_linear = svm.SVC(kernel='linear')
    classifier_linear.fit(train_vectors, y)
    return classifier_linear   
    
def predict(vectorizer, classifier_linear,text_test):
    prediction_linear = np.array([])  
    
    if len(text_test) != 0:
        test_vectors = vectorizer.transform(text_test)
        prediction_linear = classifier_linear.predict(test_vectors)
    
    return prediction_linear
    
def calculate(prediction_linear):
    pos = 0
    neg = 0
    for each in prediction_linear:
        if each == 4:
            pos += 1
        elif each == 0:
            neg += 1
    return (pos, neg)
    
def unnamed(communities, user_to_tweets, vectorizer, classifier_linear):
    user_sentiment = {}
    user_sentiment_predictions ={}    
    cluster_keys = communities.keys()
    
    for eachKey in cluster_keys:
        
        for eachMember in communities[eachKey]:
            tweet_list = []
            for eachTweet in user_to_tweets[eachMember]:
                tweet_list.append(eachTweet)
            predictions = predict(vectorizer, classifier_linear, tweet_list)
            pos, neg = calculate(predictions)
            user_sentiment[eachMember] = [pos,neg]
            user_sentiment_predictions[eachMember] = predictions
    return (user_sentiment,user_sentiment_predictions)

def cluster_positivity(communities, user_sentiment):
    cluster_keys = communities.keys()
    cluster_positivity = {}
    for eachKey in cluster_keys:
        pos = 0
        neg = 0
        for eachMember in communities[eachKey]:
            if eachMember in user_sentiment.keys():
                pos += user_sentiment[eachMember][0]
                neg += user_sentiment[eachMember][1]
        positivity = pos - neg
        cluster_positivity[eachKey] = positivity
    return cluster_positivity
    
def main():
    # Now, we have some people who talked about the topic.
    # They are clustered into communities.
    # Now, we analyze their tweets.
    
    #Read pickle files: 1. friends_and_followers_TWEETS.pkl 2. top_user_TWEETS.pkl 3. dict_clusters.pkl 4. friends_and_follower_IDs.pkl
    
    friend_and_follower_TWEETS = read_friend_and_follower_TWEETS()
    print("Read friend and follower tweets.")
    
    top_user_TWEETS = read_top_user_TWEETS()
    print("Read top user tweets.")
    
    communities = read_cluster_data()
    print("Read cluster data")    
    
    friend_and_follower_IDs = read_friends_and_followers_IDs()
    print("Read friend and follower IDs.")

    user_to_tweets = {}

    user_to_tweets = top_user_names_to_tweets(user_to_tweets, top_user_TWEETS) 
    print("User tweets added to dictionary.")
    
    user_to_tweets = friends_and_follower_ids_to_tweets(user_to_tweets, friend_and_follower_TWEETS)
    print("Friend and follower tweets added to dictionary.")           
    
    tweets = read_training_data()
    
    dictionary = build_vocab_training_data(tweets['text']) #Passing a series object.
    print("Dictionary built from training data. It has %d words" % len(dictionary.keys()))        
    
    all_tweets = all_tweets_in_list(user_to_tweets)    
    print("All tweets from each user is in a big list now. Length of list is %d" % len(all_tweets))
    
    X, y =get_data_to_train_model()    
    print("Received Data (X, y) to train model.")
 
    #-------------------------
    vectorizer, train_vectors = train(tweets['text'])
    print("Data ready to train model...")
    
    classifier_linear = build_model(train_vectors, y)
    print("Model built and trained...")
    
    prediction_linear = predict(vectorizer, classifier_linear,all_tweets)
    #print(prediction_linear)
    print("Predictions received...")
    
    pos, neg = calculate(prediction_linear)
    print("Overall, %d positive tweets and %d negative tweets. Now, lets drill-down for each member of each community." % (pos, neg))
    print()
    
    user_sentiment, user_sentiment_predictions = unnamed(communities, user_to_tweets, vectorizer, classifier_linear) 
    #print(len(user_sentiment.keys()))

    positive_instance = None
    negative_instance = None    
    
    for k,v in user_sentiment_predictions.items():
        for eachPrediction in v:
            if eachPrediction == 4:
                index = list(v).index(eachPrediction)
                for k1,v1 in user_to_tweets.items():
                    if k == k1:
                        positive_instance = v1[index]
            elif eachPrediction == 0:
                index = list(v).index(eachPrediction)
                for k1,v1 in user_to_tweets.items():
                    if k == k1:
                        negative_instance = v1[index]
                        
    print('Positive : %s'% positive_instance)
    print()
    print('Negative : %s'% negative_instance)
    print()
    
    clust_positivity = cluster_positivity(communities, user_sentiment) 
    for k,v in clust_positivity.items():
        print('cluster %d : positivity: %d' % (k,v))
        
    #--------Let us pickle this information to summarize (user, messages, # of communitites, avg # of users per community, number of instances per class, example of each class.)
    dump_list = []
    dump_list.append(user_to_tweets)
    dump_list.append(all_tweets)
    dump_list.append(communities)
    dump_list.append(pos)
    dump_list.append(neg)
    dump_list.append(positive_instance)
    dump_list.append(negative_instance)

    output_dump_list = open('Classify_Data/summarize.pkl','wb')
    pickle.dump(dump_list, output_dump_list) 
    print("Data stored in pickle file. Proceed to summarize...")
    
    
if __name__ == '__main__':
    main()