"""
collect.py
"""

#Module Imports
from TwitterAPI import TwitterAPI
import pickle

#Twitter Access Credentials, Please generate credentials and fill PLACEHOLDERS...
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

def get_twitter():
    """
    This method returns a twitter object
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)
    
def search_tweets(twitter, keyword = 'Election', count = 5000):
    """
    This method searches for tweets and returns a list of tweets containing the keyword
    """
    result = twitter.request('search/tweets', {'q': keyword, 'count' : count} )    
    tweets = [t for t in result]
    return tweets
    
def find_top_screen_names(tweets):
    screen_names = []
    for tweet in tweets:
        if tweet['user']['statuses_count'] > 50000:
            #print(tweet['user']['statuses_count'])
            screen_names.append(tweet['user']['screen_name'])
    #print(screen_names)
    return screen_names[:5]  

def get_single_user_object(twitter, screen_name):
    return twitter.request( 'users/lookup', {'screen_name': screen_name} )
    
def get_all_user_objects(twitter, screen_names):
    user_list = []
    
    for eachUser in screen_names:
        user = get_single_user_object(twitter, eachUser)
        user_list.append(user)
    return user_list
          
def get_10_friends(twitter, screen_name):
    return twitter.request('friends/ids',{'screen_name':screen_name, 'count': 15})
    
def get_10_followers(twitter, screen_name):
    return twitter.request('followers/ids',{'screen_name':screen_name, 'count': 15})

def get_friends_and_follower_ids(twitter, user_list):
    
    dict_friends_and_followers = {}  
    
    for each_user in user_list:
        screen_name = each_user.json()[0]['screen_name'] 
        friends_10 = get_10_friends(twitter, screen_name)
        followers_10 = get_10_followers(twitter, screen_name)        
        
        friend_list = []
        follower_list = []
        friends_and_followers = [] # 0 - 9 friends and 10 - 19 followers
        
        for each in friends_10.json()['ids']:
            friend_list.append(each)
            friends_and_followers.append(each)            
            
        for each in followers_10.json()['ids']:
            follower_list.append(each)
            friends_and_followers.append(each)
        
        #For Each User, I have Friends and Followers
        dict_friends_and_followers[screen_name] = friends_and_followers             


    return dict_friends_and_followers
        
def get_user_object(twitter, user_id):
    return twitter.request( 'users/lookup', {'user_id': user_id} )        

def get_all_user_objects_friends_followers(twitter, user_ids):
    #Index 0 to 9 are Friend Objects and 10 to 19 are Follower Objects
    friend_and_follower_objects = {}
    
    for eachId in user_ids:
        user = get_user_object(twitter, eachId)
        friend_and_follower_objects[eachId] = user
        
    return friend_and_follower_objects        
       
def get_user_tweets(twitter, resource, parameter):
    return twitter.request(resource,parameter)
    
def get_tweets_for_all(twitter, name_or_id, flag):
    tweets_screen_name = {}
    tweets_ids = {}
    
    if flag:
        for eachName in name_or_id:
            tweets_reponse = get_user_tweets(twitter, 'statuses/user_timeline',{'screen_name':eachName} )
            #print(tweets_reponse.status_code) # CHECK-POINT
            if(tweets_reponse.status_code == 200):
                tweets = [t['text'] for t in tweets_reponse]
            else:
                tweets = []      
            
            tweets_screen_name[eachName] = tweets
        return tweets_screen_name
    else:
        for eachId in name_or_id:
            tweets_reponse_ID = get_user_tweets(twitter, 'statuses/user_timeline',{'user_id':eachId})
            #print(tweets_reponse_ID.status_code) # CHECK-POINT
            if(tweets_reponse_ID.status_code == 200):
                tweets = [t['text'] for t in tweets_reponse_ID]
            else:
                tweets = []
            
            tweets_ids[eachId] = tweets
        return tweets_ids
        
def main():
    #Get a Twitter API object
    twitter = get_twitter()
    
    #Search for tweets containing a Keyword. 
    #Default value for 'Keyword' is 'Election' and for 'count' is 5000
    tweets = search_tweets(twitter)
    
    print("Retrieved %d tweets." % len(tweets))
    
    #Now, we find the names of screen names of users who tweeted about the keyword.
    screen_names = find_top_screen_names(tweets)
    print("Retrieved screen names of top tweeters. Each top tweeter has greater than 50,000 tweets.")
    
    #Now, lets find out the user objects for these high-frequency tweeters
    top_users_objects = get_all_user_objects(twitter, screen_names)    
    top_users_objects = [user for user in top_users_objects]
    
    print('There are %d top user objects in list' % len(top_users_objects))
    
    #Now, we are going to find 10 friend IDs and 10 follower IDs. 
    dict_friends_and_followers_IDs = get_friends_and_follower_ids(twitter, top_users_objects) 
    print("Retrieved friend and follower IDs of top 5 tweeters.")
       
    dict_friends_and_followers_Objects = {}    
    
    for k,v in dict_friends_and_followers_IDs.items():
        friend_and_follower_objects = get_all_user_objects_friends_followers(twitter, v)
        dict_friends_and_followers_Objects[k] = friend_and_follower_objects
    print("Retrieved friend and follower User Objects")    
    
    
    all_friends_and_follower_IDS = []
    for k,v in dict_friends_and_followers_Objects.items():
        all_friends_and_follower_IDS.append(list(v.keys())) #-----CHECK-POINT
    
    #Now, I will fetch 20 tweets for each: 1. Top Users 2. Friends 3. Followers
    # A Dict, KEY is name and VALUE is a list of tweets
    #TRUE for screen_name and FALSE for IDs
    
    listOfDict_friend_and_follower_tweets = [] 
    
    for eachList in all_friends_and_follower_IDS:
        oneDict = get_tweets_for_all(twitter, eachList, False)
        listOfDict_friend_and_follower_tweets.append(oneDict) # List of 5 Dictionaries
    
    print("Retrieved Tweets of each of the friends and followers.")    
    

    
    dict_top_users_tweets = get_tweets_for_all(twitter, screen_names, True)
    print("Retrieved top user tweets.")    
    
    # DUMPING COLLECTED DATA TO PICKLE FILES
    output_screen_names = open('Collected_Data/screen_names.pkl','wb')
    pickle.dump(screen_names, output_screen_names)
    
    output_top_user_OBJECTS = open('Collected_Data/top_user_OBJECTS.pkl','wb')
    pickle.dump(top_users_objects, output_top_user_OBJECTS)
    
    output_dict_friends_and_followers_IDs = open('Collected_Data/friends_and_follower_IDs.pkl','wb')
    pickle.dump(dict_friends_and_followers_IDs, output_dict_friends_and_followers_IDs)
    
    output_friends_and_followers_OBJECTS = open('Collected_Data/friends_and_follower_OBJECTS.pkl','wb')
    pickle.dump(dict_friends_and_followers_Objects, output_friends_and_followers_OBJECTS)
    
    output_all_friends_and_followers_IDS = open('Collected_Data/all_friends_followers_IDs.pkl','wb')
    pickle.dump(all_friends_and_follower_IDS, output_all_friends_and_followers_IDS)
    
    output_listOfDict_friends_and_followers_tweets = open('Collected_Data/friends_and_followers_TWEETS.pkl', 'wb')
    pickle.dump(listOfDict_friend_and_follower_tweets, output_listOfDict_friends_and_followers_tweets)
    
    output_top_user_tweets = open('Collected_Data/top_user_TWEETS.pkl','wb')
    pickle.dump(dict_top_users_tweets, output_top_user_tweets)
    
    print("Successfully stored collected data in .pkl files.")
    
if __name__ == '__main__':
    main()
