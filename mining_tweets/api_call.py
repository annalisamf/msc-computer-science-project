import csv
import json
import time
import tweepy

from mining_tweets import credentials_secret

auth = tweepy.AppAuthHandler(credentials_secret.API_key, credentials_secret.API_secret_key)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


def save_json(file_name, file_content):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(file_content, f, ensure_ascii=False, indent=4)


# Helper function to handle twitter API rate limit
def limit_handled(cursor, list_name):
    while True:
        try:
            yield cursor.next()
        except tweepy.RateLimitError:  # Catch Twitter API rate limit exception and wait for 15 minutes
            print("\ndata points in list = {}".format(len(list_name)))
            print('Hit Twitter API rate limit.')
            for i in range(3, 0, -1):
                print("Wait for {} mins.".format(i * 5))
                time.sleep(5 * 60)
        except tweepy.error.TweepError:  # Catch any other Twitter API exceptions
            print('\nCaught TweepError exception')


# Function to get all tweets of a specified user
# Allows access to the most recent 3200 tweets
# Source: https://gist.github.com/yanofsky/5436496
def get_tweets(folder, screen_name):
    all_tweets = []  # initialize a list to hold all the tweets
    latest_tweets = api.user_timeline(screen_name=screen_name, count=200)  # (200 is the maximum allowed)
    all_tweets.extend(latest_tweets)  # save most recent tweets
    oldest = all_tweets[-1].id - 1  # save the id of the oldest tweet less one to avoid duplication
    while len(latest_tweets) > 0:  # get tweets until there are no left
        print("getting tweets before %s" % (oldest))
        # all subsequent requests will use the max_id param (preventing starting over and duplicates)
        latest_tweets = api.user_timeline(screen_name=screen_name, count=200, max_id=oldest)
        all_tweets.extend(latest_tweets)  # save most recent tweets
        oldest = all_tweets[-1].id - 1  # update the id of the oldest tweet less one
        print("...%s tweets downloaded so far for username %s" % (len(all_tweets), screen_name))
        ### END OF WHILE LOOP ###  # transform the tweepy tweets into a 2D array that will populate the csv
    outtweets = [
        [tweet.id_str, tweet.user.screen_name, tweet.created_at, tweet.user.location, tweet.text, tweet.favorite_count,
         tweet.in_reply_to_screen_name, tweet.retweeted] for tweet in all_tweets]
    with open('%s/%s_tweets.csv' % (folder, screen_name), 'w') as f:  # write the csv
        writer = csv.writer(f)
        writer.writerow(["id", "screen_name", "created_at", "location", "text", "likes", "in reply to", "retweeted"])
        writer.writerows(outtweets)
    pass


def retrieveTweets(folder, listOfUsers, retrievedUsersFile):
    retrieved = []
    missed = 0
    for username in listOfUsers:
        try:
            get_tweets(folder, username)
            retrieved.append(username)
            print("Retrieved users so far :", len(retrieved))
            print("Left users to retrieve :", len(listOfUsers) - (len(retrieved) + missed))
            with open("%s.txt" % retrievedUsersFile, "w") as f:
                f.write(repr(retrieved))
        except (tweepy.TweepError, IndexError) as e:
            print("impossible to download tweets for ", username)
            print("Error: ", e)
            missed += 1
            print("missed users so far: ", missed)
            continue
