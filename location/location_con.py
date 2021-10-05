import matplotlib.pyplot as plt

from location.location_utils import *
from mining_tweets.retrievedUsers import retrievedConUsers

con_df = import_rt_locations('/Users/annalisa/PycharmProjects/MSc_Project/data/con_df.pkl')

# only take the first word before the comma in the location from the user
con_df['location'] = con_df['location'].str.split(',').str[0]

# calling the function to match the location from the user to the cities in the ONS dataset
matching_city(con_df, cities_ons)

# merging the users and cities dataframes to get all in infos on the cities
con_rt_location = pd.merge(con_df, cities_ons, how='left', on='city')

# checking how the new column looks
# 'location' comes from the user setting.
# 'city' is the matched location with the ONS dataset
con_rt_location[con_rt_location['city'] != ""][['location', 'city', 'county']]

con_rt_location.to_pickle('location/con_rt_location.pkl',
                          protocol=4)  # need to save with protocol 4 for backwards compatibility

# END of matching location from descriptions

con_rt_location.shape  # (30557, 41)
con_df[con_df['city'].notna()].shape  # 9570 this is the number of cities that have a match
con_df.city.value_counts().sum()  # 9570 this is the number of cities that have a match
con_df[con_df['location'] != ""].shape  # 20961 had a location (but I matched only 9570-cities)

# locations that did not find a match
not_matched = con_df.loc[con_df['location'] != ""]
not_matched = not_matched.loc[not_matched['city'].isna()]

# adding the location from the tweets, doing the analysis separately and then join with the other dataset

# get a list of user for which I found a match from location on profile
con_users_with_loc = pd.read_pickle('location/con_rt_location.pkl')
con_users_with_loc = con_users_with_loc[con_users_with_loc['city'].notna()]['screen_name'].tolist()

# removing these users from the list of users with tweets, and get the final list of missing location users
missing_locations_users = [user for user in retrievedConUsers if user not in con_users_with_loc]
# the path of the tweets timeline
con_path = 'TweetsCon'

# create a dataframe with users and location ONLY from the tweets

# this create a df with user and location from tweets only for those users I was not able to match a location
con_loc_tweets = df_loc_from_tweets(con_path, missing_locations_users)
con_loc_tweets.to_pickle('location/con_loc_tweets.pkl', protocol=4)

# adding the name and description to make it equivalent to the other dataframe
con_loc_tweets_desc = pd.merge(con_loc_tweets, con_df[['screen_name', 'name', 'description']], how='left',
                               on='screen_name')
# rearrange order of columns
con_loc_tweets_desc = con_loc_tweets_desc[['screen_name', 'name', 'description', 'location']]

# cleaning the created locations, taking string before comma and cleaning, converting NaN to " for the next func to work
# converting the NaN values to empty string for the matching city to work
con_loc_tweets_desc['location'] = con_loc_tweets_desc['location'].str.split(',').str[0].str.lower(). \
    str.replace('uk', '').str.replace('.', '').str.strip().fillna("")

# try to match the location with the ONS cities
matching_city(con_loc_tweets_desc, cities_ons)

# with this technique I was able to match additional 691 location
# screen_names from the location in description
user_desc = con_rt_location[con_rt_location['city'].notna()]['screen_name'].tolist()
# screen_names from the location in tweets
user_tweets = con_loc_tweets_desc[con_loc_tweets_desc['city'].notna()]['screen_name'].tolist()
# checking if I have any duplicated screen_name
duplicated = set(user_desc) & set(user_tweets)  # Empty set, no duplicated

# keeping only rows where we found a match for the city
con_loc_tweets_desc = con_loc_tweets_desc[con_loc_tweets_desc.screen_name.isin(user_tweets)]

# merging the users and city dataframe to get all in infos on the cities
con_loc_tweets_desc_merged = pd.merge(con_loc_tweets_desc, cities_ons, how='left', on='city')

# removing values from the first dataframe which I have in the second one as well
con_rt_location_filtered = con_rt_location[~con_rt_location.screen_name.isin(user_tweets)]

# joining the two dataframes with the locations, the one from the desc and the one from tweets
con_locations = pd.concat([con_rt_location_filtered, con_loc_tweets_desc_merged], ignore_index=True)
con_locations.to_pickle('location/con_locations.pkl', protocol=4)

# read file if already saved
con_locations = pd.read_pickle('/Users/annalisa/PycharmProjects/MSc_Project/location/con_locations.pkl')

# percentage of top 10 counties
con_locations_top10 = con_locations.county.value_counts(normalize=True).head(10)

# plotting the frequency of counties
fig, ax = plt.subplots()
con_locations['county'].value_counts(normalize=True)[1:10].plot(ax=ax, kind='bar')
plt.show()
