import matplotlib.pyplot as plt

from location.location_utils import *
from mining_tweets.retrievedUsers import retrievedLibUsers

# this is the dataset with all the rt of the coalition with desc and cities
lib_df = import_rt_locations('data/lib_df.pkl')

# only take the first word before the comma in the location from the user
lib_df['location'] = lib_df['location'].str.split(',').str[0]

# calling the function to match the location from the user to the cities in the ONS dataset
matching_city(lib_df, cities_ons)

# merging the users and cities dataframes to get all in infos on the cities
lib_rt_location = pd.merge(lib_df, cities_ons, how='left', on='city')

# checking how the new column looks
# 'location' comes from the user setting.
# 'city' is the matched location with the ONS dataset
lib_rt_location[lib_rt_location['city'] != ""][['location', 'city', 'county']]

lib_rt_location.to_pickle('location/lib_rt_location.pkl',
                          protocol=4)  # need to save with protocol 4 for backwards compatibility

# END of matching location from descriptions


lib_rt_location.shape  # (14484, 41)
lib_df[lib_df['city'].notna()].shape  # 5369 this is the number of cities that have a match for
lib_df.city.value_counts().sum()  # 5369 this is the number of cities that have a match for
lib_df[lib_df['location'] == ""].shape  # 3850 had no location/blank
lib_df[lib_df['location'] != ""].shape  # 10634 had a location (but I matched only 5369)

# locations that did not find a match
not_matched = lib_df.loc[lib_df['location'] != ""]
not_matched = not_matched.loc[not_matched['city'].isna()]

# adding the location from the tweets, doing the analysis separately and then join with the other dataset

# get a list of user for which I found a match from location on profile
lib_users_with_loc = pd.read_pickle('location/lib_rt_location.pkl')
lib_users_with_loc = lib_users_with_loc[lib_users_with_loc['city'].notna()]['screen_name'].tolist()

# removing these users from the list of users with tweets, and get the final list of missing location users
missing_locations_users = [user for user in retrievedLibUsers if user not in lib_users_with_loc]
# the path of the tweets timeline
lib_path = 'TweetsLib'

# create a dataframe with users and location ONLY from the tweets

# this create a df with user and location from tweets only for those uses i was not able to match a location
lib_loc_tweets = df_loc_from_tweets(lib_path, missing_locations_users)
lib_loc_tweets.to_pickle('location/lib_loc_tweets.pkl', protocol=4)

# adding the name and description to make it equivalent to the other dataframe
lib_loc_tweets_desc = pd.merge(lib_loc_tweets, lib_df[['screen_name', 'name', 'description']], how='left',
                               on='screen_name')
# rearrange order of columns
lib_loc_tweets_desc = lib_loc_tweets_desc[['screen_name', 'name', 'description', 'location']]

# cleaning the created locations, taking string before comma and cleaning, libverting NaN to " for the next func to work
# libverting the NaN values to empty string for the matching city to work
lib_loc_tweets_desc['location'] = lib_loc_tweets_desc['location'].str.split(',').str[0].str.lower(). \
    str.replace('uk', '').str.replace('.', '').str.strip().fillna("")

# try to match the location with the ONS cities
matching_city(lib_loc_tweets_desc, cities_ons)

# with this technique i was able to match additional 691 location
# checking if i have any duplicated screen_name
user_desc = lib_rt_location[lib_rt_location['city'].notna()]['screen_name'].tolist()
user_tweets = lib_loc_tweets_desc[lib_loc_tweets_desc['city'].notna()]['screen_name'].tolist()
duplicated = set(user_desc) & set(user_tweets)  # Empty set, no duplicated

# keeping only rows where we found a match for the city
lib_loc_tweets_desc = lib_loc_tweets_desc[lib_loc_tweets_desc.screen_name.isin(user_tweets)]

# merging the users and city dataframe to get all in infos on the cities
lib_loc_tweets_desc_merged = pd.merge(lib_loc_tweets_desc, cities_ons, how='left', on='city')

# removing values from the first dataframe which i have in the selibd one as well
lib_rt_location_filtered = lib_rt_location[~lib_rt_location.screen_name.isin(user_tweets)]

# joining the two dataframes with the locations, the one from the desc and the one from tweets
lib_locations = pd.concat([lib_rt_location_filtered, lib_loc_tweets_desc_merged], ignore_index=True)
lib_locations.to_pickle('location/lib_locations.pkl', protocol=4)

# percentage of top 10 counties
lib_locations.county.value_counts(normalize=True).head(10)

# plotting the frequency of counties
fig, ax = plt.subplots()
lib_locations['county'].value_counts(normalize=True)[1:10].plot(ax=ax, kind='bar')
plt.show()
