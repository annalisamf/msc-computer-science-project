import matplotlib.pyplot as plt

from location.location_utils import *
# this is the dataset with all the con rt with desc and cities
from location.location_utils import df_loc_from_tweets
from mining_tweets.retrievedUsers import retrievedLabUsers

# this is the dataset with all the rt of that party with only 'screen_name', 'name', 'description', 'location' and no duplicates
lab_df = import_rt_locations('/Users/annalisa/PycharmProjects/MSc_Project/data/lab_df.pkl')

# only take the first word before the comma in the location from the user
lab_df['location'] = lab_df['location'].str.split(',').str[0]

# calling the function to match the location from the user to the cities in the ONS dataset
matching_city(lab_df, cities_ons)

# merging the users and cities dataframes to get all in infos on the cities
lab_rt_location = pd.merge(lab_df, cities_ons, how='left', on='city')

# checking how the new column looks
# 'location' comes from the user setting.
# 'city' is the matched location with the ONS dataset
lab_rt_location[lab_rt_location['city'] != ""][['location', 'city', 'county']]

lab_rt_location.to_pickle('location/lab_rt_location.pkl',
                          protocol=4)  # need to save with protocol 4 for backwards compatibility
lab_rt_location = pd.read_pickle('location/lab_rt_location.pkl')
# END of matching location from descriptions

lab_rt_location.shape  # (41454, 41)
lab_df[lab_df['city'].notna()].shape  # 15099 this are the number of cities that have a match
lab_df.city.value_counts().sum()  # 15099 this are the number of cities that have a match
lab_df[lab_df['location'] == ""].shape  # 12125 had no location/blank
lab_df[lab_df['location'] != ""].shape  # 29329 had a location (but I matched only 15099)

# locations that did not find a match
not_matched = lab_df.loc[lab_df['location'] != ""]
not_matched = not_matched.loc[not_matched['city'].isna()]

# adding the location from the tweets, doing the analysis separately and then join witht the other dataset

# get a list of user for which I found a match from location on profile
lab_users_with_loc = pd.read_pickle('location/lab_rt_location.pkl')
lab_users_with_loc = lab_users_with_loc[lab_users_with_loc['city'].notna()]['screen_name'].tolist()

# removing these users from the list of users with tweets, and get the final list of missing location users
missing_locations_users = [user for user in retrievedLabUsers if user not in lab_users_with_loc]
# the path of the tweets timeline
lab_path = 'TweetsLab'

# create a dataframe with users and location ONLY from the tweets

# this create a df with user and location from tweets only for those uses I was not able to match a location
lab_loc_tweets = df_loc_from_tweets(lab_path, missing_locations_users)
lab_loc_tweets.to_pickle('location/lab_loc_tweets.pkl', protocol=4)

# adding the name and description to make it equivalent to the other dataframe
lab_loc_tweets_desc = pd.merge(lab_loc_tweets, lab_df[['screen_name', 'name', 'description']], how='left',
                               on='screen_name')
# rearrange order of columns
lab_loc_tweets_desc = lab_loc_tweets_desc[['screen_name', 'name', 'description', 'location']]

# cleaning the created locations, taking string before comma and cleaning, labverting NaN to " for the next func to work
# converting the NaN values to empty string for the matching city to work
lab_loc_tweets_desc['location'] = lab_loc_tweets_desc['location'].str.split(',').str[0].str.lower(). \
    str.replace('uk', '').str.replace('.', '').str.strip().fillna("")

# try to match the location with the ONS cities
matching_city(lab_loc_tweets_desc, cities_ons)

# with this technique i was able to match additional 691 location
# checking if i have any duplicated screen_name
user_desc = lab_rt_location[lab_rt_location['city'].notna()]['screen_name'].tolist()
user_tweets = lab_loc_tweets_desc[lab_loc_tweets_desc['city'].notna()]['screen_name'].tolist()
duplicated = set(user_desc) & set(user_tweets)  # Empty set, no duplicated

# keeping only rows where we found a match for the city
lab_loc_tweets_desc = lab_loc_tweets_desc[lab_loc_tweets_desc.screen_name.isin(user_tweets)]

# merging the users and city dataframe to get all in infos on the cities
lab_loc_tweets_desc_merged = pd.merge(lab_loc_tweets_desc, cities_ons, how='left', on='city')

# removing values from the first dataframe which i have in the selabd one as well
lab_rt_location_filtered = lab_rt_location[~lab_rt_location.screen_name.isin(user_tweets)]

# joining the two dataframes with the locations, the one from the desc and the one from tweets
lab_locations = pd.concat([lab_rt_location_filtered, lab_loc_tweets_desc_merged], ignore_index=True)
lab_locations.to_pickle('location/lab_locations.pkl', protocol=4)

# from pickled file
lab_locations = pd.read_pickle('location/lab_locations.pkl')

# percentage of top 10 counties
lab_locations.county.value_counts(normalize=True).head(10)

# plotting the frequency of the counties
fig, ax = plt.subplots()
lab_locations['county'].value_counts(normalize=True)[1:10].plot(ax=ax, kind='bar')
plt.show()
