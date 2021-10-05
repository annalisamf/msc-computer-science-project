import numpy as np
import pandas as pd


def import_rt_locations(rt_df):
    df = pd.read_pickle(rt_df).drop_duplicates(
        subset='screen_name', keep='first')
    return df[['screen_name', 'name', 'description', 'location']]


# this function matches the the cities in the ons dataset with the location of the user in a new column "city"
# I pass into this function the users with their location and the ONS dataset
def matching_city(users_location, cities_df):
    # create a column to store the found location
    users_location['city'] = np.nan
    city_found = []
    for u_city in users_location.location:
        for city in cities_df.city.values:
            if city.lower() == u_city.lower():  # if it finds a match on the word before the ,
                match = users_location.location.str.lower() == u_city.lower()
                users_location.loc[match, ['city']] = city
                city_found.append(city)
                print(f'Match found for: {city}')
    print(f'matched {len(city_found)} locations')


cities_ons = pd.read_csv('/Users/annalisa/PycharmProjects/MSc_Project/location/IPN_GB_2019.csv', encoding='latin-1')
cities_ons.rename(columns={"place18nm": "city", "ctyltnm": "county"}, inplace=True)

# removing cities which have a comma, they are usually duplicated entries (if city has a comma, such as London, little)
cities_ons = cities_ons[~cities_ons.city.str.contains(',')]

# checking for null in the county column
cities_ons.county.isna().sum()
# there are 413 null values for county, we drop them
cities_ons = cities_ons[cities_ons.county.notna()]

# we remove the duplicate cities
cities_ons = cities_ons.drop_duplicates(subset='city')


# # another city dataset
# postcodes = pd.read_csv('postcodes.csv')
# # check how many nulls
# postcodes.isnull().sum()
# # removing nulls cities
# postcodes = postcodes[postcodes.city.notna()]
#
# postcodes[postcodes.county == 'Liverpool'][['city', 'county']]
# c = postcodes[postcodes['city'].isnull()][['city', 'county']]
# postcodes["city"] = postcodes["city"].str.replace("\s:00", "")
def df_loc_from_tweets(path, users):
    li = []
    for user in users:
        try:
            li.append(
                pd.read_csv(path + "/%s_tweets.csv" % user, index_col=None, header=0)[['screen_name', 'location']].head(
                    1))
            # li.append(pd.read_csv(path + "/%s_tweets.csv" % user, index_col=None, header=0))
            # li.append(pd.read_csv(path + "/%s_tweets.csv" % user, index_col=None, header=0).groupby(
            #     'screen_name')[
            #               'location'].apply(lambda x: '. '.join(set(x.dropna()))).reset_index())
            print(len(li))
        except:
            print(user, " not found")
            continue
    return pd.concat(li, axis=0, ignore_index=True)
