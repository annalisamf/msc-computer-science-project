import pytest

from location.location_utils import df_loc_from_tweets


def test_df_loc_from_tweets():
    test_path = '/Users/annalisa/PycharmProjects/MSc_Project/tests/data'
    test_users = ['1_pye', 'pinco', '222Harrison']
    with pytest.raises(Exception) as e_info:
        assert df_loc_from_tweets(test_path, test_users) == [['1_pye', '222Harrison'], ['Orkney', 'Cambridge, England']]
