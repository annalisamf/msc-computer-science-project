import pytest

from gender.gender_utils import *


def test_joined_df():
    test_path = '/Users/annalisa/PycharmProjects/MSc_Project/tests/data'
    test_users = ['00snook', '1_nutcracker']
    rows = [[1271142875833741312, 1271923930706608129], ['00snook', '1_nutcracker']]
    with pytest.raises(Exception) as e_info:
        assert joined_df(test_users, test_path) == rows
