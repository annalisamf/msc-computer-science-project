from topic_model.tm_utils import *


def test_lemmatization():
    tweet = [
        'RT @TravelLeisure: This Ghost Town Was Swallowed Up by Nature and Now It\'s an Eerily Beautiful Hiking Destination https://t.co/2HYSpkwDcd h…',
        '"@MMass1ve @Gotztheironhand @Cernovich Also......did I use the either of the words ""like"" or ""literally"" anywhere in… https://t.co/bvKvmQwXuT"',
        '"RT @DFBHarvard: “Freedom is never more than one generation away from extinction."']
    assert lemmatization(tweet) == [['ghost', 'town', 'swallowed', 'nature', 'eerily', 'beautiful', 'hiking'],
                                    ['use', 'words', 'like', 'literally'],
                                    ['freedom', 'generation', 'away', 'extinction']]
