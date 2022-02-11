"""
This is a utility file that allows us to perform simple tasks such as I/O disk writing etc
we use pickle object. This is like just preserving a pickle object in your disk
"""

import pickle


def save_file(path, obj):
    """
    Saves file object via pickle

    :param path: str
    :param obj: object
    """

    try:
        with open(path, 'wb') as fp:
            pickle.dump(obj, fp)
    except Exception as err:
        print('pickle error: ', str(err))


def load_file(path):
    """
    Loads the object via pickle

    :param path: str
    :return: object
    """

    try:
        with open(path, 'rb') as fp:
            file = pickle.load(fp)
            return file
    except Exception as err:
        print('load error: ', str(err))


