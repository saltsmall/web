import pandas as pd
import numpy as np
import math

def loadIcode(val="id"):
    if val =="dict":
        icode = np.load("../data/dict_industryCode.npy", allow_pickle=True)
    elif val == "id":
        icode = np.load("../data/id_industryCode.npy", allow_pickle=True).item()
    elif val =="vec":
        icode = np.load("../data/code2vec.npy", allow_pickle=True).item()
    return icode

def loadChar(val="id"):
    if val == "dict":
        char = np.load("../data/dict_char.npy", allow_pickle=True)
    elif val == "id":
        char = np.load("../data/id_char.npy", allow_pickle=True).item()
    elif val == "vec":
        char = np.load("../data/char2vec.npy", allow_pickle=True).item()
    return char

