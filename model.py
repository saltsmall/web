#!C:\Users\seungminKim\anaconda3\python.exe
# -*- coding: utf-8 -*-
import sys
import codecs

# stdout의 인코딩을 UTF-8로 강제 변환한다
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

import cgi
import cgitb
cgitb.enable()
print("content-type: text/html; charset=utf-8\n")

from tensorflow.keras.utils import to_categorical
import numpy as np
import util

code2vec = util.loadIcode(val="vec")
char2id = util.loadChar(val="id")
id2char = util.loadChar(val="dict")

char_size = 2276
#, custom_objects={'Functional':Model.Functional}
from tensorflow.keras.models import Model
from keras.models import load_model
model = load_model('20200806_v1.h5',compile=False)

def weightedPick(weight):
    t = np.cumsum(weight)
    s = np.sum(weight)

    return np.searchsorted(t, np.random.rand(1) * s)


def sentence_generation(model, length):
    ix = [np.random.randint(2276)]
    y_char = [id2char[ix[-1]]]
    #print(ix[-1], '번 글자', y_char[-1], '로 예측을 시작!')

    # X = np.zeros((1, 2, 2276))  # (1, 2, 2276) 크기의 X 생성. 즉, LSTM의 입력 시퀀스 생성
    #Y = np.zeros((1, 2, 14))

    X = np.zeros((1, length, 2276))
    Y = np.zeros((1, length, 14))

    Y[0] = np.array(code2vec['Q05A08'])


    for i in range(0, length):
        # X[0][0] = X[0][1]
        # X[0][1][ix[-1]] = 1  # X[0][i%2][예측한 글자의 인덱스] = 1, 즉, 예측 글자를 다음 입력 시퀀스에 추가
        X[0][i][ix[-1]] = 1

        # ix = weightedPick(model.predict([X, Y])[0][-1])
        ix = weightedPick(model.predict([X[:, :i+1, :],Y[:, :i+1, :]])[0][-1])

        y_char.append(id2char[ix[-1]])

    return ('').join(y_char)


print(sentence_generation(model, 10))
