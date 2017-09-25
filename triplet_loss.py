#coding=utf8

from keras.layers import Input,Embedding,LSTM,Dense,Lambda
from keras.models import Model
from keras import backend as K
from numpy import *
from keras.layers.merge import dot
from one_hot_prepare import *

word_size = 128
nb_features = 2000
nb_classes = 10
encode_size = 64
margin = 0.1

embedding_q = Embedding(nb_features,word_size,name='question_embedding')
embedding_a = Embedding(nb_features,word_size,name='answer_embedding')

lstm_encoder = LSTM(encode_size)

def encode_q(input):
    return lstm_encoder(embedding_q(input))

def encode_a(input):
    return lstm_encoder(embedding_a(input))

q_input = Input(shape=(None,))
a_right = Input(shape=(None,))
a_wrong = Input(shape=(None,))
q_encoded = encode_q(q_input)
a_right_encoded = encode_a(a_right)
a_wrong_encoded = encode_a(a_wrong)

q_encoded = Dense(encode_size)(q_encoded) #一般的做法是，直接讲问题和答案用同样的方法encode成向量后直接匹配，但我认为这是不合理的，我认为至少经过某个变换。

right_cos = dot([q_encoded,a_right_encoded], -1, normalize=True)
wrong_cos = dot([q_encoded,a_wrong_encoded], -1, normalize=True)

loss = Lambda(lambda x: K.relu(margin+x[0]-x[1]))([wrong_cos,right_cos])

model_train = Model(inputs=[q_input,a_right,a_wrong], outputs=loss)
model_q_encoder = Model(inputs=q_input, outputs=q_encoded)
model_a_encoder = Model(inputs=a_right, outputs=a_right_encoded)

model_train.compile(optimizer='adam', loss=lambda y_true,y_pred: y_pred)
model_q_encoder.compile(optimizer='adam', loss='mse')
model_a_encoder.compile(optimizer='adam', loss='mse')



# q = ['ksjffafs',]
# a1 = array(['jffafsdff '])
# a2 = array(['kcxjvklxsjafs'])
#
# dict = set(q.tolist() + a1.tolist() + a2.tolist())

txt_path = './'

label_text = {}
with open(txt_path + '1_point.txt', 'rb') as fp:
    lines_1 = fp.readlines()

label_text[0] = lines_1[:1000]

# oh_q = prepareOneHot(label_text)
#
# oh_a2 = prepareOneHot(label_text)

with open(txt_path + '5_point.txt', 'rb') as fp:
    lines_2 = fp.readlines()

label_text[1] = lines_2[:1000]

oh = prepareOneHot(label_text)

x,y,_,dicts = oh.get_pad_X_y(nb_features)

q = []
right_a = []
wrong_a = []
for i, single in enumerate(y):
    if single == 0:
        q.append(x[i])
        right_a.append(x[i])
    else:
        wrong_a.append(x[i])


y = ones(len(q))
model_train.fit([array(q),array(right_a),array(wrong_a)], y, epochs=3)
#其中q,a1,a2分别是问题、正确答案、错误答案的batch，
model_train.save('./loss.model')
model_train.save_weights('./loss.weights')

# model_train.load_weights('./loss.weights')

q = u'家里的大洗衣机基本就闲置了，这个更方便'
sequence = [dicts[char] for char in q]
X_q = pad_sequences([sequence], maxlen=nb_features)

a_r = u'不错，家里的大洗衣机基本就闲置了.'
sequence = [dicts[char] for char in a_r]
X_a_r = pad_sequences([sequence], maxlen=nb_features)


a_w = u'但不是很舒服，最大的问题是，穿的时间臭脚！'
sequence = [dicts[char] for char in a_w]
X_a_w = pad_sequences([sequence], maxlen=nb_features)




import numpy as np
def predict(model, q, a):

    user_vector = model.get_layer('question_embedding').get_weights()[0][q]
    item_matrix = model.get_layer('answer_embedding').get_weights()[0][a]

    # user_vector = model_q_encoder.predict([q])
    # item_matrix = model_a_encoder.predict([a])

    scores = (np.dot(user_vector,
                     item_matrix.T))

    return scores

print model_train.predict([array(X_q), array(X_a_r), array(X_a_w)])

results = predict(model_train, array(X_q), array(X_a_r))
print results

