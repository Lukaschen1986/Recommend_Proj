# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
os.getcwd()
os.chdir("D:/my_project/Python_Project/iTravel/itravel_recommend")
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from scipy.stats import itemfreq
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, ZeroPadding2D, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, Multiply, Embedding, Add, Dot
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.advanced_activations import PReLU
from keras.layers.noise import GaussianNoise
from keras import backend as K
K.image_data_format()
K.set_image_data_format('channels_first')
from keras import initializers
from keras.regularizers import l1, l2
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
from keras.models import load_model, model_from_json, model_from_yaml
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, History
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import cv2

# load user-item
_filename = "./txt/ml-100k/u1.base"
df_user_item = pd.read_csv(_filename, sep="\t", usecols=[0, 1, 2], names=["user_id", "movie_id", "rating"])
matrix = pd.pivot_table(df_user_item, index="user_id", columns="movie_id", values="rating", aggfunc=np.mean, fill_value=0, margins=False)
#matrix = matrix.values

user_vec = []
for user_id in df_user_item["user_id"].values:
    data = matrix.iloc[matrix.index == user_id].values[0].tolist()
    user_vec.append(data)
user_vec = np.array(user_vec)
user_vec.shape

movie_vec = []
for movie_id in df_user_item["movie_id"].values:
    data = matrix.iloc[:, matrix.columns == movie_id].values.T[0].tolist()
    movie_vec.append(data)
movie_vec = np.array(movie_vec)

#user_num = len(set(_df_user_item.user_id))
#movie_num = len(set(_df_user_item.movie_id))
#def create_matrix(_df_user_item, user_num, movie_num):
#    matrix = np.zeros((user_num, movie_num), dtype=np.float32)
#    for line in range(len(_df_user_item)):
#        # line = 1
#        data = _df_user_item.iloc[line,:]
#        user_id = int(data[0])
#        movie_id = int(data[1])
#        rating = int(data[2])
#        matrix[user_id - 1][movie_id - 1] = int(rating>=1)

rating = df_user_item.rating.values

# load user
_filename = "./txt/ml-100k/u.user"
_user_info = pd.read_csv(_filename, sep="|", names=["user_id", "age", "gender", "occupation", "zipcode"])
_user_info = pd.get_dummies(_user_info, columns=["gender", "occupation", "zipcode"], drop_first=False)

_df_user = df_user_item[["user_id"]].merge(_user_info, on="user_id", how="left")
user_add = _df_user.values[:,1:]

# load item
_filename = "./txt/ml-100k/u.item"
_item_info = pd.read_csv(_filename, sep="|",\
                         usecols=[0,1,2,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],\
                         names=["movie_id","movie_title","release_date","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23"])
_item_info = pd.get_dummies(_item_info, columns=["movie_title","release_date"], drop_first=False)

_df_item = df_user_item[["movie_id"]].merge(_item_info, on="movie_id", how="left")
movie_add = _df_item.values[:,1:]

S_U = Input(shape=[user_vec.shape[1]], dtype=np.float32, name="S_U")
X_U = Input(shape=[user_add.shape[1]], dtype=np.float32, name="X_U")

S_I = Input(shape=[movie_vec.shape[1]], dtype=np.float32, name="S_I")
X_I = Input(shape=[movie_add.shape[1]], dtype=np.float32, name="X_I")

# aSDAE-U
def aSDAE(S, X, latent_dim, hidden_unit_nums=[], stddev=0.1, lam=0.5):
    S_noise = GaussianNoise(stddev=0.01)(S)
    X_noise = GaussianNoise(stddev=0.01)(X)
    h = S_noise
    # num = 1000
    for num in hidden_unit_nums:
        h_s = Dense(units=num, 
                    activation=None, # softmax, sigmoid, relu
                    use_bias=True,
                    bias_initializer=initializers.zeros(),
                    kernel_initializer=initializers.TruncatedNormal(0.0, 0.01), 
                    kernel_regularizer=l2(0.001))(h)
        h_x = Dense(units=num, 
                    activation=None, # softmax, sigmoid, relu
                    use_bias=True,
                    bias_initializer=initializers.zeros(),
                    kernel_initializer=initializers.TruncatedNormal(0.0, 0.01), 
                    kernel_regularizer=l2(0.001))(X_noise)
        h = Add()([h_s, h_x])
        h = Activation("relu")(h)

    latent_s = Dense(units=latent_dim, 
                     activation=None, # softmax, sigmoid, relu
                     use_bias=True,
                     bias_initializer=initializers.zeros(),
                     kernel_initializer=initializers.TruncatedNormal(0.0, 0.01), 
                     kernel_regularizer=l2(0.001))(h)
    latent_x = Dense(units=latent_dim, 
                     activation=None, # softmax, sigmoid, relu
                     use_bias=True,
                     bias_initializer=initializers.zeros(),
                     kernel_initializer=initializers.TruncatedNormal(0.0, 0.01), 
                     kernel_regularizer=l2(0.001))(X_noise)
    latent = Add()([latent_s, latent_x])
    latent = Activation("relu")(latent)
    h = latent

    for num in hidden_unit_nums[::-1]:
        h_s = Dense(units=num, 
                    activation=None, # softmax, sigmoid, relu
                    use_bias=True,
                    bias_initializer=initializers.zeros(),
                    kernel_initializer=initializers.TruncatedNormal(0.0, 0.01), 
                    kernel_regularizer=l2(0.001))(h)
        h_x = Dense(units=num, 
                    activation=None, # softmax, sigmoid, relu
                    use_bias=True,
                    bias_initializer=initializers.zeros(),
                    kernel_initializer=initializers.TruncatedNormal(0.0, 0.01), 
                    kernel_regularizer=l2(0.001))(X_noise)
        h = Add()([h_s, h_x])
        h = Activation("relu")(h)

    S_ = Dense(units=int(S.shape[1]), 
                 activation=None, # softmax, sigmoid, relu
                 use_bias=True,
                 bias_initializer=initializers.zeros(),
                 kernel_initializer=initializers.TruncatedNormal(0.0, 0.01), 
                 kernel_regularizer=l2(0.001))(h)
    S_ = Activation("relu")(S_)

    X_ = Dense(units=int(X.shape[1]), 
                 activation=None, # softmax, sigmoid, relu
                 use_bias=True,
                 bias_initializer=initializers.zeros(),
                 kernel_initializer=initializers.TruncatedNormal(0.0, 0.01), 
                 kernel_regularizer=l2(0.001))(h)
    X_ = Activation("relu")(X_)
    return latent, S_, X_

U_latent, S_U_, X_U_ = aSDAE(S_U, X_U, latent_dim=8, hidden_unit_nums=[1000,500], lam=0.5)
I_latent, S_I_, X_I_ = aSDAE(S_I, X_I, latent_dim=8, hidden_unit_nums=[700,500], lam=0.5)

y_ = Dot(axes=1)([U_latent, I_latent])

s_u = tf.reduce_sum(tf.square(tf.subtract(S_U,S_U_)))
x_u = tf.reduce_sum(tf.square(tf.subtract(X_U,X_U_)))
s_i = tf.reduce_sum(tf.square(tf.subtract(S_I,S_I_)))
x_i = tf.reduce_sum(tf.square(tf.subtract(X_I,X_I_)))
u = tf.reduce_sum(tf.square(U_latent))
i = tf.reduce_sum(tf.square(I_latent))

alpha_1 = 0.5; alpha_2 = 0.5
loss = lambda y_true, y_pred: tf.reduce_sum(tf.square(y_true - y_pred)) \
                                    + alpha_1*s_u + (1-alpha_1)*x_u \
                                    + alpha_2*s_i + (1-alpha_2)*x_i \
                                    + 0.5*(u+i)
model = Model(inputs=[S_U,X_U,S_I,X_I], outputs=y_)
model.summary()
#plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)

bs = 64; epc = 10; lr = 0.1; dcy = 0.01
model.compile(loss=loss, 
              optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=10**-8, decay=dcy), # , decay=dcy
              metrics=["mse"]) # accuracy, mse
early_stopping = EarlyStopping(monitor="loss", patience=2, mode="auto", verbose=1)
model_fit = model.fit([user_vec, user_add, movie_vec, movie_add], rating, batch_size=bs, epochs=epc, verbose=1, shuffle=True, callbacks=[early_stopping])

