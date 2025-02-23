# Select a dataset among 'ML-100K' and 'ML-1M'
dataset = 'ML-1M'
ML_1M_TEST_SIZE = 0.1

# Model hyperparameters
BATCH_SIZE = 1
LEARNING_RATE = 0.002
REGULARIZATION = 0.05
EPOCHS = 100

# Matrix factorization hyperparameters
LATENT_DIM = 25 # Concepts count

# TIMESTAMP
DAY_SECONDS_LENGTH = 86400

# Session
SESSION_TIME_GAP_IN_DAYS = 7 # days
SESSION_TIME_GAP_SECONDS = SESSION_TIME_GAP_IN_DAYS * DAY_SECONDS_LENGTH

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

import numpy as np

from sklearn.metrics import root_mean_squared_error

import pickle

import sys

# check if there are 2 arguments
if len(sys.argv) == 2:
    dataset = sys.argv[1]
    LATENT_DIM = int(sys.argv[2])
    print(f"Dataset: {dataset}, Latent Dimension: {LATENT_DIM}")

def load_data_100k(path='./', delimiter='\t'):
    train = np.loadtxt(path+'movielens_100k_u1.base', skiprows=0, delimiter=delimiter).astype('int32')
    test = np.loadtxt(path+'movielens_100k_u1.test', skiprows=0, delimiter=delimiter).astype('int32')

    total = np.concatenate((train, test), axis=0)
    test_size = len(test)
    total = total[total[:,3].argsort()] # Sort by timestamp
    
    train = total[:-test_size]
    test = total[-test_size:]

    train_users, train_items = set(train[:, 0]), set(train[:, 1])
    test = test[[(test_record[0] in train_users and test_record[1] in train_items) for test_record in test]]

    user_id_dict = {}
    for i, user_id in enumerate(np.unique(train[:,0]).tolist()):
        user_id_dict[user_id] = i
    
    item_id_dict = {}
    for i, item_id in enumerate(np.unique(train[:,1]).tolist()):
        item_id_dict[item_id] = i

    train = np.array([(user_id_dict[record[0]], item_id_dict[record[1]], record[2], record[3]) for record in train])
    test = np.array([(user_id_dict[record[0]], item_id_dict[record[1]], record[2], record[3]) for record in test])

    n_u = np.unique(train[:,0]).size  # num of users
    n_m = np.unique(train[:,1]).size  # num of movies
    n_train = train.shape[0]  # num of training ratings
    n_test = test.shape[0]  # num of test ratings

    print('data matrix loaded')
    print('num of users: {}'.format(n_u))
    print('num of movies: {}'.format(n_m))
    print('num of training ratings: {}'.format(n_train))
    print('num of test ratings: {}'.format(n_test))

    return n_m, n_u, train, test

def load_data_1m(path='./', delimiter='::', test_size=ML_1M_TEST_SIZE):
    data = np.genfromtxt(path+'movielens_1m_dataset.dat', skip_header=0, delimiter=delimiter).astype('int32')
    data = data[(-data[:,3]).argsort()]

    n_r = data.shape[0]  # num of ratings
    
    train, test = [], []
    user_id_dict, item_id_dict = {}, {}

    for i in range(n_r - 1, -1, -1):
        user_id, item_id, rating, timestamp = data[i]
        if i < int(test_size * n_r): # test set
            if user_id in user_id_dict and item_id in item_id_dict:
                test.append((user_id_dict[user_id], item_id_dict[item_id], rating, timestamp))
        else: # training set
            if user_id not in user_id_dict:
                user_id_dict[user_id] = len(user_id_dict)
            if item_id not in item_id_dict:
                item_id_dict[item_id] = len(item_id_dict)
            train.append((user_id_dict[user_id], item_id_dict[item_id], rating, timestamp))
    
    train, test = np.array(train), np.array(test)

    n_u = np.unique(train[:,0]).size  # num of users
    n_m = np.unique(train[:,1]).size  # num of movies
    n_train = train.shape[0]  # num of training ratings
    n_test = test.shape[0]  # num of test ratings

    print('data matrix loaded')
    print('num of users: {}'.format(n_u))
    print('num of movies: {}'.format(n_m))
    print('num of ratings: {}'.format(n_r))
    print('num of training ratings: {}'.format(n_train))
    print('num of test ratings: {}'.format(n_test))

    return n_m, n_u, train, test

# Insert the path of a data directory by yourself (e.g., '/content/.../data')
# .-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._
data_path = 'data'
# .-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._

# Data Load
try:
    if dataset == 'ML-100K':
        path = data_path + '/MovieLens_100K/'
        n_m, n_u, train, test = load_data_100k(path=path, delimiter='\t')

    elif dataset == 'ML-1M':
        path = data_path + '/MovieLens_1M/'
        n_m, n_u, train, test = load_data_1m(path=path, delimiter='::')

    else:
        raise ValueError

except ValueError as e:
    print('Error: Unable to load data')

def divide_to_sessions(data: list[tuple[int, int, int, int]]) -> dict[tuple[int, int], list[list[int]]]:
    user_sessions_dict = {}
    for user_id, item_id, _, timestamp in data:
        user_id, item_id, timestamp = int(user_id), int(item_id), int(timestamp)
        if user_id not in user_sessions_dict:
            user_sessions_dict[user_id] = []
        if len(user_sessions_dict[user_id]) == 0 or timestamp - user_sessions_dict[user_id][-1][-1][1] > SESSION_TIME_GAP_SECONDS:
            user_sessions_dict[user_id].append([])
        user_sessions_dict[user_id][-1].append((item_id, timestamp))
    for user_id, user_sessions in user_sessions_dict.items():
        for user_session_index, user_session_items_ids in enumerate(user_sessions):
            user_sessions_dict[user_id][user_session_index] = [item_id for item_id, _ in user_session_items_ids], [timestamp // DAY_SECONDS_LENGTH for _, timestamp in user_session_items_ids]
    user_item_sessions_dict = {}
    for user_id, user_sessions in user_sessions_dict.items():
        for user_session_items_ids, user_session_items_daystamps in user_sessions:
            for item_id in user_session_items_ids:
                user_item_sessions_dict[(user_id, item_id)] = tuple(user_session_items_ids), tuple(user_session_items_daystamps)
    return user_item_sessions_dict

sessions_dict = divide_to_sessions(train)

class MatrixFactorization(tf.keras.Model):
    def __init__(self, users_count, items_count, average_rating, current_day, latent_dim=LATENT_DIM, regularization_factor=REGULARIZATION):
        super().__init__()
        self.regularization_factor = regularization_factor
        self.user_embeddings = Embedding(users_count, latent_dim,
                                  embeddings_regularizer=l2(regularization_factor),
                                  name="user_embedding")
        self.item_embeddings = Embedding(items_count, latent_dim,
                                  embeddings_regularizer=l2(regularization_factor),
                                  name="item_embedding")
        self.average_rating = average_rating
        self.user_biases = Embedding(users_count, 1,
                                   embeddings_regularizer=l2(regularization_factor),
                                   name="user_bias")
        self.item_biases = Embedding(items_count, 1,
                                   embeddings_regularizer=l2(regularization_factor),
                                   name="item_bias")
        self.session_biases = Embedding(len(set([session_items_ids for session_items_ids, _ in sessions_dict.values()])), 1,
                                      embeddings_regularizer=l2(regularization_factor),
                                      name="session_bias")
        self.user_session_decaying_rates = Embedding(users_count, 1,
                                   embeddings_regularizer=l2(regularization_factor),
                                   name="user_session_decaying_rate")
        self.current_day = tf.constant(current_day, dtype=tf.float32)


        print("finish init")

    def get_session_presentation(self, user_id, session_items_ids, session_items_daystamps):
        user_embedding = self.user_embeddings(user_id)
        
        session_items_ids = tf.boolean_mask(session_items_ids, session_items_ids != -1)
        session_items_daystamps = tf.boolean_mask(session_items_daystamps, session_items_daystamps != -1)
        session_items_embeddings = self.item_embeddings(session_items_ids)
        
        session_items_predicts = tf.reduce_sum(user_embedding[:, None, :] * session_items_embeddings, axis=2)
        user_decaying_rate = tf.squeeze(self.user_session_decaying_rates(user_id))
        user_decaying_rate = tf.nn.relu(user_decaying_rate)
        session_items_decaying_factors = tf.exp(-tf.abs(session_items_daystamps - self.current_day)
                                                 * user_decaying_rate
                                                 )

        session_items_scores = session_items_predicts * session_items_decaying_factors
        return tf.reduce_mean(session_items_scores, axis=1)

    def call(self, inputs):
        user_id, item_id, session_items_ids, session_items_daystamps = inputs

        user_embedding = self.user_embeddings(user_id)
        item_embedding = self.item_embeddings(item_id)
        raw_prediction = tf.reduce_sum(user_embedding * item_embedding, axis=1)

        session_predict = self.get_session_presentation(user_id, session_items_ids, session_items_daystamps)

        total_bias = (
            tf.squeeze(self.user_biases(user_id)) +
            tf.squeeze(self.item_biases(item_id)) +
            tf.squeeze(self.session_biases(session_predict))
        )

        return raw_prediction + self.average_rating + total_bias
    
    def l2_loss(self, y_true, y_predict, user_id, item_id, session_items_ids, session_items_daystamps):
        squared_error = tf.square(y_true - y_predict)
        user_embedding, item_embedding = self.user_embeddings(user_id), self.item_embeddings(item_id)
        user_embedding_norm = tf.reduce_sum(tf.square(user_embedding), axis=1)
        item_embedding_norm = tf.reduce_sum(tf.square(item_embedding), axis=1)
        user_bias = tf.squeeze(self.user_biases(user_id))
        item_bias = tf.squeeze(self.item_biases(item_id))
        session_predict = self.get_session_presentation(user_id, session_items_ids, session_items_daystamps)
        session_bias = tf.squeeze(self.session_biases(session_predict))
        user_bias_norm = tf.square(user_bias)
        item_bias_norm = tf.square(item_bias)
        session_bias_norm = tf.square(session_bias)
        session_decaying_rate = tf.squeeze(self.user_session_decaying_rates(user_id))
        session_decaying_rate_norm = tf.square(session_decaying_rate)

        regularized_loss = self.regularization_factor * (user_embedding_norm + item_embedding_norm + user_bias_norm + item_bias_norm + session_bias_norm + session_decaying_rate_norm)
        return squared_error + regularized_loss
    
    def train_step(self, data):
        (user_id, item_id, session_items_ids, session_items_daystamps), y_true = data

        with tf.GradientTape() as tape:
            y_predict = self((user_id, item_id, session_items_ids, session_items_daystamps), training=True)
            loss = self.l2_loss(y_true, y_predict, user_id, item_id, session_items_ids, session_items_daystamps)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": loss}
    
train_users, train_items, train_ratings = train[:,0], train[:,1], np.float32(train[:,2])
train_sessions_items = [sessions_dict[(row[0], row[1])][0][: sessions_dict[(row[0], row[1])][0].index(row[1]) + 1] for row in train]
train_sessions_daystamps = [sessions_dict[(row[0], row[1])][1][: sessions_dict[(row[0], row[1])][0].index(row[1]) + 1] for row in train]
max_session_length = max(len(session_items) for session_items in train_sessions_items)
train_sessions_items = np.array([session_items + (-1, ) * (max_session_length - len(session_items)) for session_items in train_sessions_items])
train_sessions_daystamps = np.array([session_daystamps + (-1, ) * (max_session_length - len(session_daystamps)) for session_daystamps in train_sessions_daystamps], dtype=np.float32)
test_users, test_items, test_ratings = test[:,0], test[:,1], np.float32(test[:,2])

train_average_rating = np.mean(train_ratings)
test_average_timestamp = np.mean(test[:,3])
test_average_daystamp = int(test_average_timestamp // DAY_SECONDS_LENGTH)

model = MatrixFactorization(n_u, n_m, train_average_rating, test_average_daystamp)
model.compile(optimizer=SGD(learning_rate=LEARNING_RATE))
history = model.fit(
    [train_users, train_items, train_sessions_items, train_sessions_daystamps], train_ratings,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1
)

(users_matrix, items_matrix, user_biases, item_biases, _, _), average_rating = model.get_weights(), model.average_rating
ratings_matrix = np.dot(users_matrix, items_matrix.T) + user_biases + item_biases.T + train_average_rating
ratings_matrix = np.clip(ratings_matrix, 1, 5)

# Save the model
with open(path + f"mf_prediction_{LATENT_DIM}_dims.pickle", 'wb') as f:
    pickle.dump(ratings_matrix, f)

# Get Test Score
# test_ratings_predicted = np.clip(model.predict([test_users, test_items]), 1, 5) # predict using the model
test_ratings_predicted = np.array([ratings_matrix[test_user, test_item] for test_user, test_item in zip(test_users, test_items)]) # predict using the matrix

# check test rmse
test_rmse = root_mean_squared_error(test_ratings, test_ratings_predicted)
print(f"Test RMSE: {test_rmse}")