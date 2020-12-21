import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)


url = 'game_data.csv'
column_names = ['home_goals_foward','home_goals_against','away_goals_foward',
                'away_goals_against','home_rank','away_rank','home_goals','away_goals']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=',', skiprows=1)

dataset = raw_dataset.copy()
dataset.tail()

dataset.isna().sum()
dataset = dataset.dropna()

# If we wanted a one hot coding
#dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})

dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
dataset.tail()


# Now split the dataset into a training set and a test set
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Inspect the data
# sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
sns.pairplot(train_dataset[column_names], diag_kind='kde')

train_dataset.describe().transpose()

# Split features from labels
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('home_goals')
test_labels = test_features.pop('home_goals')


# Normalize data
train_dataset.describe().transpose()[['mean', 'std']]


# The Normalization layer
normalizer = preprocessing.Normalization()

normalizer.adapt(np.array(train_features))

print(normalizer.mean.numpy())
first = np.array(train_features[:1])


with np.printoptions(precision=2, suppress=True):
    print('First example:', first)
    print()
    print('Normalized:', normalizer(first).numpy())



target = np.array(train_features['home_rank'])

target_normalizer = preprocessing.Normalization(input_shape=[1,])
target_normalizer.adapt(target)


target_model = tf.keras.Sequential([
    target_normalizer,
    layers.Dense(units=1)
])

target_model.summary()


target_model.predict(target[:10])

target_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

#%%time
history = target_model.fit(
    train_features['home_rank'], train_labels,
    epochs=100,
    # suppress logging
    verbose=0,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


def plot_loss(history):
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('home_rank')
    plt.ylabel('Error [home_goals]')
    plt.legend()
    plt.grid(True)

plot_loss(history)

test_results = {}

test_results['target_model'] = target_model.evaluate(
    np.array(test_features['home_rank']),
    test_labels, verbose=0)


x = tf.linspace(0.0, 250, 251)
y = target_model.predict(x)


def plot_target(x, y):
    plt.figure()
    plt.scatter(train_features['home_rank'], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('Home Goals')
    plt.ylabel('Home Rank')
    plt.legend()

plot_target(x,y)



# Extras

def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model



# MARK: Predictions

dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

history = dnn_model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

plot_loss(history)

test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)

pd.DataFrame(test_results, index=['Mean absolute error [home_goals]']).T


test_predictions = dnn_model.predict(test_features).flatten()

plt.figure()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [home_goals]')
plt.ylabel('Predictions [home_goals]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


plt.figure()
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [home_goals]')
_ = plt.ylabel('Count')


dnn_model.save('dnn_model')


""" 
To reload model
reloaded = tf.keras.models.load_model('dnn_model')

test_results['reloaded'] = reloaded.evaluate(
    test_features, test_labels, verbose=0)
"""

pd.DataFrame(test_results, index=['Mean absolute error [home_goals]']).T

plt.show()
