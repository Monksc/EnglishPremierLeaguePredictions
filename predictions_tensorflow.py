import tensorflow as tf
import keras
import numpy as np
import random
import collect_data as cdata

print(tf.version.VERSION)

def makeModel(input_len, output_len):

    model = keras.Sequential([
        keras.layers.Dense(64, activation='sigmoid', input_shape=(input_len,)),
        keras.layers.Dense(64, activation='sigmoid'),
        keras.layers.Dense(64, activation='sigmoid'),
        keras.layers.Dense(output_len, activation='sigmoid',
            bias_initializer=None),
    ])
    
    metrics = [
          keras.metrics.TruePositives(name='tp'),
          keras.metrics.FalsePositives(name='fp'),
          keras.metrics.TrueNegatives(name='tn'),
          keras.metrics.FalseNegatives(name='fn'),
          keras.metrics.BinaryAccuracy(name='accuracy'),
          keras.metrics.Precision(name='precision'),
          keras.metrics.Recall(name='recall'),
          keras.metrics.AUC(name='auc'),
    ]
    
    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics)

    return model

def trainModel(model, train_features, train_labels, batch_size=128, epochs=16):

    print(train_features.shape, train_labels.shape)
    
    careful_bias_history = model.fit(
        train_features,
        train_labels,
        batch_size=batch_size,
        epochs=epochs)
        #validation_data=(train_features, train_labels), 
        #verbose=0)


def getData():
    return cdata.getAllMyData()


features, labels = getData()

indexes = np.arange(len(features))
random.shuffle(indexes,)
features = features[indexes,]
labels = labels[indexes,]

train_features = features[:-100]
train_labels = labels[:-100]

testing_features = features[-100:]
testing_labels   = labels[-100:]

model = makeModel(train_features.shape[1], train_labels.shape[1])
trainModel(model, train_features, train_labels, batch_size=256, epochs=2**6)
    
print(model.predict(train_features))
results = model.evaluate(testing_features, testing_labels, batch_size=len(testing_features), verbose=0)

print("Loss: {:0.4f}".format(results[0]))
for name, value in zip(model.metrics_names, results):
    print(name, ': ', value)


