import tensorflow as tf
import keras
import numpy as np
import random
import collect_data as cdata
import epl

print(tf.version.VERSION)

def makeModel(inputs, output_len):

    input = tf.keras.Input(shape=(inputs.shape[1],))
    norm = keras.layers.experimental.preprocessing.Normalization()
    norm.adapt(inputs)
    layer1 = norm(input)

    model = keras.Sequential([
        layer1,
        keras.layers.Dense(8, activation='sigmoid', input_shape=(inputs.shape[1],)),
        keras.layers.Dense(6, activation='sigmoid'),
        keras.layers.Dense(output_len, activation='sigmoid'),
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


    #model.load_weights('saved_model/model5.ckpt')

    return model

def trainModel(model, train_features, train_labels, batch_size=128, epochs=16):

    print("SHAPES:")
    print(train_features.shape, train_labels.shape)
    for i in range(len(train_labels)):
        if train_labels[i].max() > 1.0 or train_labels[i].min() < 0:
            print(train_labels[i])

    
    careful_bias_history = model.fit(
        train_features,
        train_labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1)
        #validation_data=(train_features, train_labels), 
        #verbose=0)


def getData():
    return cdata.getAllMyData()


def train():
    features, labels = getData()
    
    indexes = np.arange(len(features))
    random.seed(42)
    random.shuffle(indexes,)
    features = features[indexes,]
    labels = labels[indexes,]
    
    train_size = 1000

    train_features = features[:-train_size]
    train_labels = labels[:-train_size]
    
    testing_features = features[-train_size:]
    testing_labels   = labels[-train_size:]
    
    model = makeModel(train_features, train_labels.shape[1])
    trainModel(model, train_features, train_labels, batch_size=128, epochs=2**10)
    trainModel(model, train_features, train_labels, batch_size=train_labels.shape[0], epochs=2**10)
        
    print(model.predict(train_features))
    results = model.evaluate(testing_features, testing_labels, batch_size=len(testing_features), verbose=0)
    
    print("Loss: {:0.4f}".format(results[0]))
    for name, value in zip(model.metrics_names, results):
        print(name, ': ', value)
    
    # Save the entire model as a SavedModel.
    model.save_weights("saved_model/model5.ckpt")

    print("Training Features Shape: ", train_features.shape, "Testing Labels Shape: ", testing_labels.shape)

def predictGames(filename):

    inputs, labels = cdata.getAllMyData([filename], True, 0)
    data, indexToTeam, teamToIndex, indexToGamesPlayed = epl.getData(filename)

    model = makeModel(inputs, labels.shape[1])
    outputs = model.predict(inputs)

    print("SHAPE: ", inputs.shape, labels.shape, outputs.shape)

    print("%20s : %s Goals %s %-20s     Home   Tie    Away         Accurate (Should be close to 1 for accuracy)" % ("Home Team", " " * 2, " " * 5, "Away Team"))

    indexToPoints = [0.0 for i in range(len(indexToTeam))]

    for i in range(len(data["Home Team"])):

        homeTeam = data["Home Team"][i]
        awayTeam = data["Away Team"][i]
    
        homeScore = None
        awayScore = None
        if (type(data["Result"][i]) is str):
            result = data["Result"][i].split("-")
            if len(result) == 2:
                homeScore = int(result[0].strip())
                awayScore = int(result[1].strip())
    
        homeIndex = teamToIndex[homeTeam]
        awayIndex = teamToIndex[awayTeam]
    
        homeTeamPlayed = indexToGamesPlayed[homeIndex]
        awayTeamPlayed = indexToGamesPlayed[awayIndex]

        output = np.array(outputs[i])
        p_output = output / output.sum()

        if homeScore != None:
            if homeScore > awayScore:
                indexToPoints[homeIndex] += 3.0
            elif homeScore < awayScore:
                indexToPoints[awayIndex] += 3.0
            else:
                indexToPoints[homeIndex] += 1.0
                indexToPoints[awayIndex] += 1.0
        else:
            indexToPoints[homeIndex] += p_output[0] * 3.0 + p_output[2]
            indexToPoints[awayIndex] += p_output[1] * 3.0 + p_output[2]

        if homeScore == None:
            print("%20s : %s vs %s : %-20s     %4.2f   %4.2f   %4.2f   %10.2f %8.2f" % 
                    (homeTeam, " " * 4, " " * 4, awayTeam, p_output[0], p_output[2], p_output[1], output.sum(), 1.0 - (5 * (output.sum() - 1.0)) ** 2))
        else:
            print("%20s : %4d vs %-4d : %-20s     %4.2f   %4.2f   %4.2f   %10.2f %8.2f" % 
                    (homeTeam, homeScore, awayScore, awayTeam, p_output[0], p_output[2], p_output[1], output.sum(), 1.0 - (5 * (output.sum() - 1.0)) ** 2))

    points = np.array([indexToPoints, np.arange(len(indexToPoints))])
    points = np.array(sorted(points.T, key=lambda x:x[0], reverse=True))
    print("\n\n")
    print("   %-30s %5s" % ("Team", "Points"))
    for i in range(len(points)):
        print("%2d %-30s %5.3f" % (i+1, indexToTeam[int(points[i][1])], points[i][0]))


if __name__ == "__main__":
    train()
    predictGames('epl-2020-week-0.csv')
    #predictGames('old_data/efl-championship-2020.csv')
    #predictGames('old_data/la-liga-2019.csv')
    #predictGames('old_data/bundesliga-2020.csv')
    #predictGames('old_data/ligue-1-2020.csv')
    #predictGames('old_data/serie-a-2020.csv')
    #predictGames('old_data/turkey-super-lig-2020.csv')
    #predictGames('test.csv')
