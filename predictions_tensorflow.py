import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import random
import collect_data as cdata
import epl

print(tf.version.VERSION)

def makeModel(inputs, input_len, output_len):

    print(input_len, output_len)

    #input = tf.keras.Input(shape=(input_len, ))
    #norm = keras.layers.experimental.preprocessing.Normalization()
    #norm.adapt(inputs)
    model = keras.Sequential([
        #norm,
        keras.layers.Dense(128, activation='sigmoid', input_shape=(input_len, )),
        keras.layers.Dense(64, activation='sigmoid'),
        keras.layers.Dense(64, activation='sigmoid'),
        keras.layers.Dense(64, activation='sigmoid'),
        keras.layers.Dense(output_len, activation='sigmoid'),
    ])
    #model = keras.Sequential([
    #    norm,
    #    keras.layers.Dense(32, activation='sigmoid', input_shape=(inputs.shape[1],),
    #        kernel_regularizer=keras.regularizers.l2(0.001)),
    #    keras.layers.Dense(32, activation='sigmoid',
    #        kernel_regularizer=keras.regularizers.l2(0.001)),
    #    keras.layers.Dense(output_len, activation='sigmoid',
    #        kernel_regularizer=keras.regularizers.l2(0.001)),
    #])
    
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        #optimizer=keras.optimizers.Adam(lr=1e-3),
        #loss=keras.losses.BinaryCrossentropy(),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=metrics)


    #model.load_weights('saved_model/model3.ckpt')
    model.load_weights('saved_model/model-9-9-2022.ckpt')

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
    return cdata.getAllMyData(filenames=cdata.AllSeasonsFootballCSV)

def train():
    #features, labels = getData()

    features_str = [
        "A-Rank-Home", "A-Rank-Away", "A-week-Rank-Home", "A-week-Rank-Away",
        "A-Goals-Rank-Home", "A-Goals-Rank-Away", "A-Goals-week-Rank-Home", "A-Goals-week-Rank-Away",
        "HomeTeam-Home-wins", "AwayTeam-Home-wins", "HomeTeam-Home-loss",
        "AwayTeam-Home-loss", "HomeTeam-Home-tie", "AwayTeam-Home-tie",
        "HomeTeam-Home-GoalsF", "AwayTeam-Home-GoalsF", "HomeTeam-Home-GoalsA",
        "AwayTeam-Home-GoalsA", "HomeTeam-Away-wins", "AwayTeam-Away-wins",
        "HomeTeam-Away-loss", "AwayTeam-Away-loss", "HomeTeam-Away-tie",
        "AwayTeam-Away-tie", "HomeTeam-Away-GoalsF", "AwayTeam-Away-GoalsF",
        "HomeTeam-Away-GoalsA", "AwayTeam-Away-GoalsA", "A-flip-Rank-Home",
        "A-flip-Rank-Away", "A-flip-week-Rank-Home", "A-flip-week-Rank-Away",
        "A-flip-Goals-Rank-Home", "A-flip-Goals-Rank-Away", "A-flip-Goals-week-Rank-Home",
        "A-flip-Goals-week-Rank-Away"
    ]

    targets_str = ["HWin", "Tie", "AWin"]

    training_df = pd.read_csv("old_data/jackpotdata/all/all.csv")

    features = training_df[features_str]
    targets  = training_df[targets_str]

    dataset = tf.data.Dataset.from_tensor_slices((features.values, targets.values))
    train_dataset = dataset.shuffle(len(training_df)).batch(1)
    
    #indexes = np.arange(len(features))
    #random.seed(42)
    #random.shuffle(indexes,)
    #features = features[indexes,]
    #labels = labels[indexes,]
    
    #train_size = 10000

    #train_features = features[:-train_size]
    #train_labels = labels[:-train_size]
    #
    #testing_features = features[-train_size:]
    #testing_labels   = labels[-train_size:]
    
    #model = makeModel(train_features, train_labels.shape[1])
    model = makeModel(train_dataset, len(features_str), len(targets_str))

    model.summary()
    model.fit(train_dataset, epochs=1)

    #trainModel(model, train_features, train_labels, batch_size=128, epochs=2**10)
    #trainModel(model, train_features, train_labels, batch_size=train_labels.shape[0], epochs=2**10)
    #trainModel(model, train_features, train_labels, batch_size=128, epochs=2**2)
    #trainModel(model, train_features, train_labels, batch_size=train_labels.shape[0], epochs=2**1)
        
    #print(model.predict(train_features))
    #results = model.evaluate(testing_features, testing_labels, batch_size=len(testing_features), verbose=0)
    
    #print("Loss: {:0.4f}".format(results[0]))
    #for name, value in zip(model.metrics_names, results):
    #    print(name, ': ', value)
    
    # Save the entire model as a SavedModel.
    #model.save_weights("saved_model/model-tie.ckpt")
    model.save_weights("saved_model/model-9-9-2022.ckpt")

    #print("Training Features Shape: ", train_features.shape, "Testing Labels Shape: ", testing_labels.shape)

    #print("Weights")
    #print(model.trainable_variables)

def createPredictGameFunction(filename):
    inputs, labels = cdata.getAllMyData([filename], True, 0)
    data, indexToTeam, teamToIndex, indexToGamesPlayed = epl.getData(filename)

    model = makeModel(inputs, inputs.shape[1], labels.shape[1])
    outputs = model.predict(inputs)

    def predict(homeIndex, awayIndex, index):
        output = np.array(outputs[index])
        p_output = output / output.sum()
        return (p_output[0], p_output[2], p_output[1])

    return predict

def predictGames(filename):

    inputs, labels = cdata.getAllMyData([filename], True, 0)
    data, indexToTeam, teamToIndex, indexToGamesPlayed = epl.getData(filename)

    model = makeModel(inputs, inputs.shape[1], labels.shape[1])
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

        # Print Tie
        #print("%4.2f %4.2f %4.2f %4.2f" % (inputs[i][12], inputs[i][13], inputs[i][22], inputs[i][23]))
        if homeScore == None:
            print("%20s : %s vs %s : %-20s     %4.2f   %4.2f   %4.2f   %10.2f %8.2f" % 
                    (homeTeam, " " * 4, " " * 4, awayTeam, p_output[0], p_output[2], p_output[1], output.sum(), 1.0 - (5 * (output.sum() - 1.0)) ** 2))
            #print("%20s : %s vs %s : %-20s     %4.2f" % 
            #        (homeTeam, " " * 4, " " * 4, awayTeam, output[0]))
        else:
            print("%20s : %4d vs %-4d : %-20s     %4.2f   %4.2f   %4.2f   %10.2f %8.2f" % 
                    (homeTeam, homeScore, awayScore, awayTeam, p_output[0], p_output[2], p_output[1], output.sum(), 1.0 - (5 * (output.sum() - 1.0)) ** 2))
            #print("%20s : %4d vs %-4d : %-20s     %4.2f" % 
            #        (homeTeam, homeScore, awayScore, awayTeam, output[0]))

    points = np.array([indexToPoints, np.arange(len(indexToPoints))])
    points = np.array(sorted(points.T, key=lambda x:x[0], reverse=True))
    print("\n\n")
    print("   %-30s %5s" % ("Team", "Points"))
    for i in range(len(points)):
        print("%2d %-30s %5.3f" % (i+1, indexToTeam[int(points[i][1])], points[i][0]))


    #print("Weights")
    #print(model.trainable_variables)


if __name__ == "__main__":
    #train()
    #predictGames('old_data/american-football.csv/nfl-2020.csv')
    predictGames('epl.csv')
    #predictGames('old_data/efl-championship-2020.csv')
    #predictGames('old_data/la-liga-2019.csv')
    #predictGames('old_data/bundesliga-2020.csv')
    #predictGames('old_data/ligue-1-2020.csv')
    #predictGames('old_data/serie-a-2020.csv')
    #predictGames('old_data/turkey-super-lig-2020.csv')
