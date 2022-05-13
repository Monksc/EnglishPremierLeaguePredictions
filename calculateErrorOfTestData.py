#!/usr/bin/env python3

import numpy as np
import pandas as pd
import json

def getPrediction(jsonPrediction, weekNumber, homeTeam, awayTeam):
    for i in range(len(jsonPrediction)):
        if jsonPrediction[i]['Round Number'] == weekNumber and jsonPrediction[i]['Home Team'] == homeTeam and jsonPrediction[i]['Away Team'] == awayTeam:
            return jsonPrediction[i]

    return None

def getErrors(resultsDataFileName='epl-2021.csv',
        predictedFileName='epl-predictions-stats.json', afterWeek=15):
    results = pd.read_csv(resultsDataFileName)
    predicted = json.load(open('epl-predictions-stats.json'))['games']

    meanSquaredError = 0.0
    correctAnswers = 0
    totalResults = 0
    isWin = 0
    isTie = 0
    for i in range(len(results)):

        if results['Round Number'][i] < 11:
            continue

        homeScore = 0
        awayScore = 0
        try:
            scores = results.Result[i].split('-')
            homeScore = int(scores[0])
            awayScore = int(scores[1])
        except:
            continue

        totalResults += 1

        p = getPrediction(predicted, results['Round Number'][i], results['Home Team'][i], results['Away Team'][i])
        homeWinP = p['home_win']
        awayWinP = p['away_win']
        tieP = p['tie']
        prediction = 0
        if awayWinP > homeWinP and awayWinP > tieP:
            prediction = 1
        elif tieP > homeWinP and tieP > awayWinP:
            prediction = 2

        if (homeScore > awayScore and prediction == 0) or (awayScore > homeScore and prediction == 1) or (homeScore == awayScore and prediction == 2):
            correctAnswers += 1

        correctGuess = [0.0, 0.0, 0.0]
        if homeScore > awayScore:
            isWin += 1
            correctGuess[0] = 1.0
        elif awayScore > homeScore:
            isWin += 1
            correctGuess[1] = 1.0
        elif homeScore == awayScore:
            isTie += 1
            correctGuess[2] = 1.0

        diff = np.array(correctGuess) - np.array([homeWinP, awayWinP, tieP])

        meanSquaredError += (diff * diff).sum() / 3.0

    return (meanSquaredError / totalResults, correctAnswers/totalResults, isWin, isTie)


def main(resultsDataFileName='epl-2021.csv',
        predictedFileName='epl-predictions-stats.json'):
    (meanSquaredError, percentCorrect, isWin, isTie) = getErrors(resultsDataFileName, predictedFileName)
    print(meanSquaredError, percentCorrect, isWin, isTie)


if __name__ == '__main__':
    main()
