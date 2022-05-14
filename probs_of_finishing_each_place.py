import epl
import predictions_tensorflow
import random
import numpy as np

def calculateProbsOfEachGameInASeason(csv_file_name, predict):
    data, indexToTeam, teamToIndex, indexToGamesPlayed = epl.getData(csv_file_name)

    game_data = []

    for i in range(len(data["Home Team"])):
        homeTeam = data["Home Team"][i]
        awayTeam = data["Away Team"][i]

        homeIndex = teamToIndex[homeTeam]
        awayIndex = teamToIndex[awayTeam]

        winnerProp, tieProb, loserProb = predict(homeIndex, awayIndex, i)
        total = winnerProp + tieProb + loserProb

        winnerProp = winnerProp / total
        tieProb = tieProb / total
        loserProb = loserProb / total

        try:
            result = data["Result"][i].split("-")
            homeGoals = int(result[0])
            awayGoals = int(result[1])
        except:
            homeGoals = None
            awayGoals = None


        game_data.append({
            'Home Team': homeTeam,
            'Away Team': awayTeam,
            'Round Number': int(data['Round Number'][i]),
            'Date': data['Date'][i],
            'Location': data['Location'][i],
            'home_win': float(winnerProp),
            'tie': float(tieProb),
            'away_win': float(loserProb),
            'home_goals': homeGoals,
            'away_goals': awayGoals,
        })

    return game_data


def calculateProbs(predictionCount, csv_file_name, predict):
    data, indexToTeam, teamToIndex, indexToGamesPlayed = epl.getData(csv_file_name)

    indexToPlaceFinishedToTimesFinished = [[0 for _ in range(len(indexToTeam))] for _ in range(len(indexToTeam))]
    indexToPoints = [0 for _ in range(len(indexToTeam))]
    indexToExpectedPoints = np.array([0.0 for _ in range(len(indexToTeam))])

    gameIndexLeftToPlay = []
    for i in range(len(data["Home Team"])):

        homeTeam = data["Home Team"][i]
        awayTeam = data["Away Team"][i]

        homeIndex = teamToIndex[homeTeam]
        awayIndex = teamToIndex[awayTeam]

        if not(type(data["Result"][i]) is str):
            gameIndexLeftToPlay.append(i)

            winnerProp, tieProb, loserProb = predict(homeIndex, awayIndex, i)
            total = winnerProp + tieProb + loserProb
            winnerProp = winnerProp / total
            tieProb = tieProb / total
            loserProb = loserProb / total

            indexToExpectedPoints[homeIndex] += winnerProp * 3 + tieProb
            indexToExpectedPoints[awayIndex] += loserProb * 3 + tieProb

            continue

        result = data["Result"][i].split("-")
        if len(result) != 2:
            gameIndexLeftToPlay.append(i)

            winnerProp, tieProb, loserProb = predict(homeIndex, awayIndex, i)
            total = winnerProp + tieProb + loserProb
            winnerProp = winnerProp / total
            tieProb = tieProb / total
            loserProb = loserProb / total

            indexToExpectedPoints[homeIndex] += winnerProp * 3 + tieProb
            indexToExpectedPoints[awayIndex] += loserProb * 3 + tieProb

            continue

        homeScore = int(result[0].strip())
        awayScore = int(result[1].strip())


        if homeScore > awayScore:
            indexToPoints[homeIndex] += 3
            indexToExpectedPoints[homeIndex] += 3
        elif homeScore < awayScore:
            indexToPoints[awayIndex] += 3
            indexToExpectedPoints[awayIndex] += 3
        else:
            indexToPoints[homeIndex] += 1
            indexToPoints[awayIndex] += 1
            indexToExpectedPoints[homeIndex] += 1
            indexToExpectedPoints[awayIndex] += 1


    for rounds in range(predictionCount):
        if rounds % 1000 == 0:
            print(rounds, (rounds / predictionCount))
        roundIndexToPoints = indexToPoints.copy()
        for i in gameIndexLeftToPlay:
            homeTeam = data["Home Team"][i]
            awayTeam = data["Away Team"][i]

            homeIndex = teamToIndex[homeTeam]
            awayIndex = teamToIndex[awayTeam]

            winnerProp, tieProb, loserProb = predict(homeIndex, awayIndex, i)
            total = winnerProp + tieProb + loserProb
            winnerProp = winnerProp / total
            tieProb = tieProb / total

            if random.random() <= winnerProp:
                roundIndexToPoints[homeIndex] += 3
            elif random.random() <= winnerProp + tieProb:
                roundIndexToPoints[homeIndex] += 1
                roundIndexToPoints[homeIndex] += 1
            else:
                roundIndexToPoints[awayIndex] += 3

        rankings = [(roundIndexToPoints[i], i) for i in range(len(roundIndexToPoints))]
        rankings.sort(reverse=True)

        for i in range(len(rankings)):
            indexToPlaceFinishedToTimesFinished[rankings[i][1]][i] += 1

    return np.array(indexToPlaceFinishedToTimesFinished) / predictionCount, indexToExpectedPoints, indexToTeam



def main():
    predictor = predictions_tensorflow.createPredictGameFunction('epl.csv')
    indexToPlaceFinishedToTimesFinished, indexToTeam = calculateProbs(10**3, 'epl.csv', predictor)

    print(indexToPlaceFinishedToTimesFinished)
    print(indexToTeam)

if __name__ == '__main__':
    main()

