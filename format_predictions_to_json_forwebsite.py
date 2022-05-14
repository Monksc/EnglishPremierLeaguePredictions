import epl
import pagerank
import predictions_tensorflow
import probs_of_finishing_each_place as probs
import json
import numpy as np

def main(csv_file_name='epl.csv', output_file_name='epl-predictions-stats.json'):

    model = epl.getData(csv_file_name)
    data, indexToTeam, teamToIndex, indexToGamesPlayed = model
    predictor = predictions_tensorflow.createPredictGameFunction(csv_file_name)
    indexToPlaceFinishedToTimesFinished, indexToMeanPoints, indexToTeam = probs.calculateProbs(10**5, csv_file_name, predictor)

    week = 40

    A = np.array(epl.getMatrixForSeason(week, model, None))
    R = pagerank.rank(A)

    rankings = [(indexToMeanPoints[i], i) for i in range(len(indexToMeanPoints))]
    rankings.sort(reverse=True)

    jsonData = []
    for value in rankings:
        probabilties = indexToPlaceFinishedToTimesFinished[value[1]]
        jsonData.append({
            "name": indexToTeam[value[1]],
            "probability": list(probabilties),
            "expected": value[0],
            "championslegue": probabilties[0] + probabilties[1] + probabilties[2] + probabilties[3],
            "relegated": probabilties[-1] + probabilties[-2] + probabilties[-3],
            "pagerank": R[value[1]][0],
        })

    games_data = probs.calculateProbsOfEachGameInASeason(csv_file_name, predictor)

    with open(output_file_name, 'w') as out:
        out.write(json.dumps({'teams': list(jsonData), 'games': games_data}))


if __name__ == '__main__':
    import os
    csv_file_name = 'epl.csv'
    output_file_name = 'epl-predictions-stats.json'
    if len(os.sys.argv) > 1:
        csv_file_name = os.sys.argv[1]
    if len(os.sys.argv) > 2:
        output_file_name = os.sys.argv[2]

    print(csv_file_name, output_file_name)
    main(csv_file_name, output_file_name)

