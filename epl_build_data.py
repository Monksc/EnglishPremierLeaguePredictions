import numpy as np
import pagerank
import epl
import csv


def getInputs(model):

    data, indexToTeam, teamToIndex, indexToGamesPlayed = model

    inputs = []
    outputs = []

    lastWeek = 8
    hasHadWeek = False

    indexToTotalScoreF = [0.0 for i in range(len(indexToTeam))]
    indexToTotalScoreA = [0.0 for i in range(len(indexToTeam))]

    for i in range(len(data["Home Team"])):

        currentWeek = data["Round Number"][i]

        homeTeamIndex = teamToIndex[data["Home Team"][i]]
        awayTeamIndex = teamToIndex[data["Away Team"][i]]

        if not(type(data["Result"][i]) is str):
            continue
        result = data["Result"][i].split("-")
        if len(result) != 2:
            continue

        homeScore = int(result[0].strip())
        awayScore = int(result[1].strip())
        
        indexToTotalScoreF[homeTeamIndex] += homeScore
        indexToTotalScoreA[homeTeamIndex] += awayScore

        indexToTotalScoreF[awayTeamIndex] += awayScore
        indexToTotalScoreA[awayTeamIndex] += homeScore


        if currentWeek < lastWeek:
            continue

        if lastWeek < currentWeek or not(hasHadWeek):
            hasHadWeek = True

            A = epl.getMatrixForSeason(currentWeek-1, model)
            A = np.array(A)
            R = pagerank.rank(A)
            currentWeek = lastWeek


        # Gather Inputs

        newInputs = []
        newInputs.append((indexToTotalScoreF[homeTeamIndex] - homeScore) / (currentWeek-1))
        newInputs.append((indexToTotalScoreA[homeTeamIndex] - awayScore) / (currentWeek-1))
        
        newInputs.append((indexToTotalScoreF[awayTeamIndex] - awayScore) / (currentWeek-1))
        newInputs.append((indexToTotalScoreA[awayTeamIndex] - homeScore) / (currentWeek-1))

        newInputs.append(R[homeTeamIndex])
        newInputs.append(R[awayTeamIndex])


        # Gather Outputs

        newOutputs = [homeScore, awayScore]

        inputs.append(newInputs)
        outputs.append(newOutputs)

    return inputs, outputs



if __name__ == "__main__":

    inputs = []
    outputs = []

    for fileName in ["epl-2020-week-0.csv", "old_data/epl-2015.csv", "old_data/epl-2016.csv", "old_data/epl-2017.csv", "old_data/epl-2018.csv", "old_data/epl-2019.csv" ]:
        model = epl.getData(fileName)
        new_inputs, new_outputs = getInputs(model)

        inputs += new_inputs
        outputs += new_outputs
    
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    print(inputs.shape, outputs.shape)

    with open('game_data.csv', mode='w') as csv_file:

        fieldnames = ['home_goals_foward', 'home_goals_against', 'away_goals_foward', 'away_goals_against', 'home_rank', 'away_rank', 'home_goals', 'away_goals']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()

        for i in range(len(inputs)):

            data = {}
            for j in range(len(inputs[i])):
                if type (inputs[i][j]) == np.ndarray:
                    data[fieldnames[j]] = inputs[i][j][0]
                else:
                    data[fieldnames[j]] = inputs[i][j]
            for j in range(len(outputs[i])):
                data[fieldnames[j+len(inputs[i])]] = outputs[i][j]

            writer.writerow(data)


