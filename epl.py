import numpy as np
import pandas as pd
import random
import pagerank
import sys


def getData(csvFileName, flip_result=False):

    data = pd.read_csv(csvFileName)

    indexToTeam = []
    teamToIndex = {}
    indexToGamesPlayed = []
    for i in range(len(data["Home Team"])):

        team = data["Home Team"][i]
        index = len(indexToTeam)

        if not(team in teamToIndex):
            indexToTeam.append(team)
            teamToIndex[team] = index
            indexToGamesPlayed.append(0.0)

        if (type(data["Result"][i]) is str):
            if len(data["Result"][i].split("-")) == 2:
                index = teamToIndex[team]
                indexToGamesPlayed[index] += 1

    for i in range(len(data["Home Team"])):
        if not(type(data["Result"][i]) is str):
            continue
        if flip_result:
            data["Result"][i] = data["Result"][i][::-1]

    return data, indexToTeam, teamToIndex, indexToGamesPlayed


def createWeekMultiplier(reg=8):
    def f(week, round_number):
        return 1.0 / (week - round_number + reg) ** 2
    return f

def getMatrixForSeason(week, model, week_multiplier=None):

    data, indexToTeam, teamToIndex, indexToGamesPlayed = model

    # Calculate Games Played
    indexToGamesPlayed = [0 for i in range(len(indexToTeam))]
    for i in range(len(data["Home Team"])):
        if data["Round Number"][i] > week:
            break

        if not(type(data["Result"][i]) is str):
            continue

        result = data["Result"][i].split("-")
        if len(result) != 2:
            continue

        homeTeam = data["Home Team"][i]
        awayTeam = data["Away Team"][i]

        homeIndex = teamToIndex[homeTeam]
        awayIndex = teamToIndex[awayTeam]

        indexToGamesPlayed[homeIndex] += 1
        indexToGamesPlayed[awayIndex] += 1


    A = [[0.0 for i in range(len(indexToTeam))] for j in range(len(indexToTeam))]

    for i in range(len(data["Home Team"])):

        if data["Round Number"][i] > week:
            break

        homeTeam = data["Home Team"][i]
        awayTeam = data["Away Team"][i]

        if not(type(data["Result"][i]) is str):
            continue

        result = data["Result"][i].split("-")
        if len(result) != 2:
            continue

        homeScore = int(result[0].strip())
        awayScore = int(result[1].strip())

        homeIndex = teamToIndex[homeTeam]
        awayIndex = teamToIndex[awayTeam]

        homeTeamPlayed = indexToGamesPlayed[homeIndex]
        awayTeamPlayed = indexToGamesPlayed[awayIndex]

        # Adds in a weight to latest games
        #weight = 1.0 #/ (week - data["Round Number"][i] + 8) ** 2
        weight = 1.0
        if week_multiplier != None:
            weight = week_multiplier(week, data["Round Number"][i])

        if homeScore > awayScore:
            A[awayIndex][homeIndex] += 3.0 * weight # * (1/homeTeamPlayed)
        elif homeScore < awayScore:
            A[homeIndex][awayIndex] += 3.0 * weight # * (1/awayTeamPlayed)
        else:
            A[awayIndex][homeIndex] += 1.0 * weight # * (1/homeTeamPlayed)
            A[homeIndex][awayIndex] += 1.0 * weight # * (1/awayTeamPlayed)

        A[awayIndex][homeIndex] += (0.5) * homeScore # * (1/homeTeamPlayed)
        A[homeIndex][awayIndex] += (0.5) * awayScore # * (1/awayTeamPlayed)
    return A + np.identity(len(A)) + 0.1


# Get the rankings

def getValue(item):
    return item[1]

def printRankings(rankings, model):

    data, indexToTeam, teamToIndex, indexToGamesPlayed = model

    rankings = sorted(rankings, key=getValue, reverse=True)
    for i in range(len(rankings)):
        print((i+1), "%.3f " % rankings[i][1], indexToTeam[rankings[i][0]])


# Project on to the future games

def getSeasonStats(R, week, data, teamToIndex):

    indexToExpectedPoints = [ 0 for i in range(len(teamToIndex)) ]
    indexToRandomPoints   = [ 0 for i in range(len(teamToIndex)) ]
    indexToMorePoints     = [ 0 for i in range(len(teamToIndex)) ]

    for i in range(len(data["Home Team"])):

        homeTeam = data["Home Team"][i]
        awayTeam = data["Away Team"][i]

        homeIndex = teamToIndex[homeTeam]
        awayIndex = teamToIndex[awayTeam]


        # Not exact
        if data["Round Number"][i] > week or not(type(data["Result"][i]) is str) or len(data["Result"][i].split("-")) != 2:
            homeTeamValue = R[homeIndex][0]
            awayTeamValue = R[awayIndex][0]

            # Calulate More Points

            if homeTeamValue > awayTeamValue:
                indexToMorePoints[homeIndex] += 3
            elif homeTeamValue < awayTeamValue:
                indexToMorePoints[awayIndex] += 3
            else:
                indexToMorePoints[homeIndex] += 1
                indexToMorePoints[awayIndex] += 1

            # Calulate Random Points

            total = homeTeamValue + awayTeamValue
            if total == 0:
                homeTeamValue += 0.01
                awayTeamValue += 0.01
                total = homeTeamValue + awayTeamValue

            #if ((homeTeamValue / total) * (1-(homeTeamValue / total)))**2 > random.random():
            if random.random() > 0.72:
                indexToRandomPoints[homeIndex] += 1
                indexToRandomPoints[awayIndex] += 1
            elif homeTeamValue / total > random.random():
                indexToRandomPoints[homeIndex] += 3
            else:
                indexToRandomPoints[awayIndex] += 3

            # Calulate Expected Points
            # Maybe change 3 to like 2.2 or expected points per game

            hS = (homeTeamValue / total) ** 2
            aS = (awayTeamValue / total) ** 2

            hChance = hS / (hS + aS)
            aChance = aS / (hS + aS)

            indexToExpectedPoints[homeIndex] += 3 * hChance
            indexToExpectedPoints[awayIndex] += 3 * aChance

            continue


        result = data["Result"][i].split("-")

        # Calculate the known score

        homeScore = int(result[0].strip())
        awayScore = int(result[1].strip())

        if homeScore > awayScore:
            indexToExpectedPoints[homeIndex] += 3
            indexToRandomPoints[homeIndex] += 3
            indexToMorePoints[homeIndex] += 3
        elif awayScore > homeScore:
            indexToExpectedPoints[awayIndex] += 3
            indexToRandomPoints[awayIndex] += 3
            indexToMorePoints[awayIndex] += 3
        else:
            indexToExpectedPoints[homeIndex] += 1
            indexToRandomPoints[homeIndex] += 1
            indexToMorePoints[homeIndex] += 1
            indexToExpectedPoints[awayIndex] += 1
            indexToRandomPoints[awayIndex] += 1
            indexToMorePoints[awayIndex] += 1

    return indexToExpectedPoints, indexToRandomPoints,  indexToMorePoints



# Graph Stats

def getRankings(week, model, teamToIndex, week_multiplier=None):

    A = np.array(getMatrixForSeason(week, model, week_multiplier))
    R = pagerank.rank(A)

    rankings = [(i, R[i]) for i in range(len(indexToTeam))]

    indexToExpectedPoints, indexToRandomPoints, indexToMorePoints = getSeasonStats(R, week, model[0], teamToIndex)

    return indexToExpectedPoints

def plotSeason(maxWeek, model, week_multiplier=None):

    data, indexToTeam, teamToIndex, indexToGamesPlayed = model

    indexTeamToWeekToPoints = [[] for i in range(len(indexToTeam))]
    for i in range(maxWeek+1):
        indexToPoints = getRankings(i, model, teamToIndex, week_multiplier)
        for j in range(len(indexToTeam)):
            indexTeamToWeekToPoints[j].append(indexToPoints[j])

    import matplotlib.pyplot as plt

    rankings = [(i, indexTeamToWeekToPoints[i][-1]) for i in range(len(indexTeamToWeekToPoints))]
    rankings = sorted(rankings, key=getValue, reverse=True)
    for i in range(len(rankings)):
        j = rankings[i][0]
        plt.plot(indexTeamToWeekToPoints[j], label=indexToTeam[j])

    plt.ylabel('Expected Points')
    plt.xlabel('Week')
    plt.legend(bbox_to_anchor=(1.00, 1), loc="upper left")
    #plt.legend(bbox_to_anchor=(0, 0), loc="upper left")
    plt.show()



if __name__ == "__main__":

    csvFileName = "epl.csv"
    week = 40
    if len(sys.argv) > 1:
        csvFileName = sys.argv[1]
    if len(sys.argv) > 2:
        week = int(sys.argv[2])

    model = getData(csvFileName)
    data, indexToTeam, teamToIndex, indexToGamesPlayed = model

    # Get matrix A and Ranking
    A = getMatrixForSeason(week, model, None) #createWeekMultiplier(8))
    R = pagerank.rank(A)

    rankings = [(i, R[i]) for i in range(len(indexToTeam))]

    printRankings(rankings, model)

    indexToExpectedPoints, indexToRandomPoints, indexToMorePoints = getSeasonStats(R, week, data, teamToIndex)

    expectedPointsRanking = [(i, indexToExpectedPoints[i]) for i in range(len(indexToExpectedPoints))]
    randomPointsRanking = [(i, indexToRandomPoints[i]) for i in range(len(indexToRandomPoints))]
    morePointsRanking = [(i, indexToMorePoints[i]) for i in range(len(indexToMorePoints))]

    print("\nExpected Points")
    printRankings(expectedPointsRanking, model)

    print("\nWeighted Random Points")
    printRankings(randomPointsRanking, model)

    print("\nExact Points")
    printRankings(morePointsRanking, model)

    plotSeason(week, model)


