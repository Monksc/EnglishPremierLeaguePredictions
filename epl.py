import numpy as np
import pandas as pd
import random
import pagerank
import sys


def getData(csvFileName):
    
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

    return data, indexToTeam, teamToIndex, indexToGamesPlayed


def getMatrixForSeason(week, model):
    
    data, indexToTeam, teamToIndex, indexToGamesPlayed = model

    A = [[0.0 for i in range(len(indexToTeam))] for j in range(len(indexToTeam))]
    
    for i in range(len(data["Home Team"])):

        if data["Round Number"][i] > week:
            continue
    
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
    
        if homeScore > awayScore:
            A[awayIndex][homeIndex] += (1/homeTeamPlayed) * 3.0
        elif awayScore < homeScore:
            A[homeIndex][awayIndex] += (1/awayTeamPlayed) * 3.0
        else:
            A[awayIndex][homeIndex] += (1/homeTeamPlayed) * 1.0
            A[homeIndex][awayIndex] += (1/awayTeamPlayed) * 1.0
    
        #A[awayIndex][homeIndex] += (0.2) * (1/homeTeamPlayed) * homeScore
        #A[homeIndex][awayIndex] += (0.2) * (1/awayTeamPlayed) * awayScore
    return A


# Get the rankings

def getValue(item):
    return item[1]

def printRankings(rankings, model):

    data, indexToTeam, teamToIndex, indexToGamesPlayed = model

    rankings = sorted(rankings, key=getValue, reverse=True)
    for i in range(len(rankings)):
        print((i+1), "%.3f " % rankings[i][1], indexToTeam[rankings[i][0]])


# Project on to the future games

def getSeasonStats(R, week):

    indexToExpectedPoints = [ 0 for i in range(len(indexToTeam))]
    indexToRandomPoints   = [ 0 for i in range(len(indexToTeam))]
    indexToMorePoints     = [ 0 for i in range(len(indexToTeam))]
    
    for i in range(len(data["Home Team"])):
    
        homeTeam = data["Home Team"][i]
        awayTeam = data["Away Team"][i]
    
        homeIndex = teamToIndex[homeTeam]
        awayIndex = teamToIndex[awayTeam]
    
    
        # Not exact
        if data["Round Number"][i] > week or not(type(data["Result"][i]) is str) or len(data["Result"][i].split("-")) != 2:
            homeTeamValue = R[homeIndex]
            awayTeamValue = R[awayIndex]
            
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
            if ((homeTeamValue / total) * (1-(homeTeamValue / total)))**2 > random.random():
                indexToRandomPoints[homeIndex] += 1
                indexToRandomPoints[awayIndex] += 1
            if homeTeamValue / total > random.random():
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

def getRankings(week, model):

    A = np.array(getMatrixForSeason(week, model))
    R = pagerank.rank(A)
    
    rankings = [(i, R[i]) for i in range(len(indexToTeam))]
    
    indexToExpectedPoints, indexToRandomPoints, indexToMorePoints = getSeasonStats(R, week)

    return indexToExpectedPoints

def plotSeason(maxWeek, model):

    data, indexToTeam, teamToIndex, indexToGamesPlayed = model
    
    indexTeamToWeekToPoints = [[] for i in range(len(indexToTeam))]
    for i in range(maxWeek+1):
        indexToPoints = getRankings(i, model)
        for j in range(len(indexToTeam)):
            indexTeamToWeekToPoints[j].append(indexToPoints[j])

    import matplotlib.pyplot as plt

    for j in range(len(indexToTeam)):
        plt.plot(indexTeamToWeekToPoints[j], label=indexToTeam[j])
    plt.ylabel('Expected Points')
    plt.legend(bbox_to_anchor=(1.00, 1), loc="upper left")
    #plt.legend(bbox_to_anchor=(0, 0), loc="upper left")
    plt.show()


def createScore(maxScore):
    def score(lastWeek, currentWeek):
    
        diff = 0.0
        for i in range(len(lastWeek)):
            diff += (lastWeek[i] - currentWeek[i]) ** 2
    
        return diff / len(lastWeek)
    def f(lastWeek, currentWeek):
        return score(lastWeek, currentWeek) <= maxScore
    return f

def createDiff(maxDiff):
    def diff(lastWeek, currentWeek):
    
        diff = 0.0
        for i in range(len(lastWeek)):
            diff += abs(lastWeek[i] - currentWeek[i])
    
        return diff / len(lastWeek)
    def f(lastWeek, currentWeek):
        return diff(lastWeek, currentWeek) <= maxDiff
    return f

def createGoodEnough(amountOfTeams, changePosition):
    def goodEnough(lastWeek, currentWeek):
    
        bad = 0
        for i in range(len(lastWeek)):
            if abs(lastWeek[i] - currentWeek[i]) > changePosition:
                bad += 1
    
        return bad < amountOfTeams
    return goodEnough

def whatWeek(isGood, model):

    data, indexToTeam, teamToIndex, indexToGamesPlayed = model

    def getPlacement(week):
        rankings = getRankings(week)
        rankings = [(i, rankings[i]) for i in range(len(indexToTeam))]
        rankings = sorted(rankings, key=getValue, reverse=True)
        newValues = [rankings[i][0] for i in range(len(rankings))]
        return newValues


    i = 38
    lastWeek = getPlacement(i)
    while i > 1:

        weekI = getPlacement(i)

        if not(isGood(lastWeek, weekI)):
            return i
        i -= 1


    return 0
        

if __name__ == "__main__":

    csvFileName = "epl-2020-week-0.csv"
    week = 40
    if len(sys.argv) > 1:
        csvFileName = sys.argv[1]
    if len(sys.argv) > 2:
        week = int(sys.argv[2])
    
    model = getData(csvFileName)
    data, indexToTeam, teamToIndex, indexToGamesPlayed = model


    # Get matrix A and Ranking
    A = getMatrixForSeason(week, model)
    A = np.array(A)
    R = pagerank.rank(A)
    
    rankings = [(i, R[i]) for i in range(len(indexToTeam))]
    
    printRankings(rankings, model)
    
    indexToExpectedPoints, indexToRandomPoints, indexToMorePoints = getSeasonStats(R, week)
    
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
    
    print(whatWeek(createGoodEnough(5, 5), model))
    print(whatWeek(createScore(9**2), model))
    print(whatWeek(createDiff(9), model))


