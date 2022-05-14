import numpy as np
import epl
import pagerank
import pandas as pd
import csv_file_names as csv_names

def loopThroughData(model, funcs, only_result=True, starting_week=10):

    data, indexToTeam, teamToIndex, indexToGamesPlayed = model

    collected_data = {
        "team_count" : len(indexToTeam),
        "inputs" : [],
        "outputs" : [],
    }

    for i in range(len(data["Home Team"])):

        result = data["Result"][i]
        if not(type(data["Result"][i]) is str):
            result = ""

        homeTeam = data["Home Team"][i]
        awayTeam = data["Away Team"][i]

        homeIndex = teamToIndex[homeTeam]
        awayIndex = teamToIndex[awayTeam]

        result = result.split("-")
        if len(result) == 2:
            homeScore = int(result[0].strip())
            awayScore = int(result[1].strip())
        elif only_result:
            continue
        else:
            homeScore = None
            awayScore = None


        week = data["Round Number"][i]
        for f in funcs:
            f(collected_data, week, week>=starting_week, homeIndex, awayIndex, homeScore, awayScore)

    return collected_data


def createWeekMultiplier(multiplier):
    def f(current_score):
        #return 1.0 / ((lastWeek - week) ** 2)
        return current_score * multiplier
    return f

def createA(var, wins_multiplier=1.0, goals_multiplier=0.0, week_multiplier=None):

    def f(collected_data, week, canAddInput, homeIndex, awayIndex, homeScore, awayScore):

        if var not in collected_data:
            A = np.array([[0.0 for i in range(collected_data["team_count"])] for j in range(collected_data["team_count"])])
            collected_data[var] = A

        if "totals" in collected_data:
            homeTeamPlayed = collected_data["totals"]["Hplayed"][homeIndex] + collected_data["totals"]["Aplayed"][homeIndex]
            awayTeamPlayed = collected_data["totals"]["Hplayed"][awayIndex] + collected_data["totals"]["Aplayed"][awayIndex]
            if homeTeamPlayed == 0:
                homeTeamPlayed = 1
            if awayTeamPlayed == 0:
                awayTeamPlayed = 1
        else:
            homeTeamPlayed = 1.0
            awayTeamPlayed = 1.0
    
        # This is when we are predicting games we havent seen yet
        if homeScore == None:
            return

        
        # Move this above the if homeScore == None for a more acurracte prediction of future games
        # Like it is now will say if the teams were to play each other now in current form
        if week_multiplier != None:
            collected_data[var][awayIndex][homeIndex] = week_multiplier(collected_data[var][awayIndex][homeIndex])

        if homeScore > awayScore:
            collected_data[var][awayIndex][homeIndex] += wins_multiplier * 3.0 # * (1/homeTeamPlayed)
        elif homeScore < awayScore:
            collected_data[var][homeIndex][awayIndex] += wins_multiplier * 3.0 # * (1/awayTeamPlayed)
        else:
            collected_data[var][awayIndex][homeIndex] += wins_multiplier * 1.0 # * (1/homeTeamPlayed)
            collected_data[var][homeIndex][awayIndex] += wins_multiplier * 1.0 # * (1/awayTeamPlayed)
    
        collected_data[var][awayIndex][homeIndex] += goals_multiplier * homeScore # * (1/homeTeamPlayed)
        collected_data[var][homeIndex][awayIndex] += goals_multiplier * awayScore # * (1/awayTeamPlayed)

    return f

def addRankToInputs(var):

    def f(collected_data, week, canAddInput, homeIndex, awayIndex, homeScore, awayScore):

        if not(canAddInput):
            return

        # Also happens in createA
        # Could happen here depends on when canAddInput
        if var not in collected_data:
            R = [[1.0/collected_data['team_count']] for i in range(collected_data['team_count'])]
        else:
            A = collected_data[var]
            R = pagerank.rank(A)

        collected_data["inputs"][-1].append(R[homeIndex][0])
        collected_data["inputs"][-1].append(R[awayIndex][0])

    return f

def addNewInputAndOutput(collected_data, week, canAddInput, homeIndex, awayIndex, homeScore, awayScore):
    if canAddInput:
        collected_data["inputs"].append([])

        result = [0.0, 0.0, 0.0]

        # This is when we are predicting games we havent seen yet
        if homeScore != None:
            if homeScore > awayScore:
                result[0] = 1.0
            elif homeScore < awayScore:
                result[1] = 1.0
            else:
                result[2] = 1.0

        collected_data["outputs"].append(result)

def addAverageToInputs():

    def f(collected_data, week, canAddInput, homeIndex, awayIndex, homeScore, awayScore):

        if "totals" not in collected_data:

            team_count = collected_data["team_count"]

            collected_data["totals"] = {
                    "Hplayed"  : [0.0 for i in range(team_count)],
                    "Hwins"   : [0.0 for i in range(team_count)],
                    "Htie"    : [0.0 for i in range(team_count)],
                    "Hloss"   : [0.0 for i in range(team_count)],
                    "HgoalsF" : [0.0 for i in range(team_count)],
                    "HgoalsA" : [0.0 for i in range(team_count)],
                    "Aplayed"  : [0.0 for i in range(team_count)],
                    "Awins"   : [0.0 for i in range(team_count)],
                    "Atie"    : [0.0 for i in range(team_count)],
                    "Aloss"   : [0.0 for i in range(team_count)],
                    "AgoalsF" : [0.0 for i in range(team_count)],
                    "AgoalsA" : [0.0 for i in range(team_count)],
            }
            

        if canAddInput:
        
            # Home

            homeTotal = collected_data["totals"]["Hplayed"][homeIndex]
            awayTotal = collected_data["totals"]["Hplayed"][awayIndex]
            if homeTotal == 0.0:
                homeTotal = 1.0
            if awayTotal == 0.0:
                awayTotal = 1.0

            collected_data["inputs"][-1].append(collected_data["totals"]["Hwins"][homeIndex] / homeTotal)
            collected_data["inputs"][-1].append(collected_data["totals"]["Hwins"][awayIndex] / awayTotal)

            collected_data["inputs"][-1].append(collected_data["totals"]["Hloss"][homeIndex] / homeTotal)
            collected_data["inputs"][-1].append(collected_data["totals"]["Hloss"][awayIndex] / awayTotal)

            collected_data["inputs"][-1].append(collected_data["totals"]["Htie"][homeIndex] / homeTotal)
            collected_data["inputs"][-1].append(collected_data["totals"]["Htie"][awayIndex] / awayTotal)

            collected_data["inputs"][-1].append(collected_data["totals"]["HgoalsF"][homeIndex] / homeTotal)
            collected_data["inputs"][-1].append(collected_data["totals"]["HgoalsF"][awayIndex] / awayTotal)

            collected_data["inputs"][-1].append(collected_data["totals"]["HgoalsA"][homeIndex] / homeTotal)
            collected_data["inputs"][-1].append(collected_data["totals"]["HgoalsA"][awayIndex] / awayTotal)


            # Away

            homeTotal = collected_data["totals"]["Aplayed"][homeIndex]
            awayTotal = collected_data["totals"]["Aplayed"][awayIndex]
            if homeTotal == 0.0:
                homeTotal = 1.0
            if awayTotal == 0.0:
                awayTotal = 1.0

            collected_data["inputs"][-1].append(collected_data["totals"]["Awins"][homeIndex] / homeTotal)
            collected_data["inputs"][-1].append(collected_data["totals"]["Awins"][awayIndex] / awayTotal)

            collected_data["inputs"][-1].append(collected_data["totals"]["Aloss"][homeIndex] / homeTotal)
            collected_data["inputs"][-1].append(collected_data["totals"]["Aloss"][awayIndex] / awayTotal)

            collected_data["inputs"][-1].append(collected_data["totals"]["Atie"][homeIndex] / homeTotal)
            collected_data["inputs"][-1].append(collected_data["totals"]["Atie"][awayIndex] / awayTotal)

            collected_data["inputs"][-1].append(collected_data["totals"]["AgoalsF"][homeIndex] / homeTotal)
            collected_data["inputs"][-1].append(collected_data["totals"]["AgoalsF"][awayIndex] / awayTotal)

            collected_data["inputs"][-1].append(collected_data["totals"]["AgoalsA"][homeIndex] / homeTotal)
            collected_data["inputs"][-1].append(collected_data["totals"]["AgoalsA"][awayIndex] / awayTotal)



        # Inputting New Scores

        if homeScore != None:
            collected_data["totals"]["Hplayed"][homeIndex] += 1
            collected_data["totals"]["Aplayed"][awayIndex] += 1

            collected_data["totals"]["HgoalsF"][homeIndex] += homeScore
            collected_data["totals"]["AgoalsF"][awayIndex] += awayScore

            collected_data["totals"]["HgoalsA"][homeIndex] += awayScore
            collected_data["totals"]["AgoalsA"][awayIndex] += homeScore

            if homeScore > awayScore:
                collected_data["totals"]["Hwins"][homeIndex] += 1
                collected_data["totals"]["Aloss"][awayIndex] += 1
            elif homeScore < awayScore:
                collected_data["totals"]["Awins"][awayIndex] += 1
                collected_data["totals"]["Hloss"][homeIndex] += 1
            else:
                collected_data["totals"]["Htie"][homeIndex] += 1
                collected_data["totals"]["Atie"][awayIndex] += 1


    return f


def getAllMyData(filenames=[
            #"old_data/american-football.csv/nfl-2020.csv",
            #"old_data/american-football.csv/nfl-2019.csv",
            #"old_data/american-football.csv/nfl-2018.csv",
            #"old_data/american-football.csv/nfl-2017.csv",
            #"old_data/american-football.csv/nfl-2016.csv",
            "old_data/bundesliga-2018.csv",
            "old_data/bundesliga-2019.csv",
            "old_data/bundesliga-2020.csv",
            "old_data/efl-championship-2019.csv",
            "old_data/efl-championship-2020.csv",
            "old_data/la-liga-2018.csv",
            "old_data/la-liga-2019.csv",
            #"old_data/la-liga-2020.csv",
            "old_data/ligue-1-2018.csv",
            "old_data/ligue-1-2019.csv",
            "old_data/ligue-1-2020.csv",
            "old_data/serie-a-2017.csv",
            "old_data/serie-a-2018.csv",
            "old_data/serie-a-2019.csv",
            "old_data/serie-a-2020.csv",
            "old_data/turkey-super-lig-2019.csv",
            "old_data/turkey-super-lig-2020.csv",
            "old_data/epl-2009.csv",
            "old_data/epl-2010.csv",
            "old_data/epl-2011.csv",
            "old_data/epl-2012.csv",
            "old_data/epl-2013.csv",
            "old_data/epl-2014.csv",
            "old_data/epl-2015.csv",
            "old_data/epl-2016.csv",
            "old_data/epl-2017.csv",
            "old_data/epl-2018.csv",
            "old_data/epl-2019.csv",
            "old_data/epl-2020.csv",
            ], getAllInputs=False, starting_week=10):

    total_inputs  = []
    total_outputs = []

    for filename in filenames:

        print("READING ", filename)

        bothFuncs = [
            addNewInputAndOutput,
            addRankToInputs("A"),
            addRankToInputs("A-week"),
            addRankToInputs("A-G"),
            addRankToInputs("A-G-week"),
            createA("A",        1.0, 0.0, None),
            createA("A-week",   1.0, 0.0, createWeekMultiplier(0.5)),
            createA("A-G",      0.0, 1.0, None),
            createA("A-G-week", 0.0, 1.0, createWeekMultiplier(0.5)),
            addAverageToInputs(),
        ]

        repeatFuncs = [
            addNewInputAndOutput,
            addRankToInputs("A"),
            addRankToInputs("A-week"),
            addRankToInputs("A-G"),
            addRankToInputs("A-G-week"),
            createA("A",        1.0, 0.0, None),
            createA("A-week",   1.0, 0.0, createWeekMultiplier(0.5)),
            createA("A-G",      0.0, 1.0, None),
            createA("A-G-week", 0.0, 1.0, createWeekMultiplier(0.5)),
        ]

        model = epl.getData(filename, False)
        data = loopThroughData(model, bothFuncs, not(getAllInputs), starting_week)

        inputs  = data["inputs"]
        outputs = data["outputs"]

        model = epl.getData(filename, True)
        data = loopThroughData(model, repeatFuncs, not(getAllInputs), starting_week)

        inps = data["inputs"]

        assert(len(inputs) == len(inps))
        for i in range(len(inputs)):
            inputs[i] += inps[i]

        total_inputs += inputs
        total_outputs += outputs

    return np.array(total_inputs), np.array(total_outputs)
        



if __name__ == "__main__":

    csv_files = csv_names.AllSeasonsFootballCSV
    outputFileName = "old_data/jackpotdata/all/all.csv"

    outputDF = pd.DataFrame(columns=[
        "A-Rank-Home", "A-Rank-Away",
        "A-week-Rank-Home", "A-week-Rank-Away",
        "A-Goals-Rank-Home", "A-Goals-Rank-Away",
        "A-Goals-week-Rank-Home", "A-Goals-week-Rank-Away",
        "HomeTeam-Home-wins", "AwayTeam-Home-wins",
        "HomeTeam-Home-loss", "AwayTeam-Home-loss",
        "HomeTeam-Home-tie", "AwayTeam-Home-tie",
        "HomeTeam-Home-GoalsF", "AwayTeam-Home-GoalsF",
        "HomeTeam-Home-GoalsA", "AwayTeam-Home-GoalsA",
        "HomeTeam-Away-wins", "AwayTeam-Away-wins",
        "HomeTeam-Away-loss", "AwayTeam-Away-loss",
        "HomeTeam-Away-tie", "AwayTeam-Away-tie",
        "HomeTeam-Away-GoalsF", "AwayTeam-Away-GoalsF",
        "HomeTeam-Away-GoalsA", "AwayTeam-Away-GoalsA",
        "A-flip-Rank-Home", "A-flip-Rank-Away",
        "A-flip-week-Rank-Home", "A-flip-week-Rank-Away",
        "A-flip-Goals-Rank-Home", "A-flip-Goals-Rank-Away",
        "A-flip-Goals-week-Rank-Home", "A-flip-Goals-week-Rank-Away",
        "HWin", "Tie", "AWin",
    ])

    count = 0
    for i in range(len(csv_files)):
        inputs, outputs = getAllMyData(filenames=[csv_files[i]])

        for j in range(len(inputs)):
            row = inputs[j].tolist() + outputs[j].tolist()

            outputDF.loc[count] = row
            count += 1

    outputDF.to_csv(outputFileName)

