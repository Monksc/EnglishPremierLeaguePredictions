import numpy as np
import epl
import pagerank

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
            continue
        
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


def createWeekMultiplier(lastWeek):
    def f(week):
        return 1.0 / ((lastWeek - week) ** 2)
    return f

def createA(var, wins_multiplier=1.0, goals_multiplier=0.0, week_multiplier=None):

    def f(collected_data, week, canAddInput, homeIndex, awayIndex, homeScore, awayScore):

        if var not in collected_data:
            A = np.array([[0.0 for i in range(collected_data["team_count"])] for j in range(collected_data["team_count"])])
            collected_data[var] = A

        weight = 1.0
        if week_multiplier != None:
            weight = week_multiplier(week)

        if "totals" in collected_data:
            homeTeamPlayed = collected_data["totals"]["played"][homeIndex]
            awayTeamPlayed = collected_data["totals"]["played"][awayIndex]
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

        if homeScore > awayScore:
            collected_data[var][awayIndex][homeIndex] += wins_multiplier * (1/homeTeamPlayed) * 3.0 * weight
        elif awayScore < homeScore:
            collected_data[var][homeIndex][awayIndex] += wins_multiplier * (1/awayTeamPlayed) * 3.0 * weight
        else:
            collected_data[var][awayIndex][homeIndex] += wins_multiplier * (1/homeTeamPlayed) * 1.0 * weight
            collected_data[var][homeIndex][awayIndex] += wins_multiplier * (1/awayTeamPlayed) * 1.0 * weight
    
        collected_data[var][awayIndex][homeIndex] += goals_multiplier * (1/homeTeamPlayed) * homeScore
        collected_data[var][homeIndex][awayIndex] += goals_multiplier * (1/awayTeamPlayed) * awayScore

    return f

def addRankToInputs(var):

    def f(collected_data, week, canAddInput, homeIndex, awayIndex, homeScore, awayScore):

        if not(canAddInput):
            return

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
                    "played" : [0.0 for i in range(team_count)],
                    "wins"   : [0.0 for i in range(team_count)],
                    "tie"    : [0.0 for i in range(team_count)],
                    "loss"   : [0.0 for i in range(team_count)],
                    "goalsF" : [0.0 for i in range(team_count)],
                    "goalsA" : [0.0 for i in range(team_count)],
            }
            
        if homeScore != None:
            collected_data["totals"]["played"][homeIndex] += 1
            collected_data["totals"]["played"][awayIndex] += 1

            collected_data["totals"]["goalsF"][homeIndex] += homeScore
            collected_data["totals"]["goalsF"][awayIndex] += awayScore

            collected_data["totals"]["goalsA"][homeIndex] += awayScore
            collected_data["totals"]["goalsA"][awayIndex] += homeScore

            if homeScore > awayScore:
                collected_data["totals"]["wins"][homeIndex] += 1
                collected_data["totals"]["loss"][awayIndex] += 1
            elif homeScore < awayScore:
                collected_data["totals"]["wins"][awayIndex] += 1
                collected_data["totals"]["loss"][homeIndex] += 1
            else:
                collected_data["totals"]["tie"][homeIndex] += 1
                collected_data["totals"]["tie"][awayIndex] += 1

        if not(canAddInput):
            return


        homeTotal = collected_data["totals"]["played"][homeIndex]
        awayTotal = collected_data["totals"]["played"][awayIndex]

        collected_data["inputs"][-1].append(collected_data["totals"]["wins"][homeIndex] / homeTotal)
        collected_data["inputs"][-1].append(collected_data["totals"]["wins"][awayIndex] / awayTotal)

        collected_data["inputs"][-1].append(collected_data["totals"]["loss"][homeIndex] / homeTotal)
        collected_data["inputs"][-1].append(collected_data["totals"]["loss"][awayIndex] / awayTotal)

        collected_data["inputs"][-1].append(collected_data["totals"]["tie"][homeIndex] / homeTotal)
        collected_data["inputs"][-1].append(collected_data["totals"]["tie"][awayIndex] / awayTotal)

        collected_data["inputs"][-1].append(collected_data["totals"]["goalsF"][homeIndex] / homeTotal)
        collected_data["inputs"][-1].append(collected_data["totals"]["goalsF"][awayIndex] / awayTotal)

        collected_data["inputs"][-1].append(collected_data["totals"]["goalsA"][homeIndex] / homeTotal)
        collected_data["inputs"][-1].append(collected_data["totals"]["goalsA"][awayIndex] / awayTotal)

    return f


def getAllMyData(filenames=[
            "old_data/epl-2015.csv",
            "old_data/epl-2016.csv",
            "old_data/epl-2017.csv",
            "old_data/epl-2018.csv",
            "old_data/epl-2019.csv",
            "epl-2020-week-0.csv",
            ], getAllInputs=False, starting_week=10):

    total_inputs  = []
    total_outputs = []

    for filename in filenames:

        bothFuncs = [
            createA("A",        1.0, 0.0, None),
            createA("A-week",   1.0, 0.0, createWeekMultiplier(42)),
            createA("A-G",      0.0, 1.0, None),
            createA("A-G-week", 0.0, 1.0, createWeekMultiplier(42)),
            addNewInputAndOutput,
            addRankToInputs("A"),
            addRankToInputs("A-week"),
            addRankToInputs("A-G"),
            addRankToInputs("A-G-week"),
            addAverageToInputs(),
        ]

        repeatFuncs = [
            createA("A",        1.0, 0.0, None),
            createA("A-week",   1.0, 0.0, createWeekMultiplier(42)),
            createA("A-G",      0.0, 1.0, None),
            createA("A-G-week", 0.0, 1.0, createWeekMultiplier(42)),
            addNewInputAndOutput,
            addRankToInputs("A"),
            addRankToInputs("A-week"),
            addRankToInputs("A-G"),
            addRankToInputs("A-G-week"),
        ]

        model = epl.getData(filename, False)
        data = loopThroughData(model, bothFuncs, not(getAllInputs), starting_week)

        inputs  = data["inputs"]
        outputs = data["outputs"]

        model = epl.getData(filename, True)
        data = loopThroughData(model, bothFuncs, not(getAllInputs), starting_week)

        inps = data["inputs"]

        assert(len(inputs) == len(inps))
        for i in range(len(inputs)):
            inputs[i] += inps[i]

        total_inputs += inputs
        total_outputs += outputs

    return np.array(total_inputs), np.array(total_outputs)
        

