import pandas as pd

def makeCSVs(oldCsvFileName, newCsvFileName):

    year = {}
    data = pd.read_csv(oldCsvFileName)
    outputPD = {}
    teams_seen = {}
    round_number = {}

    for i in range(len(data.Season)):

        tier = data['tier'][i]

        if not(tier in year):
            year[tier] = None

        # new year
        if year[tier] != data.Season[i]:

            if year[tier] != None:
                outputPD[tier].to_csv(newCsvFileName + '-' + str(year[tier]) + '-' + str(tier) + '.csv')
                print(year[tier], tier)


            outputPD[tier] = pd.DataFrame(columns=[
                'Round Number', 'Date',
                'Home Team', 'Away Team', 'Result'])

            round_number[tier] = 0
            teams_seen[tier] = set()
            year[tier] = data.Season[i]

        date = data['Date'][i]
        home = data['home'][i]
        away = data['visitor'][i]
        hgoals = str(data['hgoal'][i])
        vgoals = str(data['vgoal'][i])
        
        outputPD[tier].loc[i] = [round_number[tier], date, home, away, hgoals + ' - ' + vgoals]

        if home in teams_seen[tier]:
            round_number[tier] += 1
            teams_seen[tier] = set()

        teams_seen[tier].add(home)
        teams_seen[tier].add(away)
        

if __name__ == "__main__":
    makeCSVs('old_data/jackpot/data-raw/england.csv',  'old_data/jackpotdata/epl/epl')
    makeCSVs('old_data/jackpot/data-raw/belgium.csv',  'old_data/jackpotdata/belgium/belgium')
    makeCSVs('old_data/jackpot/data-raw/france.csv',   'old_data/jackpotdata/france/france')
    makeCSVs('old_data/jackpot/data-raw/germany.csv',  'old_data/jackpotdata/germany/germany')
    makeCSVs('old_data/jackpot/data-raw/germany2.csv', 'old_data/jackpotdata/germany2/germany2')




