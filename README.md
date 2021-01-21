# EnglishPremierLeaguePredictions
Predicts Matches for the English Premier League. Can be changed for other leagues.

# Page Rank
To do a page rank to see predictions for ranking of teams and predictions for end of season type in
```
python3 eply.py <csv file for schedule> <week>
```

# Predicting Games
Currently it doesnt predict score. You can do a page rank to see who is the better team and finally adding
    in what percent chance does a team to have to win.


# Neural Network
Uses several variations to page rank, wins, tie, loss, from home to away games to see
    the chance a team has to win. Currently the model is extremly overfitted just to see
    my limits for how big the model can be. I also need more data on past seasons. Only
    have seasons 2015-2016+
```
python3 predictions_tensorflow.py
```
