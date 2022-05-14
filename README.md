# EnglishPremierLeaguePredictions
Predicts Matches for the English Premier League. Can be changed for other leagues.

# Page Rank
To do a page rank to see predictions for ranking of teams and predictions for end of season type in
```
python3 epl.py <csv file for schedule> <week>
```

# Predicting Games
Currently it doesnt predict score. You can do a page rank to see who is the better team and finally adding
    in what percent chance does a team to have to win.


# Neural Network
Uses several variations to page rank, wins, tie, loss, from home to away games to see
    the chance a team has to win. 
    You may have to change around some code in predictions_tensorflow.py.
```
python3 predictions_tensorflow.py
```

# Folder 'old_data'
This folder contains data from other leagues from previous years. The name format usually goes by
    "{name of the league}-{year season started}"

May need to run command below
```
cd old_data/jackpotdata/all/
tar -xf all.csv.tar.gx
```

# To Create the data
```
python3 make_data_like_my.py
```

# To Train Model
You may have to change around some code in predictions_tensorflow.py.
```
python3 predictions_tensorflow.py
```

