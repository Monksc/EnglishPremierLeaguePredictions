 ## 🚀 Overview

This project ingests historical match data and generates predictive insights for football matches, including:

- Win / Draw / Loss probabilities
- Team performance rankings
- Match outcome predictions across multiple leagues

The goal is to build a system that can evaluate team strength and produce actionable probability estimates based on real-world data.

---

## 🧠 Approach

The system combines multiple techniques:

- Statistical modeling (including ranking systems and probability estimation)
- Machine learning models (TensorFlow-based)
- Data-driven feature generation from historical match results
- Evaluation against test datasets to measure prediction accuracy

---

## 📊 Data Pipeline

- Collects match data across multiple leagues (EPL, Bundesliga, La Liga, Serie A)
- Processes and normalizes data into structured formats
- Generates training and test datasets
- Outputs predictions in JSON format for downstream use

---

## ⚙️ Features

- Multi-league support
- Automated data processing and model training
- Prediction output for match outcomes
- Error evaluation and model validation tools

---

## 📈 Example Output

- Probability of win/draw/loss for a given match
- Team strength rankings
- Model performance metrics on test data

---

## 🛠 Tech Stack

- Python
- TensorFlow
- Data processing pipelines
- JSON-based output for integration with applications

---

# Use Cases

---

## Page Rank
To do a page rank to see predictions for ranking of teams and predictions for end of season type in
```
python3 epl.py <csv file for schedule> <week>
```

---

## Predicting Games
Currently it doesnt predict score. You can do a page rank to see who is the better team and finally adding
    in what percent chance does a team to have to win.

---

## Neural Network
Uses several variations to page rank, wins, tie, loss, from home to away games to see
    the chance a team has to win. 
    You may have to change around some code in predictions_tensorflow.py.
```
python3 predictions_tensorflow.py
```

---

## Folder 'old_data'
This folder contains data from other leagues from previous years. The name format usually goes by
    "{name of the league}-{year season started}"

May need to run command below
```
cd old_data/jackpotdata/all/
tar -xf all.csv.tar.gx
```
---

## To Create the data
```
python3 make_data_like_my.py
```
---

## To Train Model
You may have to change around some code in predictions_tensorflow.py.
```
python3 predictions_tensorflow.py
```

