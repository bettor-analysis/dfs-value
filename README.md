# DFS Value Finder Collection

This repo is a set of R scripts to help spot undervalued players in Daily Fantasy Sports (DFS) across different sports. It uses regression models to predict salaries based on Rotowire projections (like fantasy points), then grades players on their "value" delta. Great for lineup building on sites like DraftKings or FanDuel.

## What It Does
- Loads player data from CSV files (e.g., salaries, projections from Rotowire).
- Builds and compares models: linear, log-linear, polynomial, random forest, XGBoost.
- Evaluates them with metrics like RMSE, MAE, MAPE, and R².
- Grades players: A (great value), B (good), C (moderate), X (fade) based on how under/overpriced they are.
- Includes extras like visuals, grade counts, and player lookups.


## Requirements
- R and packages: `readr`, `dplyr`, `ggplot2`, `Metrics`, `randomForest`, `xgboost`, `tibble`. Install with `install.packages(c(...))`.
- Data: Grab Rotowire projections and DFS salaries, save as CSV in `/data` (e.g., columns: PLAYER, SAL, FPTS).

## Scripts 

### MLB (mlb-value-finder.R)
Analyzes MLB players. Predicts salary from FPTS, grades based on deltas (e.g., ≥600 = A).

How to run:
1. Add your data to `data/rw-mlb-player-pool.csv`.
2. Run the script.
3. Swap models in the grading section or search for players like "Aaron Civale".

Sample output: Model comparison table, graded players, etc.

### All Sports (dfsvalue_app.R)
Interactive Shiny app that works with any sport’s DFS slate. Upload a CSV of player name, projected points (FPTS) and salary (SAL), train multiple regression models, compare their performance, and generate value-graded player tables. Includes real-time filtering by player, salary range, and grade.


## Tips
- Customize: Change grade cutoffs or pick your fave model (XGBoost is solid default).
- Data sources: Rotowire for projections, DFS sites for salaries—update daily for slates.
- Ideas: Combine with optimizers for full lineups, or automate data pulls if you expand.
