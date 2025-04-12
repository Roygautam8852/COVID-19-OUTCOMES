# COVID-19-OUTCOMES
# COVID-19 EDA

## Overview
This **Exploratory Data Analysis (EDA)** on a COVID-19 dataset investigates cases, hospitalizations, and deaths by vaccination status, age, and time. Built with **Python**, it showcases data cleaning, modeling, and visualizations to reveal public health trends.

## Features
- **Data Cleaning**: Kept age groups like `0-4`, fixed typos (`'05-Nov` to `5-11`), and handled missing values (~2450 rows, 22 columns).
- **Weekly Trends**: Line charts highlight 2022 rate peaks for unvaccinated, vaccinated, and boosted groups.
- **Age Analysis**: Bar charts show severe outcomes in older groups (65+) and cases in younger ones (`0-4`).
- **Vaccination Impact**: Log-transformed regression models vaccination effects, colored by age.
- **Outcomes**: Pie and bar plots emphasize case dominance.
- **Correlations**: Fixed heatmap shows complete rate-outcome links.

## Technologies
- Python: Pandas, NumPy, Matplotlib, Seaborn, Plotly, Scikit-learn, SciPy
- Jupyter Notebook / Google Colab

## Files
- `eda_covid19.ipynb`: Code
- `covid-19_dataset.csv`: Input
- `cleaned_covid-19_dataset.csv`: Output
- `vaccination_impact_regression.png`: Regression
- `correlation_heatmap.png`: Heatmap

## Setup
1. Clone:
   ```bash
   git clone https://github.com/Roygautam8252/covid19-eda.git
