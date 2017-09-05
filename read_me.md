# Load Forecasting using Sarimax models

## Introduction
Distributed Energy Resources such as solar and storage are disrupting our traditional
energy supply and to ensure that they are being used to their full potential,
energy demand forecasting is critical. Knowledge of future energy demand allows
'smart' energy system to make the correct dispatch strategy decision in the present moment,
offering the most value to the customers.

## Data
I am working with 5 minute energy demand data (voltage and current for all three phases) from
an energy service provider currently operating in Kenya and Ghana. This project also uses
NASA meteorological data from the MERRA2 dataset.

## Approach
In the study, I developed a pipeline that:
- cleans load data (involves manual examination of the data)
- creates and adds relevant features
- evaluates the performance of a range of forecasting models on one-step-ahead forecasting
-- Baseline 1: Forecast previous energy demand
-- Baseline 2: Forecast average energy demand for that period in Time
-- Sarima : Forecast based of a tuned Sarima model
-- Sarimax : Forecast based on a tuned SarimaX with fixed beta coefficients for the linear regression
-- Sarimax : Forecast based on a tuned SarimaX with variable beta coefficients for the linear regression
