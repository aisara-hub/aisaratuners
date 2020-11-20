# **AiSara Hyperparameter Tuning API And The Number of Calls Need for Optimization Run**

**aisaratuners.aisara_keras_tuner module utilizes [AiSara Hyperparameter Tuning API](https://rapidapi.com/aisara-technology-aisara-technology-default/api/aisara-hyperparameter-tuning) for its optimizer where the user just need to call **run_opti()** function after initiating HpOptimization class and automatically several API calls for the different endpoints will be performed.**

The number of [AiSara Hyperparameter Tuning](https://rapidapi.com/aisara-technology-aisara-technology-default/api/aisara-hyperparameter-tuning) API calls needed for one round in the optimization process can be summarized as follows:

| API endpoint |      number of calls per round     |  remark |
|:----------:|:-------------:|:------:|
| General Prediction |  1 | check API subscription |
| LHC |    1   |   for hps sampling  |
| Fit | depends on number of trials  | calculate the error of each trial |
| Predict | depends on number of trials  | |  calculate the error of each trial |
| General Prediction | 1  |  for maximum error calculation |
| General Prediction | 1  |  for objective function calculation |

Additional 3 [AiSara Hyperparameter Tuning](https://rapidapi.com/aisara-technology-aisara-technology-default/api/aisara-hyperparameter-tuning) API calls are needed when **plot_search_space()** function is called, these calls can be summarized as follows:

| API endpoint |      number of calls for 3d surface plotting    |
|:----------:|:-------------:|
| General Grid Aisara  |  1 | 
| Fit  |  1 |
| Predict  |  1 |

To calculate how many [AiSara Hyperparameter Tuning](https://rapidapi.com/aisara-technology-aisara-technology-default/api/aisara-hyperparameter-tuning) API calls needed for **one optimization run** in aisara_keras_tuner model you can use the following formulas:

* run_opti()

`api_calls = 2+(3+𝟐*number_trials)*(number_rounds-1)+2*(number_rounds-2)`

* run_opti() + plot_search_space()

`api_calls =5+(3+𝟐*number_trials)*(number_rounds-1)+2*(number_rounds-2)`


