PPG DaLiA Heart Rate Estimation Project
==============================

Estimate heartrate using PPG Sensor measurements. Dataset : https://archive.ics.uci.edu/ml/datasets/PPG-DaLiA

Considered Features
------------
- PPG signal channel
- Accelerometer x, y and z channels
- Age
- Gender
- Physical fitness level (sport)
- BMI (calculated)

Target 
------------
- Heart rate (bpm)

Training Process
------------
- CNN LSTM model
- LOSO validation for each subject

Model Performance
------------

|Subject| MAE |
| ----- | --- |
| 1 | 9.992 |
| 2 | 7.796 |
| 3 | 12.445 |
| 4 | 9.892 |
| 5 | 43.913 |
| 6 | 25.698 |
| 7 | 12.398 |
| 8 | 14.125 |
| 9 | 10.165 |
| 10 | 11.020|
| 11 | 22.729 |
| 12 | 12.894 |
| 13 | 12.987 |
| 14 | 12.090 |
| 15 | 13.507 |
| Average | 15.443 |

The model does not use any FFT of the input signals and considers only the raw signals

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
