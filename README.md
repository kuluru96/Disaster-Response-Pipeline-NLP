# Disaster Response Pipeline


## Motivation

This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset contains pre-labelled tweet and messages from real-life disaster events. The project aim is to build a Natural Language Processing (NLP) model to categorize messages on a real time basis.

This project is divided in the following key sections:

1. Processing data, building an ETL pipeline to extract data from source
2. Build a machine learning pipeline to train the which can classify text message in various categories
3. Run a web app which can show model results in real time


## Libraries and Packages
* Python 
* NumPy, SciPy, Pandas, Sciki-Learn
* NLTK
* SQLalchemy
* Pickle
* Flask, Plotly

### Running Program:
1. Following commands can be used to run the model in the project's directory to set up the database, train model and save the model.

    - Run ETL pipeline to clean data and store the processed data in the database via:
        `python data/process_data.py data/messages.csv data/categories.csv data/disaster_response_db.db`
    - Run the ML pipeline that loads data from DB, trains classifier and saves the classifier as a pickle file via:
        `python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Files

1. **ETL Preparation Notebook**: This Jupyter Notebook contains the implemented ETL pipeline and database creation
2. **ML Pipeline Preparation Notebook**: This notebook houses the Machine Learning Pipeline developed with NLTK and Scikit-Learn
3. **app/templates/***: html files for web app
4. **data/process_data.py**: Extract Train Load (ETL) pipeline used for data cleaning, feature extraction, and storing data in a SQLite database
5. **models/train_classifier.py**: A machine learning pipeline that loads data, trains a model, and saves the trained model as a .pkl file for later use
6. **run.py**: Flask web app used to classify disaster messages

## Acknowledgements

* [Udacity](https://www.udacity.com/)'s Data Science Nanodegree Program
* [Figure Eight](https://www.figure-eight.com/) for providing the relevant dataset



