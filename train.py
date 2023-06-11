# Importing necessary libraries
import joblib
import os
import click

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from sklearn.feature_extraction import DictVectorizer as dv


#setting up tracking uri and experiment
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Random_regressor")

#setting up autologging 
mlflow.autolog()



# Defining a function to load pickle files
def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return joblib.load(f_in)

# Defining a command-line interface (CLI) function
@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="/Users/mandeebot/Desktop/MLOPS/week2/output"
)



def run_train(data_path: str):
    #creating mlflow run
    with mlflow.start_run():

        # Loading the training and validation data from pickle files
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        #setting experiment tags
        mlflow.set_tag('developer','mandeebot')

        #logging parameters
        mlflow.log_param('train-data-path','/Users/mandeebot/Desktop/MLOPS/week2/data_files/green_tripdata_2022-01.parquet')
        mlflow.log_param('train-data-path2','/Users/mandeebot/Desktop/MLOPS/week2/data_files/green_tripdata_2022-02.parquet')
        mlflow.log_param('train-data-path3','/Users/mandeebot/Desktop/MLOPS/week2/data_files/green_tripdata_2022-03.parquet')

        # Creating a Random Forest Regressor model
        rf = RandomForestRegressor(max_depth=10, random_state=0)

        # Fitting the model to the training data
        rf.fit(X_train, y_train)

        # Predicting the target values for the validation data
        y_pred = rf.predict(X_val)

        # Calculating the root mean squared error (RMSE)
        rmse = mean_squared_error(y_val, y_pred, squared=False)

        # Logging the RMSE metric to MLflow
        mlflow.log_metric("rmse", rmse)

        #setting up a model artificats
        mlflow.log_artifact(local_path='models/random_f.bin',artifact_path="artifacts_store")

        #loads the dictionary vectorizer object
        dv1 = load_pickle('/Users/mandeebot/Desktop/MLOPS/week2/output/dv.pkl')

        #creates a folder for models in directory
        os.makedirs('models')

        #saves trained model 
        with open('models/random_f.bin','wb') as f_out:
            joblib.dump((dv1,rf),f_out)

# Running the CLI function if the file is being run as the main program
if __name__ == "__main__":
    run_train()