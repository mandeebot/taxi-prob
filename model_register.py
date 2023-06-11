import os
import joblib
import click
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#rework code base on inputing code to check balance ai injecting any unwanted commands
#rework code 

HPO_EXPERIMENT_NAME = "random-forest-hyperopt" #replace name with nlp experiment hyper parameter optizmer name
EXPERIMENT_NAME = "random-forest-best-models" #replace name with name of model used
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state', 'n_jobs'] # setting model parameters



#setting monitoring tracking, experiment name and autologging 
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()

#function to upload model file
def load_pickle(filename):
    """
    Load a pickled object from a file.

    Args:
        filename (str): The path to the pickle file.

    Returns:
        The object that was pickled.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there is an error reading the file.
        pickle.UnpicklingError: If the pickle file is corrupt.
    """

    with open(filename, "rb") as f_in:
        return joblib.load(f_in)


#function to start mlflow runs,load dataset train and evaluate model and log model on mlflow
def train_and_log_model(data_path, params):

    """
    Trains a Random Forest regressor model and logs the results to MLflow.

    Args:
        data_path (str): The path to the directory containing the training, validation, and test data.
        params (dict): A dictionary of parameters for the Random Forest regressor.

    Returns:
        A RandomForestRegressor model.
    """
    # Load the training, validation, and test sets.
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    # Convert the parameters to integers.
    for param in RF_PARAMS:
        params[param] = int(params[param])

    # Create a Random Forest regressor.
    rf = RandomForestRegressor(**params)

    # Fit the model to the training data.
    rf.fit(X_train, y_train)

    # Evaluate the model on the validation and test sets.
    val_rmse = mean_squared_error(y_val, rf.predict(X_val), squared=False)
    mlflow.log_metric("val_rmse", val_rmse)
    test_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)
    mlflow.log_metric("test_rmse", test_rmse)

    # Return the model.
    return rf

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
#function to register model 
def run_register_model(data_path: str, top_n: int):

    """
    Registers the best model from a set of hyperparameter optimization runs.

    Args:
    data_path: The path to the data used to train the models.
    top_n: The number of top models to consider.

    Returns:
    The name of the registered model.
    """

    #Initialize the MLflow client.
    client = MlflowClient()

    #Retrieve the top_n model runs and log the models.
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(experiment_ids=experiment.experiment_id,
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=top_n,
    order_by=["metrics.rmse ASC"])

    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    #Select the model with the lowest test metric.
    best_run = client.search_runs(experiment_ids=experiment.experiment_id, order_by=["metrics.rmse ASC"])[0]

    #Register the best model.
    mlflow.register_model("runs:/{}".format(best_run.run_id),name="New-York_Taxi_Regressor")

    return best_run.run_id

if __name__ == '__main__':
    run_register_model()
