# Import necessary libraries
import os
import joblib
import click
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

# Define a function to dump a pickle file
def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return joblib.dump(obj, f_out)

# Define a function to read a parquet file
def read_dataframe(filename: str):
    df = pd.read_parquet(filename)
    return df

# Define a function to preprocess the data
def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    # Create a new column called `PU_DO`
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    # Get the categorical and numerical columns
    categorical = ['PU_DO']
    numerical = ['trip_distance']

    # Convert the categorical columns to strings
    df[categorical] = df[categorical].astype(str)

    # Convert the DataFrame to a dictionary of dictionaries
    dicts = df[categorical + numerical].to_dict(orient='records')

    # If `fit_dv` is True, fit the DictVectorizer and transform the data
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv

# Define a command-line interface (CLI)
@click.command()
@click.option(
    "--raw_data_path",
    help="/Users/mandeebot/Desktop/MLOPS/week2/data_files"
)
@click.option(
    "--dest_path",
    help="/Users/mandeebot/Desktop/MLOPS/week2/new_files"
)

#reading all data
def run_data_prep(raw_data_path: str, dest_path: str, dataset: str = "green"):
    # Load the parquet files
    df_train = read_dataframe(
        os.path.join(raw_data_path, f"{dataset}_tripdata_2022-01.parquet")
    )
    df_val = read_dataframe(
        os.path.join(raw_data_path, f"{dataset}_tripdata_2022-02.parquet")
    )
    df_test = read_dataframe(
        os.path.join(raw_data_path, f"{dataset}_tripdata_2022-03.parquet")
    )

    # Extract the target
    target = 'tip_amount'
    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values

    # Fit the DictVectorizer and preprocess the data
    dv = DictVectorizer()
    X_train, dv = preprocess(df_train, dv, fit_dv=True)
    X_val, _ = preprocess(df_val, dv, fit_dv=False)
    X_test, _ = preprocess(df_test, dv, fit_dv=False)

    # Create the `dest_path` folder if it doesn't already exist
    os.makedirs(dest_path, exist_ok=True)

    # Save the DictVectorizer and datasets
    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))


if __name__ == '__main__':
    run_data_prep()