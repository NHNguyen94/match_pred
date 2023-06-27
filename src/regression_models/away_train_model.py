import os
import sys
from pathlib import Path
sys.path.append(str(Path(sys.argv[0]).absolute().parent.parent.parent))
from utils.enums import (FilePath,
                         FeatureNames,
                         )
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_model():
    # Read the raw data and process it
    df = pd.read_feather(path=FilePath.regression_processed_train_file_path)
    #Split the train and test data
    X = df.drop(columns=[FeatureNames.HOME_SCORE, FeatureNames.AWAY_SCORE, FeatureNames.ID], axis=1)
    y = df[FeatureNames.AWAY_SCORE]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #Train the model
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    #Evaluate the model and save the evaluation results
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("Mean Absolute Error:", mae)

    evaluation_results = {
        'Mean Squared Error': mse,
        'Root Mean Squared Error': rmse,
        'Mean Absolute Error': mae
    }

    df = pd.DataFrame.from_dict(evaluation_results,
                                orient='index',
                                columns=['score'])

    df.to_csv(FilePath.regression_home_evaluation_path)
    #Save the model
    model_path = FilePath.regressor_home_model_path
    model.save_model(model_path)
    # Check if the model file is saved
    if os.path.exists(model_path):
        print(f"Model saved successfully to path: {FilePath.regressor_home_model_path}")
    else:
        print("Failed to save the model")

if __name__ == "__main__":
    train_model()

