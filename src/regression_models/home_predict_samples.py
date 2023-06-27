import sys
from pathlib import Path
sys.path.append(str(Path(sys.argv[0]).absolute().parent.parent.parent))
from utils.enums import (FilePath,
                         FeatureNames,
                         Constants,
                         )
import pandas as pd
import xgboost as xgb

def predict():
    #Load pre-processed data
    df = pd.read_feather(path=FilePath.processed_predict_file_path)
    X = df.drop(columns=[FeatureNames.ID], axis=1)
    #Load the model
    model = xgb.XGBRegressor()
    model.load_model(FilePath.regressor_home_model_path)
    #Predict probabilities
    probabilities = model.predict_proba(X)
    #Convert the probabilities to dataframe
    target_enum = list(Constants.target_label.keys())
    df_probabilities = pd.DataFrame(probabilities, columns=target_enum)
    #Round up to 3 digits
    df_probabilities = df_probabilities.round(3)
    #Save the prediction
    predicted_result = pd.concat([df[[FeatureNames.ID]], df_probabilities], axis=1)
    desired_order = [FeatureNames.ID,Constants.HOME,Constants.DRAW,Constants.AWAY]
    predicted_result = predicted_result[desired_order]
    predicted_result.to_csv(FilePath.submission_path, index=False)

if __name__ == "__main__":
    predict()
