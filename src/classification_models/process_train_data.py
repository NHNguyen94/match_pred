import sys
from pathlib import Path
sys.path.append(str(Path(sys.argv[0]).absolute().parent.parent.parent))
from utils.processing import clean_data
from utils.enums import (FilePath,
                         FeatureNames,
                         Constants,
                         )
from utils.feature_engineering import FeatureEngineering
import pandas as pd

feature_engineering = FeatureEngineering()

def clean_train_data():
    # Read the raw data
    df = pd.read_feather(path=FilePath.train_file_path)
    # Refer to the demo_pipeline_1 notebook, these 25 matches are lacking too much information,
    # and since the number (25 matches) is not that much, hard code to remove shall be used to simplify the process
    df = df[~df[FeatureNames.ID].isin(Constants.ids_missing_many_info)]
    processed_df = clean_data(df=df,
                              mode=Constants.TRAIN_MODE,
                              )
    processed_df = feature_engineering.label_the_target(df=processed_df)
    #Save the processed data
    processed_df.to_feather(path=FilePath.classification_processed_train_file_path)

if __name__ == "__main__":
    clean_train_data()

