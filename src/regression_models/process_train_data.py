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
    df_score = pd.read_feather(path=FilePath.targets_for_regression_path)
    # Refer to the demo_pipeline_1 notebook, these 25 matches are lacking too much information,
    # and since the number (25 matches) is not that much, hard code to remove shall be used to simplify the process
    df = df[~df[FeatureNames.ID].isin(Constants.ids_missing_many_info)]
    processed_df = clean_data(df=df,
                              mode=Constants.TRAIN_MODE,
                              )
    processed_df = feature_engineering.drop_target_column(df=processed_df)
    # Read the score data
    df_score = feature_engineering.separate_targets_regression(df=df_score)
    df_score = feature_engineering.drop_target_column(df=df_score)
    # Join to get new score data as labels
    final_data = processed_df.merge(df_score,
                                    on=FeatureNames.ID,
                                    how=Constants.LEFT_JOIN,
                                    )
    #Save the processed data
    final_data.to_feather(path=FilePath.regression_processed_train_file_path)

if __name__ == "__main__":
    clean_train_data()
