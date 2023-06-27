import sys
from pathlib import Path
sys.path.append(str(Path(sys.argv[0]).absolute().parent.parent))
from utils.processing import clean_data
from utils.enums import (FilePath,
                         Constants,
                         )
import pandas as pd

def clean_test_data():
    # Read the raw data and process it
    df = pd.read_feather(path=FilePath.predict_file_path)
    # Process the data
    processed_df = clean_data(df=df,
                              mode=Constants.PREDICT_MODE,)
    #Save the processed data
    processed_df.to_feather(path=FilePath.processed_predict_file_path)

if __name__ == "__main__":
    clean_test_data()

