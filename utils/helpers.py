import pandas as pd
from pandas import DataFrame
from typing import List
from utils.enums import (Patterns,
                         DTypes,
                         )

class DataProcessor():
    def __init__(self):
        pass

    def get_list_column_names_by_pattern(self,
                                         df: DataFrame,
                                         pattern: str,
                                         ) -> List[str]:

        return df.filter(like=pattern).columns.tolist()

    def cast_dtype_multiple_cols(self,
                                 df: DataFrame,
                                 col_list: List[str],
                                 dtype: str,
                                 ) -> DataFrame:

        for col in col_list:
            df[col] = df[col].astype(dtype)

        return df

    def cast_datetime_cols(self,
                           df: DataFrame,
                           ) -> DataFrame:
        col_list = self.get_list_column_names_by_pattern(df=df,
                                                         pattern=Patterns.DATE,
                                                         )

        return self.cast_dtype_multiple_cols(df=df,
                                             col_list=col_list,
                                             dtype=DTypes.DATETIME_64,
                                             )

    def get_date_diff(self,
                      df: DataFrame,
                      intended_col_name: str,
                      date_col_1: str,
                      date_col_2: str,
                      ) -> DataFrame:
        df[intended_col_name] = df[date_col_1] - df[date_col_2]
        df[intended_col_name] = df[intended_col_name] / pd.Timedelta(days=1)

        return df
