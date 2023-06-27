import pandas as pd
from pandas import DataFrame
import numpy as np
from utils.enums import (FeatureNames,
                         Patterns,
                         DTypes,
                         Constants,
                         )
from utils.helpers import DataProcessor

data_processor = DataProcessor()

class FeatureEngineering():
    def __init__(self):
        pass

    def drop_coach_columns(self,
                           df: DataFrame,
                           ) -> DataFrame:
        coach_related_cols = data_processor.get_list_column_names_by_pattern(df=df, pattern=Patterns.COACH)
        df = df.drop(columns=coach_related_cols, axis=1)  # Drop all columns related to coach, refer to the demo_pipeline_1 notebook

        return df

    def home_team_get_days_from_last_match(self,
                                           df: DataFrame,
                                           ) -> DataFrame:
        df = data_processor.get_date_diff(df=df,
                                          intended_col_name=FeatureNames.HOME_TEAM_DAYS_FROM_LAST_MATCH,
                                          date_col_1=FeatureNames.MATCH_DATE,
                                          date_col_2=FeatureNames.HOME_TEAM_HISTORY_MATCH_DATE_1,
                                          )

        return df

    def away_team_get_days_from_last_match(self,
                                           df: DataFrame,
                                           ) -> DataFrame:
        df = data_processor.get_date_diff(df=df,
                                          intended_col_name=FeatureNames.AWAY_TEAM_DAYS_FROM_LAST_MATCH,
                                          date_col_1=FeatureNames.MATCH_DATE,
                                          date_col_2=FeatureNames.AWAY_TEAM_HISTORY_MATCH_DATE_1,
                                          )

        return df

    def get_30_day_ago_match_date(self,
                                  df: DataFrame,
                                  ) -> DataFrame:
        df[FeatureNames._30_DAY_AGO_MATCH_DATE] = df[FeatureNames.MATCH_DATE] - pd.DateOffset(days=30)

        return df

    def get_home_team_match_date_index(self,
                                       df: DataFrame,
                                       ) -> DataFrame:
        home_match_date_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.HOME_TEAM_HISTORY_MATCH_DATE_,
                                                            )
        i = 1
        for col in home_match_date_cols:
            index_col = f"{Patterns.HOME_TEAM_DATE_INDEX_}{i}"
            mask = (df[col] >= df[FeatureNames._30_DAY_AGO_MATCH_DATE]) & (df[col] <= df[FeatureNames.MATCH_DATE])
            df[index_col] = mask.astype(DTypes.INT32)
            i += 1

        return df

    def get_away_team_match_date_index(self,
                                       df: DataFrame,
                                       ) -> DataFrame:
        away_match_date_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.AWAY_TEAM_HISTORY_MATCH_DATE_,
                                                            )
        i = 1
        for col in away_match_date_cols:
            index_col = f"{Patterns.AWAY_TEAM_DATE_INDEX_}{i}"
            mask = (df[col] >= df[FeatureNames._30_DAY_AGO_MATCH_DATE]) & (df[col] <= df[FeatureNames.MATCH_DATE])
            df[index_col] = mask.astype(DTypes.INT32)
            i += 1

        return df

    def get_home_team_total_matches_last_30_days(self,
                                                 df: DataFrame,
                                                 ) -> DataFrame:
        home_match_date_index_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.HOME_TEAM_DATE_INDEX_,
                                                            )
        df[FeatureNames.HOME_TEAM_TOTAL_MATCHES_LAST_30_DAYS] = df[home_match_date_index_cols].sum(axis=1)

        return df

    def get_away_team_total_matches_last_30_days(self,
                                                 df: DataFrame,
                                                 ) -> DataFrame:
        away_match_date_index_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.AWAY_TEAM_DATE_INDEX_,
                                                            )
        df[FeatureNames.AWAY_TEAM_TOTAL_MATCHES_LAST_30_DAYS] = df[away_match_date_index_cols].sum(axis=1)

        return df

    def drop_home_match_date_columns(self,
                                     df: DataFrame,
                                     ) -> DataFrame:
        home_match_date_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.HOME_TEAM_HISTORY_MATCH_DATE_,
                                                            )
        df = df.drop(home_match_date_cols, axis=1)

        return df

    def drop_away_match_date_columns(self,
                                     df: DataFrame,
                                     ) -> DataFrame:
        away_match_date_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.AWAY_TEAM_HISTORY_MATCH_DATE_,
                                                            )
        df = df.drop(away_match_date_cols, axis=1)

        return df

    def get_home_team_matches_play_home_last_30_days(self,
                                                     df: DataFrame,
                                                     ) -> DataFrame:
        df[FeatureNames.HOME_TEAM_MATCHES_PLAY_HOME_LAST_30_DAYS] = 0
        for i in range(1, 11):
            df[FeatureNames.HOME_TEAM_MATCHES_PLAY_HOME_LAST_30_DAYS] \
                += df[f'{Patterns.HOME_TEAM_HISTORY_IS_PLAY_HOME_}{i}'] \
                * df[f'{Patterns.HOME_TEAM_DATE_INDEX_}{i}']

        return df

    def get_away_team_matches_play_home_last_30_days(self,
                                                     df: DataFrame,
                                                     ) -> DataFrame:
        df[FeatureNames.AWAY_TEAM_MATCHES_PLAY_HOME_LAST_30_DAYS] = 0
        for i in range(1, 11):
            df[FeatureNames.AWAY_TEAM_MATCHES_PLAY_HOME_LAST_30_DAYS] \
                += df[f'{Patterns.AWAY_TEAM_HISTORY_IS_PLAY_HOME_}{i}'] \
                * df[f'{Patterns.AWAY_TEAM_DATE_INDEX_}{i}']

        return df

    def drop_home_team_play_home_columns(self,
                                         df: DataFrame,
                                         ) -> DataFrame:
        home_team_play_home_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.HOME_TEAM_HISTORY_IS_PLAY_HOME_,
                                                            )
        df = df.drop(home_team_play_home_cols, axis=1)

        return df

    def drop_away_team_play_home_columns(self,
                                         df: DataFrame,
                                         ) -> DataFrame:
        away_team_play_home_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.AWAY_TEAM_HISTORY_IS_PLAY_HOME_,
                                                            )
        df = df.drop(away_team_play_home_cols, axis=1)

        return df

    def get_home_team_cup_comp_last_10_matches(self,
                                               df: DataFrame,
                                               ) -> DataFrame:
        home_team_is_cup_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.HOME_TEAM_HISTORY_IS_CUP_,
                                                            )
        df[FeatureNames.HOME_TEAM_MATCHES_CUP_COMP_LAST_10_MATCHES] = df[home_team_is_cup_cols].sum(axis=1)

        return df

    def get_away_team_cup_comp_last_10_matches(self,
                                               df: DataFrame,
                                               ) -> DataFrame:
        away_team_is_cup_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.AWAY_TEAM_HISTORY_IS_CUP_,
                                                            )
        df[FeatureNames.AWAY_TEAM_MATCHES_CUP_COMP_LAST_10_MATCHES] = df[away_team_is_cup_cols].sum(axis=1)

        return df

    def get_home_team_if_last_match_cup(self,
                                        df: DataFrame,
                                        ) -> DataFrame:
        df[FeatureNames.HOME_TEAM_IS_LAST_MATCH_CUP] =\
            np.where(df[FeatureNames.HOME_TEAM_HISTORY_IS_CUP_1] == 1, 1, 0)

        return df

    def get_away_team_if_last_match_cup(self,
                                        df: DataFrame,
                                        ) -> DataFrame:
        df[FeatureNames.AWAY_TEAM_IS_LAST_MATCH_CUP] =\
            np.where(df[FeatureNames.AWAY_TEAM_HISTORY_IS_CUP_1] == 1, 1, 0)

        return df

    def get_home_team_cup_comp_last_30_days(self,
                                            df: DataFrame,
                                            ) -> DataFrame:
        home_team_is_cup_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.HOME_TEAM_HISTORY_IS_CUP_,
                                                            )
        df[home_team_is_cup_cols] = df[home_team_is_cup_cols].fillna(0)
        df[FeatureNames.HOME_TEAM_MATCHES_CUP_COMP_LAST_30_DAYS] = 0
        for i in range(1, 11):
            df[FeatureNames.HOME_TEAM_MATCHES_CUP_COMP_LAST_30_DAYS] \
                += df[f'{Patterns.HOME_TEAM_HISTORY_IS_CUP_}{i}'] \
                * df[f'{Patterns.AWAY_TEAM_DATE_INDEX_}{i}']

        return df

    def get_away_team_cup_comp_last_30_days(self,
                                            df: DataFrame,
                                            ) -> DataFrame:
        away_team_is_cup_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.AWAY_TEAM_HISTORY_IS_CUP_,
                                                            )
        df[away_team_is_cup_cols] = df[away_team_is_cup_cols].fillna(0)
        df[FeatureNames.AWAY_TEAM_MATCHES_CUP_COMP_LAST_30_DAYS] = 0
        for i in range(1, 11):
            df[FeatureNames.AWAY_TEAM_MATCHES_CUP_COMP_LAST_30_DAYS] \
                += df[f'{Patterns.AWAY_TEAM_HISTORY_IS_CUP_}{i}'] \
                * df[f'{Patterns.AWAY_TEAM_DATE_INDEX_}{i}']

        return df

    def drop_home_team_cup_comp_columns(self,
                                        df: DataFrame,
                                        ) -> DataFrame:
        home_team_is_cup_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.HOME_TEAM_HISTORY_IS_CUP_,
                                                            )
        df = df.drop(home_team_is_cup_cols, axis=1)

        return df

    def drop_away_team_cup_comp_columns(self,
                                        df: DataFrame,
                                        ) -> DataFrame:
        away_team_is_cup_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.AWAY_TEAM_HISTORY_IS_CUP_,
                                                            )
        df = df.drop(away_team_is_cup_cols, axis=1)

        return df

    def get_home_team_avg_goals_last_10_matches(self,
                                                  df: DataFrame,
                                                  ) -> DataFrame:
        home_team_goal_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.HOME_TEAM_HISTORY_GOALS_,
                                                            )
        df[FeatureNames.HOME_TEAM_AVG_GOALS_LAST_10_MATCHES] = df[home_team_goal_cols].mean(axis=1)

        return df

    def get_away_team_avg_goals_last_10_matches(self,
                                                  df: DataFrame,
                                                  ) -> DataFrame:
        away_team_goal_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.AWAY_TEAM_HISTORY_GOALS_,
                                                            )
        df[FeatureNames.AWAY_TEAM_AVG_GOALS_LAST_10_MATCHES] = df[away_team_goal_cols].mean(axis=1)

        return df

    def get_home_team_goals_last_match(self,
                                       df: DataFrame,
                                       ) -> DataFrame:
        df[FeatureNames.HOME_TEAM_GOALS_LAST_MATCH] = \
            df[FeatureNames.HOME_TEAM_HISTORY_GOAL_1]

        return df

    def get_away_team_goals_last_match(self,
                                       df: DataFrame,
                                       ) -> DataFrame:
        df[FeatureNames.AWAY_TEAM_GOALS_LAST_MATCH] = \
            df[FeatureNames.AWAY_TEAM_HISTORY_GOAL_1]

        return df

    def get_home_team_avg_goals_last_30_days(self,
                                               df: DataFrame,
                                               ) -> DataFrame:
        home_team_goal_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.HOME_TEAM_HISTORY_GOALS_,
                                                            )
        df[home_team_goal_cols] = df[home_team_goal_cols].fillna(0)
        df[FeatureNames.HOME_TEAM_AVG_GOALS_LAST_30_DAYS] = 0
        for i in range(1, 11):
            df[FeatureNames.HOME_TEAM_AVG_GOALS_LAST_30_DAYS] \
                += df[f'{Patterns.HOME_TEAM_HISTORY_GOALS_}{i}'] \
                * df[f'{Patterns.HOME_TEAM_DATE_INDEX_}{i}']
        df[FeatureNames.HOME_TEAM_AVG_GOALS_LAST_30_DAYS] \
        /= df[FeatureNames.HOME_TEAM_TOTAL_MATCHES_LAST_30_DAYS]

        return df

    def get_away_team_avg_goals_last_30_days(self,
                                               df: DataFrame,
                                               ) -> DataFrame:
        away_team_goal_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.AWAY_TEAM_HISTORY_GOALS_,
                                                            )
        df[away_team_goal_cols] = df[away_team_goal_cols].fillna(0)
        df[FeatureNames.AWAY_TEAM_AVG_GOALS_LAST_30_DAYS] = 0
        for i in range(1, 11):
            df[FeatureNames.AWAY_TEAM_AVG_GOALS_LAST_30_DAYS] \
                += df[f'{Patterns.AWAY_TEAM_HISTORY_GOALS_}{i}'] \
                * df[f'{Patterns.AWAY_TEAM_DATE_INDEX_}{i}']
        df[FeatureNames.AWAY_TEAM_AVG_GOALS_LAST_30_DAYS] \
        /= df[FeatureNames.AWAY_TEAM_TOTAL_MATCHES_LAST_30_DAYS]

        return df

    def drop_home_team_goals_columns(self,
                                     df: DataFrame,
                                     ) -> DataFrame:
        home_team_goal_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.HOME_TEAM_HISTORY_GOALS_,
                                                            )
        df = df.drop(home_team_goal_cols, axis=1)

        return df

    def drop_away_team_goals_columns(self,
                                     df: DataFrame,
                                     ) -> DataFrame:
        away_team_goal_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.AWAY_TEAM_HISTORY_GOALS_,
                                                            )
        df = df.drop(away_team_goal_cols, axis=1)

        return df

    def get_home_team_opponent_avg_goals_last_10_matches(self,
                                                         df: DataFrame,
                                                         ) -> DataFrame:
        home_team_opponent_goal_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.HOME_TEAM_HISTORY_OPPONENT_GOAL_,
                                                            )
        df[FeatureNames.HOME_TEAM_OPPONENT_AVG_GOALS_LAST_10_MATCHES] = df[home_team_opponent_goal_cols].mean(axis=1)

        return df

    def get_away_team_opponent_avg_goals_last_10_matches(self,
                                                         df: DataFrame,
                                                         ) -> DataFrame:
        away_team_opponent_goal_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.AWAY_TEAM_HISTORY_OPPONENT_GOAL_,
                                                            )
        df[FeatureNames.AWAY_TEAM_OPPONENT_AVG_GOALS_LAST_10_MATCHES] = df[away_team_opponent_goal_cols].mean(axis=1)

        return df

    def get_home_team_opponent_goals_last_match(self,
                                                df: DataFrame,
                                                ) -> DataFrame:
        df[FeatureNames.HOME_TEAM_OPPONENT_GOALS_LAST_MATCH] = \
            df[FeatureNames.HOME_TEAM_HISTORY_OPPONENT_GOAL_1]

        return df

    def get_away_team_opponent_goals_last_match(self,
                                                df: DataFrame,
                                                ) -> DataFrame:
        df[FeatureNames.AWAY_TEAM_OPPONENT_GOALS_LAST_MATCH] = \
            df[FeatureNames.AWAY_TEAM_HISTORY_OPPONENT_GOAL_1]

        return df

    def get_home_team_opponent_avg_goals_last_30_days(self,
                                                      df: DataFrame,
                                                      ) -> DataFrame:
        home_team_opponent_goal_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.HOME_TEAM_HISTORY_OPPONENT_GOAL_,
                                                            )
        df[home_team_opponent_goal_cols] = df[home_team_opponent_goal_cols].fillna(0)
        df[FeatureNames.HOME_TEAM_OPPONENT_AVG_GOALS_LAST_30_DAYS] = 0
        for i in range(1, 11):
            df[FeatureNames.HOME_TEAM_OPPONENT_AVG_GOALS_LAST_30_DAYS] \
                += df[f'{Patterns.HOME_TEAM_HISTORY_OPPONENT_GOAL_}{i}'] \
                * df[f'{Patterns.HOME_TEAM_DATE_INDEX_}{i}']
        df[FeatureNames.HOME_TEAM_OPPONENT_AVG_GOALS_LAST_30_DAYS] \
        /= df[FeatureNames.HOME_TEAM_TOTAL_MATCHES_LAST_30_DAYS]

        return df

    def get_away_team_opponent_avg_goals_last_30_days(self,
                                                      df: DataFrame,
                                                      ) -> DataFrame:
        away_team_opponent_goal_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.AWAY_TEAM_HISTORY_OPPONENT_GOAL_,
                                                            )
        df[away_team_opponent_goal_cols] = df[away_team_opponent_goal_cols].fillna(0)
        df[FeatureNames.AWAY_TEAM_OPPONENT_AVG_GOALS_LAST_30_DAYS] = 0
        for i in range(1, 11):
            df[FeatureNames.AWAY_TEAM_OPPONENT_AVG_GOALS_LAST_30_DAYS] \
                += df[f'{Patterns.AWAY_TEAM_HISTORY_OPPONENT_GOAL_}{i}'] \
                * df[f'{Patterns.AWAY_TEAM_DATE_INDEX_}{i}']
        df[FeatureNames.AWAY_TEAM_OPPONENT_AVG_GOALS_LAST_30_DAYS] \
        /= df[FeatureNames.AWAY_TEAM_TOTAL_MATCHES_LAST_30_DAYS]

        return df

    def drop_home_team_opponent_goals_columns(self,
                                              df: DataFrame,
                                              ) -> DataFrame:
        home_team_opponent_goal_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.HOME_TEAM_HISTORY_OPPONENT_GOAL_,
                                                            )
        df = df.drop(home_team_opponent_goal_cols, axis=1)

        return df

    def drop_away_team_opponent_goals_columns(self,
                                              df: DataFrame,
                                              ) -> DataFrame:
        away_team_opponent_goal_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.AWAY_TEAM_HISTORY_OPPONENT_GOAL_,
                                                            )
        df = df.drop(away_team_opponent_goal_cols, axis=1)

        return df

    def get_home_team_avg_rating_last_10_matches(self,
                                                 df: DataFrame,
                                                 ) -> DataFrame:
        home_team_rating_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.HOME_TEAM_HISTORY_RATING_,
                                                            )
        df[FeatureNames.HOME_TEAM_AVG_RATING_LAST_10_MATCHES] = df[home_team_rating_cols].mean(axis=1)

        return df

    def get_away_team_avg_rating_last_10_matches(self,
                                                 df: DataFrame,
                                                 ) -> DataFrame:
        away_team_rating_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.AWAY_TEAM_HISTORY_RATING_,
                                                            )
        df[FeatureNames.AWAY_TEAM_AVG_RATING_LAST_10_MATCHES] = df[away_team_rating_cols].mean(axis=1)

        return df

    def get_home_team_rating_last_match(self,
                                        df: DataFrame,
                                        ) -> DataFrame:
        df[FeatureNames.HOME_TEAM_RATING_LAST_MATCH] = \
            df[FeatureNames.HOME_TEAM_HISTORY_RATING_1]

        return df

    def get_away_team_rating_last_match(self,
                                        df: DataFrame,
                                        ) -> DataFrame:
        df[FeatureNames.AWAY_TEAM_RATING_LAST_MATCH] = \
            df[FeatureNames.AWAY_TEAM_HISTORY_RATING_1]

        return df

    def get_home_team_avg_rating_last_30_days(self,
                                              df: DataFrame,
                                              ) -> DataFrame:
        home_team_rating_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.HOME_TEAM_HISTORY_RATING_,
                                                            )
        df[home_team_rating_cols] = df[home_team_rating_cols].fillna(0)
        df[FeatureNames.HOME_TEAM_AVG_RATING_LAST_30_DAYS] = 0
        for i in range(1, 11):
            df[FeatureNames.HOME_TEAM_AVG_RATING_LAST_30_DAYS] \
                += df[f'{Patterns.HOME_TEAM_HISTORY_RATING_}{i}'] \
                * df[f'{Patterns.HOME_TEAM_DATE_INDEX_}{i}']
        df[FeatureNames.HOME_TEAM_AVG_RATING_LAST_30_DAYS] \
        /= df[FeatureNames.HOME_TEAM_TOTAL_MATCHES_LAST_30_DAYS]

        return df

    def get_away_team_avg_rating_last_30_days(self,
                                              df: DataFrame,
                                              ) -> DataFrame:
        away_team_rating_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.AWAY_TEAM_HISTORY_RATING_,
                                                            )
        df[away_team_rating_cols] = df[away_team_rating_cols].fillna(0)
        df[FeatureNames.AWAY_TEAM_AVG_RATING_LAST_30_DAYS] = 0
        for i in range(1, 11):
            df[FeatureNames.AWAY_TEAM_AVG_RATING_LAST_30_DAYS] \
                += df[f'{Patterns.AWAY_TEAM_HISTORY_RATING_}{i}'] \
                * df[f'{Patterns.AWAY_TEAM_DATE_INDEX_}{i}']
        df[FeatureNames.AWAY_TEAM_AVG_RATING_LAST_30_DAYS] \
        /= df[FeatureNames.AWAY_TEAM_TOTAL_MATCHES_LAST_30_DAYS]

        return df

    def drop_home_team_rating_columns(self,
                                      df: DataFrame,
                                      ) -> DataFrame:
        home_team_rating_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.HOME_TEAM_HISTORY_RATING_,
                                                            )
        df = df.drop(home_team_rating_cols, axis=1)

        return df

    def drop_away_team_rating_columns(self,
                                      df: DataFrame,
                                      ) -> DataFrame:
        away_team_rating_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.AWAY_TEAM_HISTORY_RATING_,
                                                            )
        df = df.drop(away_team_rating_cols, axis=1)

        return df

    def get_home_team_opponent_avg_rating_last_10_matches(self,
                                                          df: DataFrame,
                                                          ) -> DataFrame:
        home_team_opponent_rating_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.HOME_TEAM_HISTORY_OPPONENT_RATING_,
                                                            )
        df[FeatureNames.HOME_TEAM_OPPONENT_AVG_RATING_LAST_10_MATCHES] = df[home_team_opponent_rating_cols].mean(axis=1)

        return df

    def get_away_team_opponent_avg_rating_last_10_matches(self,
                                                          df: DataFrame,
                                                          ) -> DataFrame:
        away_team_opponent_rating_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.AWAY_TEAM_HISTORY_OPPONENT_RATING_,
                                                            )
        df[FeatureNames.AWAY_TEAM_OPPONENT_AVG_RATING_LAST_10_MATCHES] = df[away_team_opponent_rating_cols].mean(axis=1)

        return df

    def get_home_team_opponent_rating_last_match(self,
                                                 df: DataFrame,
                                                 ) -> DataFrame:
        df[FeatureNames.HOME_TEAM_OPPONENT_RATING_LAST_MATCH] = \
            df[FeatureNames.HOME_TEAM_HISTORY_OPPONENT_RATING_1]

        return df

    def get_away_team_opponent_rating_last_match(self,
                                                 df: DataFrame,
                                                 ) -> DataFrame:
        df[FeatureNames.AWAY_TEAM_OPPONENT_RATING_LAST_MATCH] = \
            df[FeatureNames.AWAY_TEAM_HISTORY_OPPONENT_RATING_1]

        return df

    def get_home_team_opponent_avg_rating_last_30_days(self,
                                                       df: DataFrame,
                                                       ) -> DataFrame:
        home_team_opponent_rating_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.HOME_TEAM_HISTORY_OPPONENT_RATING_,
                                                            )
        df[home_team_opponent_rating_cols] = df[home_team_opponent_rating_cols].fillna(0)
        df[FeatureNames.HOME_TEAM_OPPONENT_AVG_RATING_LAST_30_DAYS] = 0
        for i in range(1, 11):
            df[FeatureNames.HOME_TEAM_OPPONENT_AVG_RATING_LAST_30_DAYS] \
                += df[f'{Patterns.HOME_TEAM_HISTORY_OPPONENT_RATING_}{i}'] \
                * df[f'{Patterns.HOME_TEAM_DATE_INDEX_}{i}']
        df[FeatureNames.HOME_TEAM_OPPONENT_AVG_RATING_LAST_30_DAYS] \
        /= df[FeatureNames.HOME_TEAM_TOTAL_MATCHES_LAST_30_DAYS]

        return df

    def get_away_team_opponent_avg_rating_last_30_days(self,
                                                       df: DataFrame,
                                                       ) -> DataFrame:
        away_team_opponent_rating_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.AWAY_TEAM_HISTORY_OPPONENT_RATING_,
                                                            )
        df[away_team_opponent_rating_cols] = df[away_team_opponent_rating_cols].fillna(0)
        df[FeatureNames.AWAY_TEAM_OPPONENT_AVG_RATING_LAST_30_DAYS] = 0
        for i in range(1, 11):
            df[FeatureNames.AWAY_TEAM_OPPONENT_AVG_RATING_LAST_30_DAYS] \
                += df[f'{Patterns.AWAY_TEAM_HISTORY_OPPONENT_RATING_}{i}'] \
                * df[f'{Patterns.AWAY_TEAM_DATE_INDEX_}{i}']
        df[FeatureNames.AWAY_TEAM_OPPONENT_AVG_RATING_LAST_30_DAYS] \
        /= df[FeatureNames.AWAY_TEAM_TOTAL_MATCHES_LAST_30_DAYS]

        return df

    def drop_home_team_opponent_rating_columns(self,
                                               df: DataFrame,
                                               ) -> DataFrame:
        home_team_opponent_rating_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.HOME_TEAM_HISTORY_OPPONENT_RATING_,
                                                            )
        df = df.drop(home_team_opponent_rating_cols, axis=1)

        return df

    def drop_away_team_opponent_rating_columns(self,
                                               df: DataFrame,
                                               ) -> DataFrame:
        away_team_opponent_rating_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.AWAY_TEAM_HISTORY_OPPONENT_RATING_,
                                                            )
        df = df.drop(away_team_opponent_rating_cols, axis=1)

        return df

    def get_home_team_league_last_10_matches(self,
                                             df: DataFrame,
                                             ) -> DataFrame:
        home_team_leagues_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.HOME_TEAM_HISTORY_LEAGUE_ID_,
                                                            )
        df[FeatureNames.HOME_TEAM_LEAGUES_LAST_10_MATCHES] = df[home_team_leagues_cols].nunique(axis=1)

        return df

    def get_away_team_league_last_10_matches(self,
                                             df: DataFrame,
                                             ) -> DataFrame:
        away_team_leagues_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.AWAY_TEAM_HISTORY_LEAGUE_ID_,
                                                            )
        df[FeatureNames.AWAY_TEAM_LEAGUES_LAST_10_MATCHES] = df[away_team_leagues_cols].nunique(axis=1)

        return df

    def get_home_team_if_last_match_same_league(self,
                                                df: DataFrame,
                                                ) -> DataFrame:

        df[FeatureNames.HOME_TEAM_SAME_LEAGUE] =\
            np.where(df[FeatureNames.HOME_TEAM_HISTORY_LEAGUE_ID_1] == df[FeatureNames.LEAGUE_ID], 1, 0)

        return df

    def get_away_team_if_last_match_same_league(self,
                                                df: DataFrame,
                                                ) -> DataFrame:

        df[FeatureNames.AWAY_TEAM_SAME_LEAGUE] =\
            np.where(df[FeatureNames.AWAY_TEAM_HISTORY_LEAGUE_ID_1] == df[FeatureNames.LEAGUE_ID], 1, 0)

        return df

    def get_home_team_league_last_30_days(self,
                                          df: DataFrame,
                                          ) -> DataFrame:
        home_team_league = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.HOME_TEAM_HISTORY_LEAGUE_ID_,
                                                            )
        df[home_team_league] = df[home_team_league].fillna(0)

        for i in range(1, 11):
            df[f'{Patterns.HOME_TEAM_HISTORY_LEAGUE_ID_}{i}'] =\
                df[f'{Patterns.HOME_TEAM_HISTORY_LEAGUE_ID_}{i}'] \
                * df[f'{Patterns.HOME_TEAM_DATE_INDEX_}{i}']

        df[FeatureNames.HOME_TEAM_LEAGUES_LAST_30_DAYS] =\
            pd.concat([df[col] for col in home_team_league], axis=1).nunique(axis=1)
        # Minus 1 to exclude the fillna(0) above, also not minus if there is no record for league_id
        df[FeatureNames.HOME_TEAM_LEAGUES_LAST_30_DAYS] =\
            np.where((df[FeatureNames.HOME_TEAM_HISTORY_LEAGUE_ID_1] != 0) &
                        (df[FeatureNames.HOME_TEAM_LEAGUES_LAST_30_DAYS] > 1),
                     df[FeatureNames.HOME_TEAM_LEAGUES_LAST_30_DAYS] - 1,
                     df[FeatureNames.HOME_TEAM_LEAGUES_LAST_30_DAYS])

        return df

    def get_away_team_league_last_30_days(self,
                                          df: DataFrame,
                                          ) -> DataFrame:
        away_team_league = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.AWAY_TEAM_HISTORY_LEAGUE_ID_,
                                                            )
        df[away_team_league] = df[away_team_league].fillna(0)

        for i in range(1, 11):
            df[f'{Patterns.AWAY_TEAM_HISTORY_LEAGUE_ID_}{i}'] =\
                df[f'{Patterns.AWAY_TEAM_HISTORY_LEAGUE_ID_}{i}'] \
                * df[f'{Patterns.AWAY_TEAM_DATE_INDEX_}{i}']

        df[FeatureNames.AWAY_TEAM_LEAGUES_LAST_30_DAYS] =\
            pd.concat([df[col] for col in away_team_league], axis=1).nunique(axis=1)
        # Minus 1 to exclude the fillna(0) above, also not minus if there is no record for league_id
        df[FeatureNames.AWAY_TEAM_LEAGUES_LAST_30_DAYS] =\
            np.where((df[FeatureNames.AWAY_TEAM_HISTORY_LEAGUE_ID_1] != 0) &
                        (df[FeatureNames.AWAY_TEAM_LEAGUES_LAST_30_DAYS] > 1),
                     df[FeatureNames.AWAY_TEAM_LEAGUES_LAST_30_DAYS] - 1,
                     df[FeatureNames.AWAY_TEAM_LEAGUES_LAST_30_DAYS])

        return df

    def drop_home_team_league_columns(self,
                                      df: DataFrame,
                                      ) -> DataFrame:
        home_team_league_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.HOME_TEAM_HISTORY_LEAGUE_ID_,
                                                            )
        df = df.drop(home_team_league_cols, axis=1)

        return df

    def drop_away_team_league_columns(self,
                                      df: DataFrame,
                                      ) -> DataFrame:
        away_team_league_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.AWAY_TEAM_HISTORY_LEAGUE_ID_,
                                                            )
        df = df.drop(away_team_league_cols, axis=1)

        return df

    def drop_date_index_columns(self,
                                df: DataFrame,
                                ) -> DataFrame:
        date_index_cols = \
            data_processor.get_list_column_names_by_pattern(df=df,
                                                            pattern=Patterns.DATE_INDEX,
                                                            )
        df = df.drop(date_index_cols, axis=1)

        return df

    def drop_redundant_columns(self,
                               df: DataFrame,
                               ) -> DataFrame:
        df = df.drop(columns=FeatureNames.FINAL_REDUNDANT_FEATURES, axis=1)

        return df

    def features_from_match_date(self,
                                 df: DataFrame,
                                 ) -> DataFrame:
        df[FeatureNames.DOW_MATCH] = df[FeatureNames.MATCH_DATE].dt.dayofweek
        df[FeatureNames.MONTH_MATCH] = df[FeatureNames.MATCH_DATE].dt.month
        df[FeatureNames.YEAR_MATCH] = df[FeatureNames.MATCH_DATE].dt.year
        df[FeatureNames.WEEK_MATCH] = df[FeatureNames.MATCH_DATE].dt.isocalendar().week

        df[FeatureNames.DOW_MATCH] = df[FeatureNames.DOW_MATCH].astype(DTypes.INT32)
        df[FeatureNames.MONTH_MATCH] = df[FeatureNames.MONTH_MATCH].astype(DTypes.INT32)
        df[FeatureNames.YEAR_MATCH] = df[FeatureNames.YEAR_MATCH].astype(DTypes.INT32)
        df[FeatureNames.WEEK_MATCH] = df[FeatureNames.WEEK_MATCH].astype(DTypes.INT32)

        return df

    def cast_int_cup_comp(self,
                          df: DataFrame,
                          ) -> DataFrame:
        df[FeatureNames.IS_CUP] = df[FeatureNames.IS_CUP].fillna(0)
        df[FeatureNames.IS_CUP] = df[FeatureNames.IS_CUP].astype(DTypes.INT32)

        return df

    def drop_match_date_column(self,
                               df: DataFrame,
                               ) -> DataFrame:
        df = df.drop(columns=FeatureNames.MATCH_DATE, axis=1)

        return df

    def remove_matches_with_less_history_last_30_days(self,
                                                      df: DataFrame,
                                                      ) -> DataFrame:
        df = df[(df[FeatureNames.HOME_TEAM_LEAGUES_LAST_30_DAYS].notnull()) &\
                (df[FeatureNames.AWAY_TEAM_LEAGUES_LAST_30_DAYS].notnull())]

        return df

    def remove_matches_of_team_without_last_rating(self,
                                                   df: DataFrame,
                                                   ) -> DataFrame:
        df = df[df[FeatureNames.HOME_TEAM_HISTORY_RATING_1].notnull() &
                df[FeatureNames.HOME_TEAM_HISTORY_OPPONENT_RATING_1].notnull() &
                df[FeatureNames.AWAY_TEAM_HISTORY_RATING_1].notnull() &
                df[FeatureNames.AWAY_TEAM_HISTORY_OPPONENT_RATING_1].notnull()]

        return df

    def cast_int64_to_int32(self,
                            df: DataFrame,
                            ) -> DataFrame:
        int64_columns = df.select_dtypes(include=DTypes.INT64).columns
        df[int64_columns] = df[int64_columns].astype(DTypes.INT32)

        return df

    def cast_float64_to_float32(self,
                                df: DataFrame,
                                ) -> DataFrame:
        float64_columns = df.select_dtypes(include=DTypes.FLOAT64).columns
        df[float64_columns] = df[float64_columns].astype(DTypes.FLOAT32)

        return df

    def fillna_all_columns(self,
                           df: DataFrame,
                           ) -> DataFrame:
        df = df.fillna(0)

        return df

    def label_the_target(self,
                         df: DataFrame,
                         ) -> DataFrame:
        df[FeatureNames.TARGET] = df[FeatureNames.TARGET].replace(Constants.target_label)

        return df

    def separate_targets_regression(self,
                                    df: DataFrame,
                                    ) -> DataFrame:
        df[[FeatureNames.HOME_SCORE, FeatureNames.AWAY_SCORE]] = \
            df[FeatureNames.SCORE].str.split('-', expand=True).astype(DTypes.INT32)

        df = df.drop(columns=FeatureNames.SCORE, axis=1)

        return df

    def drop_target_column(self,
                           df: DataFrame,
                           ) -> DataFrame:
        df = df.drop(columns=FeatureNames.TARGET, axis=1)

        return df
