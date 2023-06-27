import sys
from pathlib import Path
sys.path.append(str(Path(sys.argv[0]).absolute().parent.parent))
from utils.helpers import DataProcessor
from utils.feature_engineering import FeatureEngineering
from pandas import DataFrame

data_processor = DataProcessor()
feature_engineering = FeatureEngineering()

def clean_data(df: DataFrame,
               mode:str,
               ) -> DataFrame:
    df.reset_index(drop=True, inplace=True)
    data_point_before_cleaning = len(df)
    #The below feature engineering fuctions must follow the order of the list,
    #because some functions depend on the output of the previous function,
    #also the historical columns have to be dropped after extracting insights from them,
    #to not let the dataframe fragmented with many columns,
    #refer to the demo_pipeline_1 notebook
    feature_engineering_train = [feature_engineering.drop_coach_columns,
                                data_processor.cast_datetime_cols,
                                #Feature engineering for match dates
                                feature_engineering.home_team_get_days_from_last_match,
                                feature_engineering.get_30_day_ago_match_date,
                                feature_engineering.get_home_team_match_date_index,
                                feature_engineering.get_away_team_match_date_index,
                                feature_engineering.get_home_team_total_matches_last_30_days,
                                feature_engineering.get_away_team_total_matches_last_30_days,
                                feature_engineering.drop_home_match_date_columns,
                                feature_engineering.drop_away_match_date_columns,
                                #Feature engineering for play home
                                feature_engineering.get_home_team_matches_play_home_last_30_days,
                                feature_engineering.get_away_team_matches_play_home_last_30_days,
                                feature_engineering.drop_home_team_play_home_columns,
                                feature_engineering.drop_away_team_play_home_columns,
                                #Feature engineering for cup competition
                                feature_engineering.get_home_team_cup_comp_last_10_matches,
                                feature_engineering.get_away_team_cup_comp_last_10_matches,
                                feature_engineering.get_home_team_if_last_match_cup,
                                feature_engineering.get_away_team_if_last_match_cup,
                                feature_engineering.get_home_team_cup_comp_last_30_days,
                                feature_engineering.get_away_team_cup_comp_last_30_days,
                                feature_engineering.drop_home_team_cup_comp_columns,
                                feature_engineering.drop_away_team_cup_comp_columns,
                                #Feature engineering for goals
                                feature_engineering.get_home_team_avg_goals_last_10_matches,
                                feature_engineering.get_away_team_avg_goals_last_10_matches,
                                feature_engineering.get_home_team_goals_last_match,
                                feature_engineering.get_away_team_goals_last_match,
                                feature_engineering.get_home_team_avg_goals_last_30_days,
                                feature_engineering.get_away_team_avg_goals_last_30_days,
                                feature_engineering.drop_home_team_goals_columns,
                                feature_engineering.drop_away_team_goals_columns,
                                #Feature engineering for opponent goals
                                feature_engineering.get_home_team_opponent_avg_goals_last_10_matches,
                                feature_engineering.get_away_team_opponent_avg_goals_last_10_matches,
                                feature_engineering.get_home_team_opponent_goals_last_match,
                                feature_engineering.get_away_team_opponent_goals_last_match,
                                feature_engineering.get_home_team_opponent_avg_goals_last_30_days,
                                feature_engineering.get_away_team_opponent_avg_goals_last_30_days,
                                feature_engineering.drop_home_team_opponent_goals_columns,
                                feature_engineering.drop_away_team_opponent_goals_columns,
                                #Remove matches with teams w/o last rating
                                feature_engineering.remove_matches_of_team_without_last_rating,
                                #Feature engineering for ratings
                                feature_engineering.get_home_team_avg_rating_last_10_matches,
                                feature_engineering.get_away_team_avg_rating_last_10_matches,
                                feature_engineering.get_home_team_rating_last_match,
                                feature_engineering.get_away_team_rating_last_match,
                                feature_engineering.get_home_team_avg_rating_last_30_days,
                                feature_engineering.get_away_team_avg_rating_last_30_days,
                                feature_engineering.drop_home_team_rating_columns,
                                feature_engineering.drop_away_team_rating_columns,
                                #Feature engineering for opponent ratings
                                feature_engineering.get_home_team_opponent_avg_rating_last_10_matches,
                                feature_engineering.get_away_team_opponent_avg_rating_last_10_matches,
                                feature_engineering.get_home_team_opponent_rating_last_match,
                                feature_engineering.get_away_team_opponent_rating_last_match,
                                feature_engineering.get_home_team_opponent_avg_rating_last_30_days,
                                feature_engineering.get_away_team_opponent_avg_rating_last_30_days,
                                feature_engineering.drop_home_team_opponent_rating_columns,
                                feature_engineering.drop_away_team_opponent_rating_columns,
                                #Feature engineering for leagues
                                feature_engineering.get_home_team_league_last_10_matches,
                                feature_engineering.get_away_team_league_last_10_matches,
                                feature_engineering.get_home_team_if_last_match_same_league,
                                feature_engineering.get_away_team_if_last_match_same_league,
                                feature_engineering.get_home_team_league_last_30_days,
                                feature_engineering.get_away_team_league_last_30_days,
                                feature_engineering.drop_home_team_league_columns,
                                feature_engineering.drop_away_team_league_columns,
                                #Drop redundant columns
                                feature_engineering.drop_date_index_columns,
                                feature_engineering.drop_redundant_columns,
                                #Feature engineering for general features
                                feature_engineering.features_from_match_date,
                                feature_engineering.cast_int_cup_comp,
                                #Drop redundant columns, and rows with too many missing values
                                feature_engineering.drop_match_date_column,
                                feature_engineering.remove_matches_with_less_history_last_30_days,
                                #Cast dtype 64 to 32 to boost up the training speed
                                feature_engineering.cast_int64_to_int32,
                                feature_engineering.cast_float64_to_float32,
                                ]

    feature_engineering_predict = [feature_engineering.drop_coach_columns,
                                data_processor.cast_datetime_cols,
                                #Feature engineering for match dates
                                feature_engineering.home_team_get_days_from_last_match,
                                feature_engineering.get_30_day_ago_match_date,
                                feature_engineering.get_home_team_match_date_index,
                                feature_engineering.get_away_team_match_date_index,
                                feature_engineering.get_home_team_total_matches_last_30_days,
                                feature_engineering.get_away_team_total_matches_last_30_days,
                                feature_engineering.drop_home_match_date_columns,
                                feature_engineering.drop_away_match_date_columns,
                                #Feature engineering for play home
                                feature_engineering.get_home_team_matches_play_home_last_30_days,
                                feature_engineering.get_away_team_matches_play_home_last_30_days,
                                feature_engineering.drop_home_team_play_home_columns,
                                feature_engineering.drop_away_team_play_home_columns,
                                #Feature engineering for cup competition
                                feature_engineering.get_home_team_cup_comp_last_10_matches,
                                feature_engineering.get_away_team_cup_comp_last_10_matches,
                                feature_engineering.get_home_team_if_last_match_cup,
                                feature_engineering.get_away_team_if_last_match_cup,
                                feature_engineering.get_home_team_cup_comp_last_30_days,
                                feature_engineering.get_away_team_cup_comp_last_30_days,
                                feature_engineering.drop_home_team_cup_comp_columns,
                                feature_engineering.drop_away_team_cup_comp_columns,
                                #Feature engineering for goals
                                feature_engineering.get_home_team_avg_goals_last_10_matches,
                                feature_engineering.get_away_team_avg_goals_last_10_matches,
                                feature_engineering.get_home_team_goals_last_match,
                                feature_engineering.get_away_team_goals_last_match,
                                feature_engineering.get_home_team_avg_goals_last_30_days,
                                feature_engineering.get_away_team_avg_goals_last_30_days,
                                feature_engineering.drop_home_team_goals_columns,
                                feature_engineering.drop_away_team_goals_columns,
                                #Feature engineering for opponent goals
                                feature_engineering.get_home_team_opponent_avg_goals_last_10_matches,
                                feature_engineering.get_away_team_opponent_avg_goals_last_10_matches,
                                feature_engineering.get_home_team_opponent_goals_last_match,
                                feature_engineering.get_away_team_opponent_goals_last_match,
                                feature_engineering.get_home_team_opponent_avg_goals_last_30_days,
                                feature_engineering.get_away_team_opponent_avg_goals_last_30_days,
                                feature_engineering.drop_home_team_opponent_goals_columns,
                                feature_engineering.drop_away_team_opponent_goals_columns,
                                #Remove matches with teams w/o last rating
                                # feature_engineering.remove_matches_of_team_without_last_rating,
                                #Feature engineering for ratings
                                feature_engineering.get_home_team_avg_rating_last_10_matches,
                                feature_engineering.get_away_team_avg_rating_last_10_matches,
                                feature_engineering.get_home_team_rating_last_match,
                                feature_engineering.get_away_team_rating_last_match,
                                feature_engineering.get_home_team_avg_rating_last_30_days,
                                feature_engineering.get_away_team_avg_rating_last_30_days,
                                feature_engineering.drop_home_team_rating_columns,
                                feature_engineering.drop_away_team_rating_columns,
                                #Feature engineering for opponent ratings
                                feature_engineering.get_home_team_opponent_avg_rating_last_10_matches,
                                feature_engineering.get_away_team_opponent_avg_rating_last_10_matches,
                                feature_engineering.get_home_team_opponent_rating_last_match,
                                feature_engineering.get_away_team_opponent_rating_last_match,
                                feature_engineering.get_home_team_opponent_avg_rating_last_30_days,
                                feature_engineering.get_away_team_opponent_avg_rating_last_30_days,
                                feature_engineering.drop_home_team_opponent_rating_columns,
                                feature_engineering.drop_away_team_opponent_rating_columns,
                                #Feature engineering for leagues
                                feature_engineering.get_home_team_league_last_10_matches,
                                feature_engineering.get_away_team_league_last_10_matches,
                                feature_engineering.get_home_team_if_last_match_same_league,
                                feature_engineering.get_away_team_if_last_match_same_league,
                                feature_engineering.get_home_team_league_last_30_days,
                                feature_engineering.get_away_team_league_last_30_days,
                                feature_engineering.drop_home_team_league_columns,
                                feature_engineering.drop_away_team_league_columns,
                                #Drop redundant columns
                                feature_engineering.drop_date_index_columns,
                                feature_engineering.drop_redundant_columns,
                                #Feature engineering for general features
                                feature_engineering.features_from_match_date,
                                feature_engineering.cast_int_cup_comp,
                                #Drop redundant columns, and rows with too many missing values
                                feature_engineering.drop_match_date_column,
                                # feature_engineering.remove_matches_with_less_history_last_30_days,
                                #Cast dtype 64 to 32 to boost up the training speed
                                feature_engineering.cast_int64_to_int32,
                                feature_engineering.cast_float64_to_float32,
                                #Fill missing values
                                feature_engineering.fillna_all_columns,
                                ]

    if mode == 'predict':
        feature_engineering_func = feature_engineering_predict
    elif mode == 'train':
        feature_engineering_func = feature_engineering_train
    else:
        raise ValueError("mode must be either 'predict' or 'train'")

    for func in feature_engineering_func:
        df = func(df=df)

    df.reset_index(drop=True, inplace=True)

    data_point_after_cleaning = len(df)
    print(f"{data_point_before_cleaning - data_point_after_cleaning} removed out of {data_point_before_cleaning}")
    print(f"Which is equivalent to: {round((data_point_before_cleaning - data_point_after_cleaning) / data_point_before_cleaning * 100, 2)}%")

    return df


