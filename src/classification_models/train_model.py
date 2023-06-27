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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model():
    # Read the raw data and process it
    df = pd.read_feather(path=FilePath.classification_processed_train_file_path)
    #Split the train and test data
    X = df.drop(columns=[FeatureNames.TARGET, FeatureNames.ID], axis=1)
    y = df[FeatureNames.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #Train the model
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    #Evaluate the model and save the evaluation results
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

    evaluation_results = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

    df = pd.DataFrame.from_dict(evaluation_results,
                                orient='index',
                                columns=['score'],
                                )
    df.to_csv(FilePath.classification_evaluation_path)

    #Note:
    #Refer to the demo_pipeline_1 notebook for the data analysis and exploration,
    #feature importance has been applied, but the accuracy has not been improved yet,
    #the hyperparameter tuning has also been applied, but the accuracy has not been improved as well,
    #so the issue is mostly indicated to be the data itself, not the model,
    #due to the time limitation, I will not be able to explore more on the data,
    #but below are few plans moving forward that might improve the accuracy
    #+ Analyze the dataset deeper and extract the more meaningful features
    #+ Apply feature importance again to just select the most important features, focus on these, inspect the data quality, extract more insights if possible from these features
    #+ Collect external data sources if possible (weather, people-related data such as team members info, etc)

    #Save the model
    model_path = FilePath.classifier_model_path
    model.save_model(model_path)
    # Check if the model file is saved
    if os.path.exists(model_path):
        print(f"Model saved successfully to path: {FilePath.classifier_model_path}")
    else:
        print("Failed to save the model")

if __name__ == "__main__":
    train_model()

