This repo was built on top of WSL2

## Result and code submission are located at notebook and data/predicted folders
## The others are the training and prediction pipelines for classification model
## Install the dependencies: pip install -r requirements.txt
## The data stored in this repo is most likely converted to .feather to save the storage memory
## If the visibility is required, need to convert it back to .csv by using pandas library
## Refer to the notebooks/demo_pipelines/demo_pipeline_1.ipynb for more the data analytics
## How to use
1. Process data used for train classification model:
make process-train-q1

2. Process data used for prediction
make process-predict

3. Train classification model:
make train-q1

4. Predict classification data (home, away, draw)
make predict-q1

5. Process data used for train regression model:
make process-train-q3

6. Train regression model used to predict goals scored by home team:
make train-home-q3

7. Train regression model used to predict goals scored by away team:
make train-away-q3

