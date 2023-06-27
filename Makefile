process-predict:
	@python src/process_predict_data.py

process-train-q1:
	@python src/classification_models/process_train_data.py

train-q1:
	@python src/classification_models/train_model.py

predict-q1:
	@python src/classification_models/predict_samples.py

process-train-q3:
	@python src/regression_models/process_train_data.py

train-home-q3:
	@python src/regression_models/home_train_model.py

train-away-q3:
	@python src/regression_models/away_train_model.py

predict-home-q3:
	@python src/regression_models/home_predict_samples.py

predict-away-q3:
	@python src/regression_models/away_predict_samples.py