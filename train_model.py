import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
import pickle

from data_loader import feature_cols, target_col


def main():
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")
    X_train = train_data[feature_cols]
    X_test = test_data[feature_cols]
    y_train = train_data[target_col]
    y_test = test_data[target_col]

    mlflow.set_experiment('heart_diseases')

    rf_params = {"n_estimators": 100, "max_depth": 1000}
    gb_params = {"n_estimators": 100, "max_depth": 10}
    log_reg_params = {"max_iter": 1000}
    rf = RandomForestClassifier(**rf_params)
    gb = GradientBoostingClassifier(**gb_params)
    log_reg = LogisticRegression(**log_reg_params)

    models = [rf, gb, log_reg]
    models_params = [rf_params, gb_params, log_reg_params]

    for model, model_params in zip(models, models_params):
        model_name = model.__class__.__name__
        print(f"Running model {model_name}")
        with mlflow.start_run():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            for param_name, param_value in model_params.items():
                mlflow.log_param(param_name, param_value)
            mlflow.log_param("model_type", model_name)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("precision", precision)
            mlflow.sklearn.log_model(model, "model")

            mlflow.log_param('train_size', len(X_train))
            mlflow.log_param('test_size', len(X_test))

            with open(f'models/model_{model_name}.pkl', 'wb') as f:
                pickle.dump(model, f)

            print(f"Recall for model {model_name}: {recall:.4f}")
            print(f"Precision for model {model_name}: {precision:.4f}")

main()