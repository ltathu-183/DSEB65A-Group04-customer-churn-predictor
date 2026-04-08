import argparse
import pandas as pd
import joblib
import yaml
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.features.engineer import create_preprocessor, FeatureEngineer


def main(args):
   # Load data
   df = pd.read_csv(args.data)

   target = "Churn"

   X = df.drop(columns=[target])
   y = df[target]

   # Split
   X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

   # Pipeline
   pipeline = Pipeline([
   ("feature_engineering", FeatureEngineer()),
   ("preprocessor", create_preprocessor()),
   ("model", RandomForestClassifier(random_state=42))
])

   # Hyperparameter space
   param_dist = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [5, 10, 15, None],
        "model__min_samples_split": [2, 5, 10]
    }

   # MLflow
   mlflow.set_experiment("churn_model")

   with mlflow.start_run():

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=10,
            scoring="f1",
            cv=3,
            n_jobs=-1,
            verbose=1
        )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_

        # Predict
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred) 
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)  # default = binary
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        # Log MLflow
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics({
         "accuracy": acc,
         "precision": precision,
         "recall": recall,
         "f1": f1,
         "f1_macro": f1_macro,
         "f1_weighted": f1_weighted
      })

        mlflow.sklearn.log_model(best_model, "model")

        # Save model
        model_path = f"{args.model_dir}/model.pkl"
        joblib.dump(best_model, model_path)

        # Save config YAML
        config = {
            "model": {
                "name": "churn_model",
                "target_variable": "Churn",
                "best_model": "RandomForestClassifier",
                "parameters": search.best_params_
            }
        }

        with open(args.config_out, "w") as f:
            yaml.dump(config, f)

        print("Training complete")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 (binary): {f1:.4f}")
        print(f"F1 (macro): {f1_macro:.4f}")
        print(f"F1 (weighted): {f1_weighted:.4f}")


if __name__ == "__main__":
   parser = argparse.ArgumentParser()

   parser.add_argument("--data", required=True)
   parser.add_argument("--model_dir", required=True)
   parser.add_argument("--config_out", required=True)

   args = parser.parse_args()

   main(args)