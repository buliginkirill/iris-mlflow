import argparse, os, mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def main(n_estimators: int, max_depth: int):
    mlflow.autolog(log_models=True)

    with mlflow.start_run() as run:
        X, y = load_iris(return_X_y=True, as_frame=True)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42)

        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        clf.fit(X_tr, y_tr)

        # ваши метрики для test-сета
        mlflow.log_metric("test_accuracy",
                          accuracy_score(y_te, clf.predict(X_te)))
        mlflow.log_metric("test_f1_macro",
                          f1_score(y_te, clf.predict(X_te), average="macro"))

        # регистрируем именно эту модель (ONE run, ONE model)
        mlflow.register_model(f"runs:/{run.info.run_id}/model", "iris_rf")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_estimators", type=int,
                   default=int(os.getenv("N_ESTIMATORS", 100)))
    p.add_argument("--max_depth", type=int,
                   default=int(os.getenv("MAX_DEPTH", 5)))
    main(**vars(p.parse_args()))
