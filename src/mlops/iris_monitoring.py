import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

data = load_iris()
X, y = data.data, data.target

mlflow.set_experiment("Iris_Classification")
with mlflow.start_run():
    model = RandomForestClassifier()
    model.fit(X, y)
    mlflow.log_metric("accuracy", model.score(X, y))
    mlflow.sklearn.log_model(model, "model")