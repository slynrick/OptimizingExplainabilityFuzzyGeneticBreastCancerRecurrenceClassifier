from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import pandas as pd 
# Load the dataset

train_dataset = pd.read_csv("..\dataset\dadoscancer_2classes-train.csv")
test_dataset = pd.read_csv("..\dataset\dadoscancer_2classes-test.csv")

X_train_ = train_dataset.drop(labels='Class',axis=1).values 
y_train = train_dataset['Class'].values 

X_test_ = test_dataset.drop(labels='Class',axis=1).values 
y_test = test_dataset['Class'].values 

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_)
X_test = scaler.transform(X_test_)

# Define a list of classifiers to evaluate
classifiers = [
    {
        "name": "Random Forest",
        "classifier": RandomForestClassifier(),
        "params": {
            "n_estimators": [10, 50, 100],
            "max_depth": [2, 5, 10]
        }
    },
    {
        "name": "Logistic Regression",
        "classifier": LogisticRegression(),
        "params": {
            "penalty": ["l1", "l2"],
            "C": [0.1, 1, 10]
        }
    },
    {
        "name": "Support Vector Machine",
        "classifier": SVC(),
        "params": {
            "kernel": ["linear", "rbf", "poly"],
            "C": [0.1, 1, 10]
        }
    },
    {
        "name": "Naive Bayes",
        "classifier": GaussianNB(),
        "params": {
            "var_smoothing": [1e-9, 1e-7, 1e-5]
        }
    },
    {
        "name": "K-Nearest Neighbors",
        "classifier": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"]
        }
    },
    {
        "name": "Neural Network",
        "classifier": MLPClassifier(),
        "params": {
            "hidden_layer_sizes": [(10,), (50,), (100,)],
            "activation": ["logistic", "relu"],
            "alpha": [0.0001, 0.001, 0.01]
        }
    },
    {
        "name": "CatBoost",
        "classifier": CatBoostClassifier(),
        "params": {
            "iterations": [10, 50, 100],
            "learning_rate": [0.01, 0.1, 1],
            "depth": [2, 5, 10]
        }
    },
    {
        "name": "XGBoost",
        "classifier": XGBClassifier(),
        "params": {
            "n_estimators": [10, 50, 100],
            "learning_rate": [0.01, 0.1, 1],
            "max_depth": [2, 5, 10]
        }
    }
]

# Define a dictionary to hold the results
results = {}

# Loop through each classifier and perform a grid search
for clf in classifiers:
    print(f"Evaluating {clf['name']}")
    grid_search = GridSearchCV(clf["classifier"], clf["params"], cv=5)
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    best_params = grid_search.best_params_
    results[clf["name"]] = {"accuracy": accuracy, "precision": precision, "recall": recall, "best_params": best_params}

# Print the results
for name, result in results.items():
    print(f"{name} & {round(result['accuracy'],4)} & {round(result['precision'],4)} & {round(result['recall'],4)} \n {name} Best Parameters={result['best_params']}")
