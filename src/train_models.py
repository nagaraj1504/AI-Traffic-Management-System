import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor

def train_models():
    df = pd.read_csv("data/final_dataset.csv")

    # =============================
    # REGRESSION MODEL (Traffic Volume)
    # =============================

    X = df.drop("traffic_volume", axis=1)
    y = df["traffic_volume"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(),
        "GradientBoosting": GradientBoostingRegressor(),
        "DecisionTree": DecisionTreeRegressor()
    }

    best_model = None
    best_score = 0
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = r2_score(y_test, preds)
        print(f"{name} R2 Score: {score}")
        results.append((name, score))

        if score > best_score:
            best_score = score
            best_model = model

    # Save best regression model
    pickle.dump(best_model, open("models/best_model.pkl", "wb"))

    # Save model comparison results
    pd.DataFrame(results, columns=["Model", "R2"]).to_csv("models/model_scores.csv", index=False)

    print("✅ Best regression model saved")

    # =============================
    # CLASSIFICATION MODEL (Congestion)
    # =============================

    df['congestion'] = df['traffic_volume'].apply(
        lambda x: 0 if x < 2000 else 1 if x < 4000 else 2
    )

    X_class = df.drop(['traffic_volume', 'congestion'], axis=1)
    y_class = df['congestion']

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_class, y_class, test_size=0.2)

    clf = RandomForestClassifier()
    clf.fit(Xc_train, yc_train)

    preds_class = clf.predict(Xc_test)
    acc = accuracy_score(yc_test, preds_class)
    print("Congestion Classifier Accuracy:", acc)

    pickle.dump(clf, open("models/congestion_model.pkl", "wb"))

    print("✅ Congestion model saved")

if __name__ == "__main__":
    train_models()