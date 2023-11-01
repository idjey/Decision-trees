import pandas as pd
from sklearn.metrics import classification_report
import joblib
import numpy as np
from utils import load_data, save_data, load_model, save_model, get_file_path


def load_data(path):
    """Load dataset from given path."""
    return pd.read_csv(path)


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return classification report."""
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred)


def main():
    MODEL_PATH = r"C:\Users\Dj_ka\Documents\Decision trees\models\decision_tree_model.pkl"
    X_TEST_PATH = r"C:\Users\Dj_ka\Documents\Decision trees\dataset\processed\X_test.csv"
    Y_TEST_PATH = r"C:\Users\Dj_ka\Documents\Decision trees\dataset\processed\y_test.csv"
    REPORT_PATH = r"C:\Users\Dj_ka\Documents\Decision trees\reports\model_performance.txt"

    # Load model, X_test, and y_test data
    model = joblib.load(MODEL_PATH)
    X_test = load_data(X_TEST_PATH)
    # Extract the target column
    y_test = load_data(Y_TEST_PATH)['SDTM variable']

    # Convert y_test to string if it's not
    y_test = y_test.astype(str)

    # Predict and convert y_pred to string
    y_pred = model.predict(X_test)
    y_pred = y_pred.astype(str)

    # Print unique values
    print("Unique values in y_test:", np.unique(y_test))
    print("Unique values in y_pred:", np.unique(y_pred))

    # Evaluate model
    report = classification_report(y_test, y_pred)
    print(report)

    # Save report
    with open(REPORT_PATH, 'w') as file:
        file.write(report)


if __name__ == "__main__":
    main()
