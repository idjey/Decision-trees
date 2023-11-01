import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, save_data, load_model, save_model, get_file_path



def load_and_fill_data(filepath):
    """Load dataset from CSV file and handle NaN values."""
    df = pd.read_csv(filepath)

    # Fill NaN values for numeric columns with their mean
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        df[col].fillna(df[col].mean(), inplace=True)

    # Fill NaN values for non-numeric columns with their mode
    non_num_cols = df.select_dtypes(exclude=['float64', 'int64']).columns
    for col in non_num_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df



def train_decision_tree(X_train, y_train):
    """Train a Decision Tree classifier and return the trained model."""
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model on test data and print metrics."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy: .4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Plot confusion matrix (optional)
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


def save_model(model, filename):
    """Save the trained model to disk using joblib."""
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")


def main():
    # Load training and testing datasets
    X_train = load_and_fill_data(
        r"C:\Users\Dj_ka\Documents\Decision trees\dataset\processed\X_train.csv")
    y_train = load_and_fill_data(
        r"C:\Users\Dj_ka\Documents\Decision trees\dataset\processed\y_train.csv")
    X_test = load_and_fill_data(
        r"C:\Users\Dj_ka\Documents\Decision trees\dataset\processed\X_test.csv")
    y_test = load_and_fill_data(
        r"C:\Users\Dj_ka\Documents\Decision trees\dataset\processed\y_test.csv")

    # Train the model
    model = train_decision_tree(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Save the model
    save_model(model, r"C:\Users\Dj_ka\Documents\Decision trees\models\decision_tree_model.pkl")


if __name__ == "__main__":
    main()
