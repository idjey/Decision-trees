import pandas as pd
import joblib
from preprocess import load_dataset, encode_data


def add_missing_columns(df, columns):
    """Add missing columns to the DataFrame with default value 0."""
    for col in columns:
        if col not in df.columns:
            df[col] = 0
    return df


def map_sdtm_variables(input_file_path, output_file_path, model_path=r'C:\Users\Dj_ka\Documents\Decision trees\models\decision_tree_model.pkl'):
    """
    Function to map original variables to SDTM variables.
    """
    # Load the input dataset
    data = load_dataset(input_file_path)

    # Print columns for diagnosis
    print("Available Columns:", data.columns)

    # Load the trained Decision Tree model
    model = joblib.load(model_path)

    # List of features used in the trained model
    trained_features = [
        feature for feature in model.feature_importances_ if feature > 0]

    # Prepare data for prediction: Ensure it has the same columns as the training features
    prediction_data = data.copy()

    # Encode the data
    prediction_data = encode_data(prediction_data)

    # Ensure prediction data has the same columns as the training features
    prediction_data = add_missing_columns(prediction_data, trained_features)

    # Keeping only the columns used during training
    prediction_data = prediction_data[trained_features]

    # Make predictions using the model
    predictions = model.predict(prediction_data)

    # Add the predictions to the original data
    data['SDTM Variable'] = predictions

    # Save the mapped data to a new CSV file
    data.to_csv(output_file_path, index=False)

    print(f"Mapping complete! Results saved to {output_file_path}")


if __name__ == "__main__":
    # Paths for input and output can be modified as needed
    input_file = r'C:\Users\Dj_ka\Documents\Decision trees\dataset\processed\test_file.csv'
    output_file = r'C:\Users\Dj_ka\Documents\Decision trees\dataset\mapped_output\mapped_data.csv'
    map_sdtm_variables(input_file, output_file)
