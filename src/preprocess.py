# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder


# def load_data(filepath):
#     """
#     Load the dataset from the specified filepath.
#     """
#     return pd.read_csv(filepath)


# def handle_missing_values(data):
#     """
#     Handle missing values in the dataset.
#     For this example, we'll fill missing values with the median for numerical columns
#     and the mode for categorical columns.
#     """
#     for col in data.columns:
#         if data[col].dtype == 'O':  # Categorical columns
#             mode_val = data[col].mode()[0]
#             data[col].fillna(mode_val, inplace=True)
#         else:
#             median_val = data[col].median()
#             data[col].fillna(median_val, inplace=True)
#     return data


# def encode_categorical_variables(data):
#     """
#     Convert categorical variables into numerical format.
#     """
#     label_encoders = {}
#     for col in data.columns:
#         if data[col].dtype == 'O':  # Categorical columns
#             le = LabelEncoder()
#             data[col] = le.fit_transform(data[col])
#             label_encoders[col] = le
#     return data, label_encoders


# def split_dataset(data, target_col, test_size=0.2, random_state=42):
#     """
#     Split the dataset into training and test sets.
#     """
#     X = data.drop(target_col, axis=1)
#     y = data[target_col]
#     return train_test_split(X, y, test_size=test_size, random_state=random_state)


# def main():
#     # Load data
#     data = load_data(r'C:\Users\Dj_ka\Documents\Mapping\Decision trees\dataset\map_data.csv')

#     # Handle missing values
#     data = handle_missing_values(data)

#     # Encode categorical variables
#     data, encoders = encode_categorical_variables(data)

#     # Save the processed data
#     data.to_csv(r'C:\Users\Dj_ka\Documents\Mapping\Decision trees\dataset\processed\processed_data.csv', index=False)

#     # Split the dataset into training and test sets
#     X_train, X_test, y_train, y_test = split_dataset(data, 'SDTM variable')
#     X_train.to_csv(r'C:\Users\Dj_ka\Documents\Mapping\Decision trees\dataset\processed\train.csv', index=False)
#     X_test.to_csv(r'C:\Users\Dj_ka\Documents\Mapping\Decision trees\dataset\processed\test.csv', index=False)


# if __name__ == '__main__':
#     main()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import load_data, save_data, load_model, save_model, get_file_path



def load_dataset(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)


def split_dataset(data, target_column_name):
    """Split dataset into training and testing datasets."""
    X = data.drop(target_column_name, axis=1)
    y = data[target_column_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def encode_data(train_data, test_data):
    """Encode categorical data for training and testing datasets."""
    label_encoders = {}
    for column in train_data.columns:
        le = LabelEncoder()
        le.fit(pd.concat([train_data[column], test_data[column]]))
        train_data[column] = le.transform(train_data[column])
        test_data[column] = le.transform(test_data[column])
        label_encoders[column] = le
    return train_data, test_data, label_encoders


def main():
    # Update this with your actual file path
    filepath = r'C:\Users\Dj_ka\Documents\Mapping\Decision trees\dataset\map_data.csv'
    data = load_dataset(filepath)

    # Split the dataset
    X_train, X_test, y_train, y_test = split_dataset(data, 'SDTM variable')

    # Encode data
    X_train, X_test, label_encoders = encode_data(X_train, X_test)

    # Save processed data (optional)
    X_train.to_csv(
        r'C:\Users\Dj_ka\Documents\Mapping\Decision trees\dataset\processed\X_train.csv', index=False)
    y_train.to_csv(
        r'C:\Users\Dj_ka\Documents\Mapping\Decision trees\dataset\processed\y_train.csv', index=False)
    X_test.to_csv(
        r'C:\Users\Dj_ka\Documents\Mapping\Decision trees\dataset\processed\X_test.csv', index=False)
    y_test.to_csv(
        r'C:\Users\Dj_ka\Documents\Mapping\Decision trees\dataset\processed\y_test.csv', index=False)

    print("Preprocessing completed and datasets saved!")


if __name__ == "__main__":
    main()
