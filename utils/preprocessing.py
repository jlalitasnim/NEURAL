import numpy as np
import pandas as pd


def load_csv(filepath):
    return pd.read_csv(filepath)


def train_test_split(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)

    n = X.shape[0]
    indices = np.random.permutation(n)
    split_index = int(n * (1 - test_size))

    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def normalize(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    return (X_train - mean) / std, (X_test - mean) / std


def one_hot_encode(y, n_classes):
    y = y.astype(int).flatten()

    encoded = np.zeros((len(y), n_classes))
    encoded[np.arange(len(y)), y] = 1

    return encoded


def prepare_data(df, target_col, problem_type, test_size=0.2):
    df = df.copy()

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' does not exist.")

    df = df.dropna(subset=[target_col])

    if df.empty:
        raise ValueError("Dataset is empty after removing rows with missing target values.")

    X_df = df.drop(columns=[target_col]).copy()
    y_series = df[target_col].copy()

    # Clean and encode input features
    for col in X_df.columns:
        numeric_col = pd.to_numeric(X_df[col], errors="coerce")
        original_non_missing = X_df[col].notna().sum()
        numeric_non_missing = numeric_col.notna().sum()

        if original_non_missing > 0 and numeric_non_missing == original_non_missing:
            mean_value = numeric_col.mean()
            if pd.isna(mean_value):
                mean_value = 0.0
            X_df[col] = numeric_col.fillna(mean_value)
        else:
            X_df[col] = X_df[col].astype(str)
            X_df[col] = X_df[col].replace(["nan", "None", "NaN"], "Unknown")
            X_df[col] = X_df[col].fillna("Unknown")

    X_df = pd.get_dummies(X_df, drop_first=False)

    if X_df.shape[1] == 0:
        raise ValueError("No usable input features found after preprocessing.")

    X = X_df.values.astype(float)

    # Encode target
    if problem_type == "binary":
        y_encoded, classes = pd.factorize(y_series)

        if len(classes) != 2:
            raise ValueError(
                f"Binary classification requires exactly 2 classes. "
                f"Found {len(classes)} classes: {list(classes)}"
            )

        y = y_encoded.reshape(-1, 1).astype(float)
        n_classes = 2

    elif problem_type == "multiclass":
        y_encoded, classes = pd.factorize(y_series)
        n_classes = len(classes)

        if n_classes < 2:
            raise ValueError("Multiclass classification requires at least 2 classes.")

        y = one_hot_encode(y_encoded, n_classes)

    elif problem_type == "regression":
        y_numeric = pd.to_numeric(y_series, errors="coerce")

        if y_numeric.isna().any():
            raise ValueError(
                "Regression requires a numeric target column. "
                "Your selected target contains text values. "
                "For Abalone, choose 'Rings' as the target, not 'Type'."
            )

        y = y_numeric.values.reshape(-1, 1).astype(float)
        n_classes = 1

    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size)
    X_train, X_test = normalize(X_train, X_test)

    return X_train, X_test, y_train, y_test, n_classes