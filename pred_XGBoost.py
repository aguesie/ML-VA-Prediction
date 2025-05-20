import argparse
import scipy.io
import xgboost as xgb
import numpy as np
from scipy.stats import mode
import random
import pickle
import os

# Set random seed for reproducibility
random.seed(10)

# Path to the cache file
CACHE_FILE = 'model_cache.pkl'


def load_data(file_path, var_name):
    """
    Load a variable from a .mat file.

    Parameters:
    file_path (str): Path to the .mat file.
    var_name (str): Name of the variable inside the .mat file.

    Returns:
    numpy.ndarray: Loaded data array.
    """
    return scipy.io.loadmat(file_path)[var_name]


def evaluate_model(y_true, y_pred):
    """
    Compute and display evaluation metrics for model predictions.

    Parameters:
    y_true (numpy.ndarray): Ground truth target values.
    y_pred (numpy.ndarray): Predicted target values.

    Prints:
    - MSE (Mean Squared Error)
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - R2 (Coefficient of Determination)
    - Maximum and minimum prediction error
    - Mode and median of absolute errors
    """
    errors = np.round(np.abs(y_true - y_pred), 2)
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))

    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    sse = np.sum(errors ** 2)
    r2 = 1 - (sse / sst)

    max_error = np.max(errors)
    min_error = np.min(errors)
    mode_error = mode(errors).mode[0]
    median_error = np.median(errors)

    print("------------- Metrics in test -------------")
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2:", r2)
    print("Max error:", max_error)
    print("Min error:", min_error)
    print("Mode:", mode_error)
    print("Median:", median_error)


def load_model(model_file):
    """
    Load the model from a file. If the model is already cached, load it from the cache.

    Parameters:
    model_file (str): Path to the model file (.json).

    Returns:
    model: The loaded XGBoost model.
    """
    # Verificar si el caché existe y tiene un modelo
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            cached_data = pickle.load(f)

        # Verificar si el modelo en caché coincide con el nombre del archivo actual
        cached_model_file = cached_data['model_file']
        if cached_model_file == model_file:
            print("Model loaded from cache.")
            return cached_data['model']
        else:
            print(f"Model file has changed. Clearing cache and loading model from disk: {model_file}")

    # Si no está en caché o si el nombre del modelo ha cambiado, cargar el modelo desde el archivo
    model = xgb.XGBRegressor()
    model.load_model(model_file)

    # Guardar el nuevo modelo en caché junto con el nombre del archivo
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump({'model': model, 'model_file': model_file}, f)

    print("Model loaded from disk and cached.")
    return model


def main(args):
    """
    Main function to load and evaluate a pre-trained XGBoost model.

    Parameters:
    args (argparse.Namespace): Parsed command-line arguments containing paths to testing data and model-saving flag.

    Actions:
    - Loads testing data from .mat files.
    - Loads a pre-trained XGBoost regressor model, using cache if available.
    - Makes predictions and optionally evaluates model performance.
    """
    X_test = load_data(args.X_test, 'XTest')

    # Load model from cache or disk
    model = load_model(args.model)

    # Make prediction
    y_pred = model.predict(X_test).reshape(-1, 1)
    y_pred = np.round(np.round(y_pred / 0.02) * 0.02, 2)

    # Save predictions to .mat file to use in MATLAB
    #scipy.io.savemat("results/y_predXGBoost.mat", {"y_pred": y_pred})

    # Optionally evaluate metrics
    if args.metrics:
        if not args.y_test:
            raise ValueError("y_test is required when --metrics is set.")
        y_test = load_data(args.y_test, 'yTest')
        evaluate_model(y_test, y_pred)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Load and optionally evaluate a pre-trained XGBoost model on input .mat files")
    parser.add_argument('--X_test', required=True, help="Path to XTest .mat file")
    parser.add_argument('--model', required=True, help="Path to a pre-trained XGBoost model (.json)")
    parser.add_argument('--metrics', action='store_true', help="If set, evaluate and print metrics")
    parser.add_argument('--y_test', help="Path to yTest .mat file (required if --metrics is set)")

    args = parser.parse_args()

    if args.metrics and not args.y_test:
        parser.print_help()
        raise ValueError("y_test is required if --metrics flag is set.")

    main(args)
