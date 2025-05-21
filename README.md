# Visual Acuity Prediction Models

This project aims to **predict a subject’s visual acuity (VA) in logMAR of a subject or subjects from their Zernike coefficients** using three different machine learning approaches:

- **A regression model based on LSBoost** implemented in MATLAB
- **A regression model using XGBoost** implemented in python
- **A convolutional neural network model** implemented in MATLAB

The LSBoost and XGBoost models are trained to directly predict VA from Zernike coefficients, age and, only in the first one, amplitude of accommodation. In contrast, the convolutional neural network model simulates the visual perception of optotypes of the subject and estimates the VA through classification performance.

## LSBoost Method

The `pred_LSBoost.m` script uses a pre-trained LSBoost regression model to predict the VA of one or multiple subjects based on: Zernike coefficients (OSA index 1 to 9), age, and amplitude of accommodation.

### Description
This function makes predictions on a test set by loading a `.mat` file containing a pre-trained LSBoost model and applying it to the data. It returns a vector of predicted VA values in logMAR.
It can optionally calculate performance metrics such as MSE, RMSE, MAE, and R² if ground truth values are provided. 
The model is internally cached for efficiency when the function is called repeatedly with the same file. This avoids reloading the .mat file each time and improves performance during batch evaluations.

The inputs of the function are:

| Argument       | Description                                                               |
| -------------- | ------------------------------------------------------------------------- |
| `XTest`        | Matrix of input features (e.g., Zernike coefficients, age, amplitude).    |
| `modelMat`     | Path to the `.mat` file containing the trained LSBoost model.             |
| `variableName` | Name of the variable inside the `.mat` file that stores the model.        |
| `metrics`      | Boolean. If `true`, evaluation metrics are printed.                       |
| `yTest`        | *(Required if metrics = true)* True target values for metric calculation. |


### Example of use

When metrics is false (no evaluation metrics calculated):

`yPred = pred_LSBoost(XTest, 'models/modelLSBoost.mat', 'Mdl', false);`

When metrics is true (evaluation metrics will be calculated):

`yPred = pred_LSBoost(XTest, 'models/modelLSBoost.mat', 'Mdl', true, yTest);`

**Requirement:** This script requires the *Statistics and Machine Learning Toolbox* in MATLAB.

## XGBoost Method
The `xgboost_predict.py` script uses a pre-trained XGBoost regression model to predict the VA of one or multiple subjects based on: Zernike coefficients (OSA index 1 to 5) and age.

### Description
This script loads a `.json` file containing a pre-trained XGBoost model and applies it to a test dataset stored in `.mat` format. Predictions are automatically saved to the `results/` folder in MATLAB `.mat` format (`y_predXGBoost.mat`), allowing for direct loading and analysis in MATLAB.
It can optionally compute evaluation metrics such as MSE, RMSE, MAE, and R² if ground truth values are provided.
To improve performance during repeated evaluations, the model is cached using Python’s pickle module, so it is only reloaded from disk when necessary.

The inputs of the function are:

| Argument       | Description                                                                        |
| -------------- | -----------------------------------------------------------------------------------|
| `XTest`        | Path to the .mat file containing the input feature matrix (XTest).                 |
| `model`        | Path to the pre-trained XGBoost model file in `.json` format.                      |
| `variableName` | Name of the variable inside the `.mat` file that stores the model.                 |
| `metrics`      | Optional flag. If set, evaluation metrics will be calculated and printed.          |
| `yTest`        | *(Required if `--metrics` is set)* Path to the .mat file with true values (yTest). |


### Example of use
To make predictions without calculating evaluation metrics:

`python pred_XGBoost.py --X_test data/XTest_5z.mat --model models/xgboost_model.json`

To make predictions and calculate evaluation metrics:

`python pred_XGBoost.py --X_test data/XTest_5z.mat --model models/xgboost_model.json --metrics --y_test data/yTest_5z.mat`
