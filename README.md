# Visual Acuity Prediction Models

This project aims to **predict a subject’s visual acuity (VA) in logMAR of a subject or subjects from their Zernike coefficients** using three different machine learning approaches:

- **A regression model based on LSBoost** implemented in MATLAB
- **A regression model using XGBoost** implemented in python
- **A convolutional neural network model** implemented in MATLAB

The LSBoost and XGBoost models are trained to directly predict VA from Zernike coefficients, age and, only in the first one, amplitude of accommodation. In contrast, the convolutional neural network model simulates the visual perception of optotypes of the subject and estimates the VA through classification performance.

The LSBoost and XGBoost models can be executed on both Windows and Linux platforms. However, the convolutional neural network model only works on Windows, since it relies on renderTextFT.mexw64, a precompiled MEX function specific to Windows, to render optotype images.


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

You can also run the first example directly from the script `exampleUse.m`, which includes a ready-to-run section demonstrating how to call the function without metrics.

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

### Setting up the environment and usage
All required packages are listed in the provided `ML-VA-prediction.yml` file. To create the environment:

`conda env create -f ML-VA-prediction.yml`

#### How to Run the Script
- **From Python (for example, via terminal or a script):**

  Make sure the Conda environment is activated:
  
  `conda activate ML-VA-prediction`
  
  Then run the script with your data:
  
  `python pred_XGBoost.py --X_test data/XTest_5z.mat --model models/xgboost_model.json --metrics --y_test data/yTest_5z.mat`

- **From MATLAB (using Python integration):**

  To run the script from MATLAB using Python, start by specifying the Python path from Conda environment in the `PythonConfig.json` file:
  
  `{"PythonExecPath": "C:/Users/your_username/anaconda3/envs/your_env_name/"}`

  Then, in MATLAB, use the provided `pythonSetUp.m` function to configure the environment automatically:

  ```
  jsonText = fileread('PythonConfig.json');
  jsonData = jsondecode(jsonText);
  pythonPath = jsonData.PythonExecPath;
  
  pythonSetUp(pythonPath);
  ```
  This function will automatically detect your OS (Windows/Linux) and set the Python environment via `pyenv`.

  You can find a complete example of how to configure the Python environment and call `pred_XGBoost.py` from MATLAB in the script `exampleUse.m`.


## Convolutional Neural Network Method
The `pred_NeuralNetwork.m` script uses a pre-trained convolutional neural network model to estimate VA in logMAR units based on pupil diameter, noise variance, contrast sensitivity scaling, Zernike coefficients, and age.

### Description
This function estimates VA by simulating the recognition of optotypes for each subject. For each VA level (from logMAR 1.0 to -0.3 with 5 optotypes in each one), it calls the helper function `generate_optotype.m` to create aberrated optotype images based on Zernike coefficients, pupil size, noise variance, age, and contrast sensitivity parameters.
These images are then classified by the network model as "recognized" or "not recognized". The number of misclassifications per level is counted, and if all 5 optotypes at a given level are misclassified, the simulation stops.
The final predicted VA for the subject is calculated by taking the last tested VA level and adding an incremental correction of 0.02 logMAR units for each optotype that was misclassified across all tested levels. This refinement allows a more precise VA estimate than just the last completely unreadable level.
The function returns a vector containing the predicted VA (in logMAR) for each subject processed.

The inputs of the function are:

| Argument       | Description                                                                        |
| -------------- | -----------------------------------------------------------------------------------|
| `Dpup`         | Scalar indicating the pupil diameter in millimeters.                               |
| `var_noise`    | Scalar defining the variance of the noise to simulate.                             |
| `phi`          | Scaling parameter for the contrast sensitivity function.                           |
| `Z`            | Matrix (N x 36) of Zernike coefficients for each subject (N = number of subjects). |
| `list_age`     | Vector of ages corresponding to each subject (rows of Z).                          |
| `modelPath`    | String path to the .mat file containing the pre-trained neural network model.      |
| `num_template` | Integer (1 or 2) selecting the optotype template used for simulation.              |


### Example of use
`yPred = pred_NeuralNetwork(3, 0.01, 1, Z, ages, 'models/final_net.mat', 2);`

You can also run this example directly from one of the cells in the `exampleUse.m` script, which loads a dataset of Zernike coefficients and ages for 5 subjects, providing a ready-to-run demonstration of the function.

**Requirement:** this function requires MATLAB with the *Deep Learning Toolbox* and access to the optotype templates Excel file (`data/Optotipos_usados.xlsx`) in the appropriate location.

### About `generate_optotype`
The helper function `generate_optotype.m` creates simulated optotype images using the subject’s Zernike coefficients (wavefront aberrations), pupil diameter, noise variance, age, and a contrast sensitivity scaling factor. The function:

- Constructs the eye’s wavefront and computes its point spread function (PSF).
- Renders the optotype character at the specified VA level.
- Convolves the optotype with the PSF to simulate optical degradation.
- Simulates retinal cone sampling and applies neural filtering using a spatial contrast sensitivity function adjusted for age.
- Adds spatial Gaussian noise to the final image.

The output is an aberrated optotype image that simulates how the optotype would appear to the subject’s visual system.
  
## Optional Post-Processing: Outlier Removal
After obtaining predictions from either the LSBoost or XGBoost models, the `removeOutliers.m` function can be used to visualize prediction errors and remove outliers based on a 3-sigma rule.

### Description
The function compares predicted VA values (`yPred`) with ground truth measurements (`yTest`), plots the prediction error distribution, and removes samples with errors beyond 3 standard deviations. It prints regression metrics (MSE, RMSE, MAE, R²) before and after outlier removal, and visualizes the change in prediction quality.
The function returns a filtered vector of predicted VA values (`yPred_filt`) with the outliers excluded.

The inputs of the funtion are:

| Argument       | Description                                         |
| -------------- | ----------------------------------------------------|
| `yTest`        | Vector of true (measured) VA values.                |
| `yPred`        | Vector of predicted VA values.                      |


### Example of use
`yPred_filt = removeOutliers(yTest, yPred);`

It is also called in the `exampleUse.m` script after generating predictions with both the LSBoost and XGBoost models, providing ready-to-run examples of its application.
