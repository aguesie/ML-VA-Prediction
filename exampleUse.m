clear
close all

%% -------------------------- LSBOOST PREDICTION --------------------------

% Load data
load('data/XTest_9z_AA.mat');
load('data/yTest_9z_AA.mat');

yPred_LSBoost = pred_LSBoost(XTest, 'models/modelLSBoost.mat', 'Mdl', false);
yPred_LSBoost_filt = removeOutliers(yTest, yPred_LSBoost);


%% -------------------------- XGBOOST PREDICTION --------------------------

% Load data
X_test = 'data/XTest_5z.mat';
y_test = 'data/yTest_5z.mat';
load('data/yTest_5z.mat');

% Prepare python environment
jsonText = fileread('PythonConfig.json');
jsonData = jsondecode(jsonText);

pythonPath = jsonData.PythonExecPath;

pythonSetUp(pythonPath);

command = sprintf('python "%s" --X_test "%s" --model "%s"', 'pred_XGBoost.py', X_test, 'models/modelXGBoost.json');
system(command);

load('results/y_predXGBoost.mat')  % Load the prediction that was just made
yPred_XGBoost_filt = removeOutliers(yTest, y_pred);


%% ---------------------- NEURAL NETWORK PREDICTION ----------------------

% Load data
load('data/ZernForNet.mat');
load('data/agesForNet.mat');

% Make prediction
yPred = pred_NeuralNetwork(3, 0.01, 1, Z, ages, 'models/final_net.mat', 2);
