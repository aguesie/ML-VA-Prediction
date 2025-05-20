function yPred = pred_LSBoost(XTest, modelMat, variableName, metrics, yTest)
    % pred_LSBoost - Predict using a pre-trained LSBoost model and optionally evaluate performance metrics.
    %
    % Syntax:
    %   When metrics is false (no evaluation metrics calculated):
    %   yPred = pred_LSBoost(XTest, modelMat, variableName, false)
    %
    %   When metrics is true (evaluation metrics will be calculated):
    %   yPred = pred_LSBoost(XTest, modelMat, variableName, true, yTest)
    %
    % Inputs:
    %   XTest        - Test feature matrix.
    %   modelMat     - Path to the .mat file containing the trained model.
    %   variableName - Name of the variable inside the .mat file representing the model.
    %   metrics      - Boolean flag. If true, the function calculates and prints performance metrics.
    %   yTest        - (Required if metrics is true) True target values for computing the evaluation metrics.
    %
    % Outputs:
    %   yPred        - Predicted values, rounded to the nearest 0.02 increment.
    %
    % Description:
    %   This function loads a pre-trained LSBoost model, makes predictions on the provided test data,
    %   and optionally calculates common regression evaluation metrics such as MSE, RMSE, MAE, R2,
    %   max/min error, mode, and median of absolute errors.
    %
    % Note:
    %   If 'metrics' is set to true, 'yTest' must be provided. Otherwise, an error is raised.

    rng(3);  % For reproducibility

    % --------------------- LOAD MODEL ONLY IF NEEDED ---------------------
    persistent cachedModel cachedFile cachedVarName

    if isempty(cachedModel) || ~strcmp(cachedFile, modelMat) || ~strcmp(cachedVarName, variableName)
        fprintf('Model cached \n')
        loadedData = load(modelMat, variableName);
        cachedModel = loadedData.(variableName);
        cachedFile = modelMat;
        cachedVarName = variableName;
    end

    model = cachedModel;

    % -------------------------- MAKE PREDICTION --------------------------
    prediction = predict(model, XTest);
    yPred = round(prediction / 0.02) * 0.02;

    % ---------------------- CALCULATE METRICS (OPTIONAL) -----------------
    if metrics
        if nargin < 5
            error('If "metrics" is true, you must provide "yTest".');
        end

        errors = abs(yTest - yPred);
        mse = mean(errors.^2);
        rmse = sqrt(mse);
        mae = mean(errors);
        sst = sum((yTest - mean(yTest)).^2);
        sse = sum(errors.^2);
        r2 = 1 - (sse / sst);
        max_error = max(errors);
        min_error = min(errors);

        fprintf('MSE: %.4f\n', mse);
        fprintf('RMSE: %.4f\n', rmse);
        fprintf('MAE: %.4f\n', mae);
        fprintf('R2: %.4f\n', r2);
        fprintf('Max error: %.4f\n', max_error);
        fprintf('Min error: %.4f\n', min_error);
        fprintf('Mode: %.4f\n', mode(errors));
        fprintf('Median: %.4f\n', median(errors));
    end
end