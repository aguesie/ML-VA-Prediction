function yPred_filt = removeOutliers(yTest, yPred)
    % remove_outliers - Visualizes prediction errors and removes outliers based on a 3-sigma rule.
    %
    % This function compares predicted visual acuity (VA) values with actual VA values, 
    % computes regression performance metrics, and removes outlier predictions using 
    % a 3-standard-deviation threshold (3σ). It provides before-and-after evaluation 
    % of the model performance and visualizes the prediction quality.
    %
    % Inputs:
    %   yTest     - Vector of true (measured) VA values.
    %   yPred     - Vector of predicted VA values.
    %
    % Output:
    %   yPred_filt - Vector of predicted VA values after removing outliers.

    
    % --------------------- BEFORE REMOVING OUTLIERS ---------------------
    % Calculate absolut errors
    errors = abs(yTest - yPred);
    
    % Plot real values of VA vs predicted values
    figure;
    scatter(yTest, yPred, 30, errors, 'filled');
    xlim([-0.4 1.2]);
    xticks(-0.4:0.2:1.2);
    c = colorbar;
    colormap(jet);
    c.Label.String = 'abs(realVA - predVA)';
    title('Actual vs. Predicted Values of Visual Acuity')
    xlabel('real VA')
    ylabel('predicted VA')
        
    % Calculate MSE
    mse = mean((errors).^2);
    
    % Calculate RMSE
    rmse = sqrt(mse);
    
    % Calculate MAE
    mae = mean(abs(errors));
    
    % Calculate R²
    sst = sum((yTest - mean(yTest)).^2);
    sse = sum((errors).^2);
    r2 = 1 - (sse / sst);
    
    % Find the maximum and minimum error
    max_error = max(errors);
    min_error = min(errors);
    
    % Show results
    fprintf('------------- With outliers -------------\n')
    fprintf('MSE: %.4f\n', mse);
    fprintf('RMSE: %.4f\n', rmse);
    fprintf('MAE: %.4f\n', mae);
    fprintf('R2: %.4f\n', r2);
    fprintf('Max error: %.4f\n', max_error);
    fprintf('Min error: %.4f\n', min_error);
    fprintf('Mode: %.4f\n', mode(errors));
    fprintf('Median: %.4f\n', median(errors));


    % -------------------------- REMOVE OUTLIERS --------------------------
    threshold = std(errors)*3;  % 3 sigma
    logic = errors<threshold;
    yPred_filt = yPred(logic == 1,:);
    yTest_filt = yTest(logic == 1,:);


    % ---------------------- AFTER REMOVING OUTLIERS ----------------------
    % Calculate absolute errors
    errors_filt = abs(yTest_filt - yPred_filt);
    
    % Calculate MSE
    mse = mean((errors_filt).^2);
    
    % Calculate RMSE
    rmse = sqrt(mse);
    
    % Calculate MAE
    mae = mean(abs(errors_filt));
    
    % Calculate R²
    sst = sum((yTest_filt - mean(yTest_filt)).^2);
    sse = sum((errors_filt).^2);
    r2 = 1 - (sse / sst);
    
    % Find the maximum and minimum error
    max_error = max(errors_filt);
    min_error = min(errors_filt);
    
    % Show results
    fprintf('------------- Without outliers -------------\n')
    fprintf('MSE: %.4f\n', mse);
    fprintf('RMSE: %.4f\n', rmse);
    fprintf('MAE: %.4f\n', mae);
    fprintf('R2: %.4f\n', r2);
    fprintf('Max error: %.4f\n', max_error);
    fprintf('Min error: %.4f\n', min_error);
    fprintf('Mode: %.4f\n', mode(errors_filt));
    fprintf('Median: %.4f\n', median(errors_filt));
    
    % Plot real values of VA vs predicted values after removing ouliers
    figure;
    scatter(yTest_filt, yPred_filt, 30, errors_filt, 'filled');
    xlim([-0.4 1.2]);
    xticks(-0.4:0.2:1.2);
    c = colorbar;
    colormap(jet);
    c.Label.String = 'abs(realVA - predVA)';
    title('Actual vs. Predicted Values of Visual Acuity without outliers')
    xlabel('real VA')
    ylabel('predicted VA')
    
end