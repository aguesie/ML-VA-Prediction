function VA_pred = pred_NeuralNetwork(Dpup, var_noise, phi, Z, list_age, modelPath, num_template)
    % pred_NeuralNetwork - Estimate visual acuity (logMAR) using a pre-trained neural network model.
    %
    % Inputs:
    %   - Dpup: scalar indicating the pupil diameter in mm
    %   - var_noise: scalar defining the variance of the noise to simulate
    %   - phi: scaling parameter for contrast sensitivity function
    %   - Z: matrix (N x 36) containing the Zernike coefficients for each subject (N = number of subjects)
    %   - list_age: vector of subject ages corresponding to each row of Z
    %   - modelPath: string path to the .mat file that contains the pre-trained neural network model
    %   - num_template - Integer (1 or 2), selects the optotype template to use for the simulation
    %
    % Output:
    %   - VA_pred: column vector (N x 1) of predicted visual acuity values in logMAR for each subject.
    %
    % Description:
    % This function simulates the perceived optotypes for each subject based on their ocular aberrations
    % (Zernike coefficients), age, pupil size and noise. It evaluates optotypes at progressively better
    % VA levels (from logMAR 1.0 to -0.3), % presenting 5 letters per level. For each level, it generates
    % aberrated optotype images based on % the subject’s visual profile, classifies them using the neural
    % network, and counts the number of misclassifications. If all 5 letters at a level are misclassified,
    % that level is considered unreadable, and the loop stops. The final predicted VA is adjusted from the
    % last tested level by adding 0.02 logMAR units per incorrectly identified optotype, refining the estimation.
    %
    % Note:
    % This function creates and deletes a temporary folder to store optotype images during processing.

    rng(0)

    % ------------------------------- LOAD DATA -------------------------------
    % Load the template with the optotype letters and their VA value
    if num_template == 1
        template = readtable('data/Optotipos_usados.xlsx', 'Sheet', 'Plantilla 1');
    elseif num_template == 2
        template = readtable('data/Optotipos_usados.xlsx', 'Sheet', 'Plantilla 2');
    end

    template.VA = str2double(template.VA);

    % Load the model
    modelStruct = load(modelPath);
    fieldNames = fieldnames(modelStruct);
    net = modelStruct.(fieldNames{1});

    num_samples = size(Z, 1);
    VA_pred = zeros(num_samples, 1);

    for i = 1:num_samples
        current_Z = Z(i, :);
        mkdir('tempFolder');
        C = zeros(36, 2);
        C(:, 1) = 0:35;
        C(:, 2) = current_Z';
        age = list_age(i);

        total_failures = 0;

        for va_logmar = 1:-0.1:-0.3
            failures = 0;
            fprintf('Sample %d - testing VA = %.1f\n', i, va_logmar)
            va = 10^(-va_logmar);
            ind_VA = find(round(template.VA, 4) == round(va_logmar, 4));
            letters = table2array(template(ind_VA, 2:end));
            letters = strjoin(letters, '');

            for l = letters
                aberrated_opto = generate_optotype(C, l, va, Dpup, var_noise, age, phi);
                name = strcat('tempFolder/', l, '_', num2str(phi), '_', num2str(va_logmar), '.png');
                imwrite(aberrated_opto, name);

                aberrated_opto = imread(name);
                aberrated_opto = imresize(aberrated_opto, [224 224]);
                [pred, score] = classify(net, aberrated_opto);

                if pred == categorical(1) && score(2) < 0.65
                    pred = categorical(0);
                end

                if pred == categorical(0)
                    total_failures = total_failures + 1;
                    failures = failures + 1;
                end
            end
            if failures == 5
                break
            end
        end
        va_final = va_logmar + total_failures * 0.02;
        disp(['Sample ', num2str(i), ' - VA = ', num2str(va_final)])
        VA_pred(i) = va_final;

        rmdir('tempFolder', 's');
    end
end
