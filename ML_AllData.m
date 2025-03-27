
clear
analysisType = 'AllData'; % Define type for this script version
data = readtable('supplement.xlsx');
labels = data{1:522,1};
prefixes = cellfun(@(x) x(1:3), labels, 'UniformOutput', false);
uniquePrefixes = unique(prefixes);
splitData = cell(length(uniquePrefixes), 1);

for i = 1:length(uniquePrefixes)
    idx = strcmp(prefixes, uniquePrefixes{i});
    varName = sprintf('A%02d', i);
    assignin('base', varName, table2array(data(idx, 2:11)));
end

De = 3; % 1 = Depth, 2 = width, 3 = PCA
if De == 1
    txt = {'Depth'};
elseif De==2
    txt = {'Width'};
elseif De==3
    txt = {'PCA'};
end

numPermutations = 5; % Amount of permutations for the uncertainy estimation
ModelIt = 5; % Number of iterations for the machine learning models.
RI = 1; % Do random iterations rather then all 560. 0 = No, 1 = Yes
if RI == 1
    RandIterations = 10; % How many iterations?
end

% The total number of models is 560 or RandIterations*(numPermutations*2)(numPermutations*7)
% The amount of time it takes for each model increases with "ModelIt"


VarNames = {'Sediment supply','Wave energy','Tidal range',...
    'Age','Current velocity','Lithology','Sea-level'};

[~,C,S] = normalize([A01; A02; A03; A04; A05; A06; A07;...
    A08; A09; A10; A11; A12; A13; A14; A15; A16]);

for i = 1:16
    varName = sprintf('A%02d', i);
    currentArray = evalin('base', varName);
    for dol = [4 6 7]
        Dat1 = currentArray(:,dol);
        Twe1 = 5*median(Dat1)/100;
        Aa1 = Dat1-Twe1; Bb1 = Dat1+Twe1;
        Cc1 = rand(1,length(Dat1))';
        Sam1 = Aa1(1) + (Bb1(1)-Aa1(1))*Cc1;
        currentArray(:,dol) = Sam1;
    end
    assignin('base', varName, currentArray);
end

S1 = normalize(A01,"center",C,"scale",S);S2 = normalize(A02,"center",C,"scale",S);S3 = normalize(A03,"center",C,"scale",S);
S4 = normalize(A04,"center",C,"scale",S);S5 = normalize(A05,"center",C,"scale",S);S6 = normalize(A06,"center",C,"scale",S);
S7 = normalize(A07,"center",C,"scale",S);S8 = normalize(A08,"center",C,"scale",S);S9 = normalize(A09,"center",C,"scale",S);
S10 = normalize(A10,"center",C,"scale",S);S11 = normalize(A11,"center",C,"scale",S);S12 = normalize(A12,"center",C,"scale",S);
S13 = normalize(A13,"center",C,"scale",S);S14 = normalize(A14,"center",C,"scale",S);S15 = normalize(A15,"center",C,"scale",S);
S16 = normalize(A16,"center",C,"scale",S);

nTotalSamples = 16;
nTrainingSamples = 13;
sampleValues = {S1; S2; S3; S4; S5; S6; S7; S8; S9; S10; S11; S12; S13; S14; S15; S16};

combinations = nchoosek(1:nTotalSamples, nTrainingSamples);
nCombinations = size(combinations, 1);
PDS = 1; Inputs = 7;
importance = NaN(nCombinations,Inputs);
r2 = zeros(nCombinations,1);MSE = zeros(nCombinations,1);MAE = zeros(nCombinations,1);
r2T = zeros(nCombinations,1);MSET = zeros(nCombinations,1);MAET = zeros(nCombinations,1);
r2Tr = zeros(nCombinations,1);MSETr = zeros(nCombinations,1);MAETr = zeros(nCombinations,1);
Predictions = zeros(nCombinations,16);pValues2 = NaN(Inputs, nCombinations);pVal2 = NaN(Inputs, nCombinations);
pValues = NaN(Inputs, nCombinations);Tweights = NaN(Inputs, nCombinations);Aweights = NaN(Inputs, nCombinations);
MseM =  zeros(1, nCombinations);maeM =  zeros(1, nCombinations);rSquaredM = zeros(1, nCombinations);
MseM1 =  zeros(1, nCombinations);maeM1 =  zeros(1, nCombinations);rSquaredM1 = zeros(1, nCombinations);
AF1 = cell(nCombinations,1);pvalA = NaN(Inputs, nCombinations);AF2 = zeros(nCombinations,2);
HM = 0; failureCount = 0;

if RI == 1
    nCombinations = randperm(nCombinations, RandIterations);
else
    nCombinations = 1:nCombinations;
end
Number = 0;
End = length(nCombinations);

for i = nCombinations
    Done = 0;DO = 0;
    Number = Number+1;
    fprintf('Iteration: %d of %d\n', Number, End)
    while DO < 0.6 || DO > 0.95
        clear TX TrainY TestX TestY
        trainIndices = combinations(i, :);
        testIndices = setdiff(1:nTotalSamples, trainIndices);
        Y1 = []; X1 = [];
        SampleList = [];
        for j = trainIndices
            DataPo = sampleValues{j};
            TX = [];
            TX =  DataPo(:,[4 5 6 7 8 9 10]);
            TrainY = DataPo(:,De) ;
            Y1 = [Y1; TrainY];
            X1 = [X1; TX];
            SampleList = [SampleList; repmat(j, length(TrainY), 1)];
        end

        Y2 = []; X2 = [];TX = [];
        for j = testIndices
            DataPo = sampleValues{j};
            TX = [];
            TX =  DataPo(:,[4 5 6 7 8 9 10]);
            TestY = DataPo(:,De);
            Y2 = [Y2; TestY];
            X2 = [X2; TX];
            SampleList = [SampleList; repmat(j, length(TestY), 1)];
        end

        XA = [X1; X2];YA = [Y1; Y2];

        [Tot,~] = size(X2);
        [Sta,~] = size(XA);
        Test = zeros(Sta,1);
        Test(Sta-Tot+1:Sta,1) = ones(Tot,1);
        part = cvpartition("CustomPartition", Test);

        svmModel = fitrsvm(XA, YA,'OptimizeHyperparameters','all',...
            'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus',...
            'ShowPlots',false,'CVPartition',part,...
            'UseParallel',true,'MaxObjectiveEvaluations',ModelIt,'Verbose',0));


        YPredT = predict(svmModel, X2);
        DO = 1 - sum((Y2 - YPredT).^2) / sum((Y2 - mean(Y2)).^2); % Test if the models failed
        Done = Done + 1;

        if Done == 60
            fprintf('Iteration %d failed.\n', Number);
            failureCount = failureCount + 1;
            break
        end

    end

    if DO > 0.6

        % These are for the PDP plots
        [pr1(:,PDS),x1(:,PDS),~] = partialDependence(svmModel,1);
        [pr2(:,PDS),x2(:,PDS),~] = partialDependence(svmModel,2);
        [pr3(:,PDS),x3(:,PDS),~] = partialDependence(svmModel,3);
        [pr4(:,PDS),x4(:,PDS),~] = partialDependence(svmModel,4);
        [pr5(:,PDS),x5(:,PDS),~] = partialDependence(svmModel,5);
        [pr6(:,PDS),x6(:,PDS),~] = partialDependence(svmModel,6);
        [pr7(:,PDS),x7(:,PDS),~] = partialDependence(svmModel,7);

        PDS = PDS+1;

        X = [X1; X2];
        Y = [Y1; Y2];

        Y_Predict = predict(svmModel, X);r2(i,:) = 1 - sum((Y - Y_Predict).^2) / sum((Y - mean(Y)).^2);
        MSE(i,:) = mean((Y_Predict - Y).^2);MAE(i,:) = mean(abs(Y_Predict - Y));
        clear p1

        % reorder the sites and make the predictions
        Calcs = [Y_Predict-Y SampleList(:,1)]; Calcs = sortrows(Calcs, 2);
        Predictions(i,:) = splitapply(@mean, Calcs(:,1), Calcs(:,2));

        % calculate the statistics for the model.
        YPredTR = predict(svmModel, X1); % Training data
        r2Tr(i,:) = 1 - sum((Y1 - YPredTR).^2) / sum((Y1 - mean(Y1)).^2);
        MSETr(i,:) = mean((YPredTR - Y1 ).^2);MAETr(i,:) = mean(abs(YPredTR - Y1));

        YPredT = predict(svmModel, X2); % Test data
        r2T(i,:) = 1 - sum((Y2 - YPredT).^2) / sum((Y2 - mean(Y2)).^2);
        MSET(i,:) = mean((YPredT - Y2 ).^2);MAET(i,:) = mean(abs(YPredT - Y2));

        weights = svmModel.SupportVectors' * svmModel.Alpha;
        weights = weights ./ norm(weights);
        importance(i,:) = weights'; % the importance for the models

        % List the kernal fuctions
        KF = svmModel.ModelParameters.KernelFunction;
        AF1{i,1} = svmModel.ModelParameters.KernelFunction;
        AF2(i,1) = svmModel.Epsilon;
        AF2(i,2) = svmModel.BoxConstraints(1,1);

        % Perform a permutation test to obtain p-values for the importance of each variable
        numVars = size(X, 2);permutedWeights = zeros(numPermutations, numVars);
        TestWeights = NaN(numPermutations, numVars); mseJ = zeros(numPermutations, 1);
        maeJ = zeros(numPermutations, 1) ; rSquaredJ = zeros(numPermutations, 1);
        mseJ1 = zeros(numPermutations, 1); maeJ1 = zeros(numPermutations, 1) ;
        rSquaredJ1 = zeros(numPermutations, 1);AllWeights = NaN(numPermutations, numVars);
        Pdiff = zeros(numPermutations,size(X1, 2));rSquaredAL = zeros(numPermutations, 1);
        peDiff= zeros(numPermutations,size(X1, 2));

        for j = 1:numPermutations
            fprintf('Permutation: %d of %d\n', j, numPermutations)
            for kk = 1:size(XA, 2)  % Pvalue model with random inputs

                permutedXA = XA;
                % permutedXA(:, kk) = XA(randperm(length(XA)), kk);
                permutedXA(:, kk) = randi([-1, 1], length(XA), 1);

                pModel = fitrsvm(permutedXA, YA,'OptimizeHyperparameters','all',...
                    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus',...
                    'ShowPlots',false,'CVPartition',part,...
                    'UseParallel',true,'MaxObjectiveEvaluations',ModelIt,'Verbose',0));

                YPr = predict(pModel, permutedXA(1:part.TrainSize,:));
                ModQ = 1 - sum((Y1 - YPr).^2) / sum((Y1 - mean(Y1)).^2);
                Pdiff(j,kk) = r2Tr(i,:) - ModQ;

                lol = mean((YPr - Y1 ).^2);

                peDiff(j,kk) = MSE(i,:)-lol;

            end
            % Pvalue model with random outputs
            permutedY = YA(randperm(length(YA))); % Randomly permute the labels
            permutedModel = fitrsvm(XA, permutedY,'OptimizeHyperparameters','all',...
                'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus',...
                'ShowPlots',false,'CVPartition',part,...
                'UseParallel',true,'MaxObjectiveEvaluations',ModelIt,'Verbose',0));

            MyV = randperm(length(X1),round(length(X1)/2))'; % Get uncertainty
            TestModel = fitrsvm(X1(MyV,:), Y1(MyV,1)); % Half data
            AllDATA = fitrsvm(X1, Y1); % All data

            Y_Predict = predict(TestModel,X1(MyV,:));
            YPreda = predict(AllDATA,X1);
            mse = mean((Y_Predict - Y(MyV,1)).^2);
            mae = mean(abs(Y_Predict - Y(MyV,1)));
            rSquaredJ(j,1) = 1 - sum((Y_Predict - mean(Y(MyV,1))).^2)   / sum((Y(MyV,1) - mean(Y(MyV,1))).^2);
            if rSquaredJ(j,1) > 0.3
                TestWeights(j,:) = TestModel.SupportVectors' * TestModel.Alpha ./ norm(TestModel.SupportVectors' * TestModel.Alpha);
            end
            rSquaredAL(j,1) = 1 - sum((YPreda - mean(Y1)).^2)   / sum((Y1 - mean(Y1)).^2);
            if rSquaredAL(j,1) > 0.3
                AllWeights(j,:) = AllDATA.SupportVectors' * AllDATA.Alpha ./ norm(AllDATA.SupportVectors' * AllDATA.Alpha);
            end
            permutedWeights(j,:) = permutedModel.SupportVectors' * permutedModel.Alpha ./ norm(permutedModel.SupportVectors' * permutedModel.Alpha); % Get the weights for the permuted model
            mseJ(j,1) = mse;
            maeJ(j,1) = mae;
        end

        pvalA(:,i) = mean(Pdiff);
        pValues2(:,i) = mean(Pdiff < 0, 1);
        pVal2(:,i) =  mean(peDiff > 0, 1);
        pValues(:,i) = arrayfun(@(k) sum(abs(permutedWeights(:,k)) >= abs(weights(k))) / numPermutations, 1:numVars); % Compute the p-values
        Tweights(:,i) = mean(TestWeights,'omitnan');
        Aweights(:,i) = mean(AllWeights,'omitnan');
    else
    end

end

% Calculate the means.
MeanImportance = mean(importance,'omitnan')';[~, idx] = sort(abs(MeanImportance), 'ascend');
[~, IDS] = sort(abs(MeanImportance), 'descend');
sortedImp = MeanImportance(idx,:);
M_MSEt = mean(nonzeros(MSET));M_MAEt = mean(nonzeros(MAET));M_RSt = mean(nonzeros(r2T));
M_MSE = mean(nonzeros(MSE));M_MAE = mean(nonzeros(MAE));M_RS = mean(nonzeros(r2));
M_MSEtr = mean(nonzeros(MSETr));M_MAEtr = mean(nonzeros(MAETr));M_RStr = mean(nonzeros(r2Tr));
TImportance = mean(Tweights, 2,'omitnan');TStdImportance = std(Tweights,0,2, 'omitnan');
AImportance = mean(Aweights, 2,'omitnan');AStdImportance = std(Aweights,0,2, 'omitnan');
meanPValues = mean(pValues, 2,'omitnan');ErrorI = std([TImportance AImportance MeanImportance],0,2);
meanPValues2 = mean(pValues2, 2,'omitnan'); meanPVal2 = mean(pvalA, 2,'omitnan'); mePVal2 = mean(pVal2, 2,'omitnan');

% Create a single figure window
figure('Name', 'Feature Importance and P-values', 'NumberTitle', 'off');

%  Subplot 1: Half data importance 
subplot(2, 3, 1);
[~, idxT] = sort(abs(TImportance), 'ascend'); % Sort by absolute importance
sortedImpT = TImportance(idxT,:);
barh(sortedImpT); % Horizontal bar plot
hold on;
yticks(1:length(idxT));
yticklabels(VarNames(idxT));
xlabel('Coefficient Estimate');
title('Importance (Half Training Data)');
axis padded;
hold off;

%  Subplot 2: All training data importance 
subplot(2, 3, 2);
[~, idxA] = sort(abs(AImportance), 'ascend'); % Sort by absolute importance
sortedImpA = AImportance(idxA,:);
barh(sortedImpA); % Horizontal bar plot
hold on;
yticks(1:length(idxA));
yticklabels(VarNames(idxA));
xlabel('Coefficient Estimate');
title('Importance (All Training Data)');
axis padded;
hold off;

%  Subplot 3: Average P-value 
subplot(2, 3, 3);
TotalPVal = (meanPValues + meanPValues2) / 2;
[~, idxP_avg] = sort(TotalPVal, 'ascend');
sortedTotP = TotalPVal(idxP_avg,:);
b_avg = bar(sortedTotP); 
xticks(1:length(idxP_avg));
xticklabels(VarNames(idxP_avg));
ylabel('Average P-value');
title('Average P-value (PST + PIT)');
axis padded;
ylim([0 max(sortedTotP)*1.15 + eps]); 

for k = 1:length(sortedTotP)
    text(k, sortedTotP(k), num2str(sortedTotP(k), '%.2f'), ...
         'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'bottom');
end

%  Subplot 4: PST P-value 
subplot(2, 3, 4);
[~, idxP_pst] = sort(meanPValues, 'ascend');
sortedPValues = meanPValues(idxP_pst,:);
b_pst = bar(sortedPValues); 
xticks(1:length(idxP_pst));
xticklabels(VarNames(idxP_pst));
ylabel('P-value');
title('PST P-value');
axis padded;
ylim([0 max(sortedPValues)*1.15 + eps]); 
% Add text labels individually using a loop
for k = 1:length(sortedPValues)
    text(k, sortedPValues(k), num2str(sortedPValues(k), '%.2f'), ...
         'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'bottom');
end

%  Subplot 5: PIT P-value 
subplot(2, 3, 5);
[~, idxP_pit] = sort(meanPValues2, 'ascend'); % Sort by actual value
sortedPValues2 = meanPValues2(idxP_pit,:);
b_pit = bar(sortedPValues2); % Vertical bar plot
xticks(1:length(idxP_pit));
xticklabels(VarNames(idxP_pit));
ylabel('P-value');
title('PIT P-value');
axis padded;
ylim([0 max(sortedPValues2)*1.15 + eps]); % Adjust ylim slightly more for text
% Add text labels individually using a loop
for k = 1:length(sortedPValues2)
    text(k, sortedPValues2(k), num2str(sortedPValues2(k), '%.2f'), ...
         'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'bottom');
end

sgtitle(['Importance and Significance for ' txt{1} ' Prediction']);

S_ER = ErrorI(idx,:); 
figure; 
barh(sortedImp);
hold on
errorbar( sortedImp,1:numel(sortedImp), ...
    S_ER,'horizontal', 'LineStyle', 'none', 'Color', 'k');
hold off 
yticks(1:numel(sortedImp)); 
yticklabels(VarNames(idx)); 
title(['Primary Feature Importance: ' txt{1}])
xlabel('Coefficient Estimate');
axis padded;

opt1 = VarNames(IDS)';
Opt = [MeanImportance(IDS) ErrorI(IDS) meanPValues(IDS) meanPValues2(IDS) meanPVal2(IDS) mePVal2(IDS)];
fprintf('Total failures: %d out of %d iterations.\n', failureCount, End);
disp(['Mean Squared Error (MSE): ' num2str(mean([M_MSE M_MSEtr M_MSEt]))]);
disp(['Mean Absolute Error (MAE): ' num2str(mean([M_MAE M_MAEtr M_MAEt]))]);
disp(['Mean R-squared (Coefficient of Determination): ' num2str(mean([M_RS M_RStr M_RSt]))]);

Op2 = [M_MSE M_MAE M_RS M_MSEtr M_MAEtr M_RStr M_MSEt M_MAEt M_RSt]';

pol=[1 .6 .1; 1 .6 .1;   1 .6 .1;  1 .6 .1;  1 .6 .1;...
    1 .6 .1; 1 .6 .1;   1 .6 .1;  1 .6 .1;  1 .6 .1;...
    1 .6 .1; 1 .6 .1;   1 .6 .1;  1 .6 .1;  1 .6 .1;...
    1 .6 .1; 1 .6 .1;   1 .6 .1;  1 .6 .1;  1 .6 .1];

Predictions(Predictions == 0) = NaN;


marginNames = {
    'Bangladesh', 'Africa (Kenya)', 'Australia (N)', 'Australia (S)', ...
    'East India', 'France', 'Africa (Guinea)', 'N. America (East)', ...
    'N. America (GoM)', 'S. America (Amazon)', 'S. America (Uruguay)', 'South China', ...
    'Africa (Congo)', 'Africa (Nigeria)', 'Africa (Ghana)', 'West India'};

figure('Name', 'Prediction Errors per Margin', 'NumberTitle', 'off');

try
    h = daboxplot(Predictions, 'mean', 0, 'color', pol, 'outliers', 0);
catch ME % Fallback if daboxplot fails or is not found
    warning('daboxplot failed or not found. Using standard boxplot.');
    boxplot(Predictions); % Use standard boxplot as a fallback
end

hold on; 

title(['Predicted vs Actual Outcomes: ' txt{1}]);
ylabel('Prediction Error (Predicted - Actual)'); % Add Y-axis label
axis padded;
yline(0, 'k--');
ylim([-1.5 1.5]);

% Set X-axis ticks and labels
xticks(1:length(marginNames));
xticklabels(marginNames);
xtickangle(45);
grid on;

hold off;
%  End Prediction Error Plot 

figure('Name', 'Partial Dependence Plots', 'NumberTitle', 'off');
pdpDataY = {pr1, pr2, pr3, pr4, pr5, pr6, pr7}; % Cell array of Y data (pr1 to pr7)
pdpDataX = {x1, x2, x3, x4, x5, x6, x7}; 
numPlots = length(pdpDataY); 

% Loop through each variable to create a subplot
for k = 1:numPlots
    subplot(3, 3, k); % Create a 3x3 grid and select the k-th subplot
    yr = pdpDataY{k}; % Get the Y data for the current variable
  
    % Calculate mean and standard deviation
    mean_y = mean(yr, 2, 'omitnan'); % Calculate mean 
    std_y = std(yr, 0, 2, 'omitnan');  % Calculate std dev

    plot(yr, '-', 'color', [1 .6 .1], 'LineWidth', .2);
    hold on;
    plot(mean_y, 'k', 'LineWidth', 2);
    plot(mean_y + std_y, 'k--', 'LineWidth', 1);
    plot(mean_y - std_y, 'k--', 'LineWidth', 1);

    hold off;
    axis padded;
    ylim([-3 3]); 
    title(VarNames{k}); 

    ylabel('Partial Dependence');

end

sgtitle(['Partial Dependence for ' txt{1} ' Prediction']);

% SAVE RESULT

% Create a structure to hold results with descriptive feild names
resultsData = struct();

%  Execution Parameters 
resultsData.executionTimestamp = datetime('now'); % Record when the analysis was run
resultsData.targetVariableIndex = De; % 1=Depth, 2=Width, 3=PCA
resultsData.targetVariableName = txt{1};
resultsData.featureNames = VarNames; % Original order
resultsData.hyperparameterOptimizationIterations = ModelIt; % MaxObjectiveEvaluations per fold
resultsData.pValuePermutations = numPermutations; % Permutations run per fold for p-values

if RI == 1
    resultsData.ranRandomSubsetInsteadOfAllFolds = 'Yes';
else
    resultsData.ranRandomSubsetInsteadOfAllFolds = 'No'; 
end
    
resultsData.TotalModels = End;

resultsData.FailedModels = failureCount;

resultsData.totalMargins = nTotalSamples;
resultsData.trainingMarginsPerFold = nTrainingSamples;


% Create a detailed table for impotrance metrics, sorted by MeanImportance (descending)
[~, IDS] = sort(abs(MeanImportance), 'descend');
sortedVarNames = VarNames(IDS)';

if ~exist('meanPVal2','var') || numel(meanPVal2) ~= numel(IDS), meanPVal2 = NaN(numel(IDS), 1); end
if ~exist('mePVal2','var') || numel(mePVal2) ~= numel(IDS), mePVal2 = NaN(numel(IDS), 1); end

try
    detailedImportanceTable = table(...
        sortedVarNames, MeanImportance(IDS), ErrorI(IDS), meanPValues(IDS), ...
        meanPValues2(IDS), ...
        'VariableNames', {'FeatureName', 'MeanImportance', 'StdDevImportance', 'PST_PValue', 'PIT_PValue'});
    resultsData.detailedImportanceMetricsTable = detailedImportanceTable;
catch ME
    warning('Could not create detailed importance table. Saving raw Opt matrix instead.');
    resultsData.detailedImportanceMetrics_Raw = Opt; % Fallback
end

%  Performance Metrics 
if length(Op2) >= 9
    resultsData.meanMSE_AllData = Op2(1); resultsData.meanMAE_AllData = Op2(2); resultsData.meanR2_AllData = Op2(3);
    resultsData.meanMSE_TrainData = Op2(4); resultsData.meanMAE_TrainData = Op2(5); resultsData.meanR2_TrainData = Op2(6);
    resultsData.meanMSE_TestData = Op2(7); resultsData.meanMAE_TestData = Op2(8); resultsData.meanR2_TestData = Op2(9);
else
    resultsData.performanceMetrics_RawVector = Op2;
    warning('Op2 variable did not contain the expected 9 performance metrics.');
end

% Prediction Errors per Margin
resultsData.predictionErrorsPerMargin = Predictions;

% Partial Dependence Plot Data
pdpData = struct();
pdpData.sedimentSupply_Y = pr1; pdpData.sedimentSupply_X = x1;
pdpData.waveEnergy_Y = pr2;     pdpData.waveEnergy_X = x2;
pdpData.tidalRange_Y = pr3;     pdpData.tidalRange_X = x3;
pdpData.age_Y = pr4;            pdpData.age_X = x4;
pdpData.currentVelocity_Y = pr5; pdpData.currentVelocity_X = x5;
pdpData.lithology_Y = pr6;      pdpData.lithology_X = x6;
pdpData.seaLevel_Y = pr7;       pdpData.seaLevel_X = x7;
resultsData.partialDependenceData = pdpData;

%  Model Parameters from Folds 
resultsData.kernelFunctionsUsed = AF1;
resultsData.epsilonAndBoxConstraints = AF2;

resultsFilename = sprintf('%s_Results_%s.mat', analysisType, resultsData.targetVariableName);
try
    save(resultsFilename, 'resultsData'); % Save the single structure variable
    fprintf('Results successfully saved to %s\n', resultsFilename);
catch ME
    fprintf('ERROR saving results to %s\n', resultsFilename);
    fprintf('Error Message: %s\n', ME.message);
end


