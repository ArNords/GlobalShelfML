
clear

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

De = 1; % 1 = Depth, 2 = width, 3 = PCA
if De == 1
    txt = {'Depth'};
elseif De==2
    txt = {'Width'};
elseif De==3
    txt = {'PCA'};
end

numPermutations = 200; % Amount of permutations for the uncertainy estimation
ModelIt = 100; % Number of iterations for the machine learning models.
RI = 0; % Do random iterations rather then all 560. 0 = No, 1 = Yes
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
KF1 = cell(nCombinations,1);pvalA = NaN(Inputs, nCombinations);
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
            DatPre = sampleValues{j};
            R_Num = randperm([length(DatPre)], 13);
            DataPo = DatPre(R_Num,:);
            TX = [];
            TX =  DataPo(:,[4 5 6 7 8 9 10]);       
            TrainY = DataPo(:,De) ;
            Y1 = [Y1; TrainY];
            X1 = [X1; TX];
            SampleList = [SampleList; repmat(j, length(TrainY), 1)];
        end

        Y2 = []; X2 = [];TX = [];
        for j = testIndices
            DatPre = sampleValues{j};
            R_Num = randperm([length(DatPre)], 13);
            DataPo = DatPre(R_Num,:);
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
        KF1{i} = svmModel.ModelParameters.KernelFunction;

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

% outputs and figures
[~, idxT] = sort(abs(TImportance), 'ascend');
sortedImpT = TImportance(idxT,:);
figure;
barh(sortedImpT);
hold on;
yticks(1:length(idxT));
yticklabels(VarNames(idxT));
xlabel('Coefficient Estimate');
title('Half data importance');

[~, idxA] = sort(abs(AImportance), 'ascend');
sortedImpA = AImportance(idxA,:);
figure;
barh(sortedImpA);
hold on;
yticks(1:length(idxA));
yticklabels(VarNames(idxA));
xlabel('Coefficient Estimate');
title('All data importance');

[~, idxP] = sort(abs(meanPValues), 'ascend');
sortedPValues = meanPValues(idxP,:);
figure
bar(sortedPValues);
xticks(1:length(idxP));
xticklabels(VarNames(idxP));
ylabel('P-value');
sortedPValues = sortedPValues';
text(1:length(sortedPValues),sortedPValues,num2str(sortedPValues'),'vert','bottom','horiz','center');
title('PST P-value');
axis padded

[~, idxP] = sort(abs(meanPValues2), 'ascend');
sortedPValues2 = meanPValues2(idxP,:);
figure
bar(sortedPValues2);
xticks(1:length(idxP));
xticklabels(VarNames(idxP));
ylabel('P-value');
sortedPValues2 = sortedPValues2';
text(1:length(sortedPValues2),sortedPValues2,num2str(sortedPValues2'),'vert','bottom','horiz','center');
title('PIT P-value');
axis padded

TotalPVal = (meanPValues+meanPValues2)/2;
[~, idxP] = sort(abs(TotalPVal), 'ascend');
sortedTotP = TotalPVal(idxP,:);
figure
bar(sortedTotP);
xticks(1:length(idxP));
xticklabels(VarNames(idxP));
ylabel('P-value');
sortedTotP = sortedTotP';
text(1:length(sortedTotP),sortedTotP,num2str(sortedTotP'),'vert','bottom','horiz','center');
title('Average P-value');
axis padded

S_ER = ErrorI(idx,:);
figure;
barh(sortedImp);
hold on
errorbar( sortedImp,1:numel(sortedImp), ...
    S_ER,'horizontal', 'LineStyle', 'none', 'Color', 'k');
hold on
yticklabels(VarNames(idx));
title(['Primary output' txt])
axis padded

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

figure
h = daboxplot(Predictions,'mean',0,'color',pol,'outliers',0);
title(['Predicted vs actual outcomes' txt])
axis padded
yline(0)
ylim([-1.5 1.5])

yr = pr1;
mean_y = mean(yr');std_y = std(yr');
figure;
plot(yr,'-','color', [1 .6 .1],'LineWidth',.2);hold on
plot(mean_y,'k','LineWidth',2)
plot(mean_y+std_y,'k--','LineWidth',1)
plot(mean_y-std_y,'k--','LineWidth',1)
axis padded
ylim([-3 3]);title([VarNames(1)])

yr = pr2;
mean_y = mean(yr');std_y = std(yr');
figure;
plot(yr,'-','color', [1 .6 .1],'LineWidth',.2);hold on
plot(mean_y,'k','LineWidth',2)
plot(mean_y+std_y,'k--','LineWidth',1)
plot(mean_y-std_y,'k--','LineWidth',1)
axis padded
ylim([-3 3]);title([VarNames(2)])

yr = pr3;
mean_y = mean(yr');std_y = std(yr');
figure;
plot(yr,'-','color', [1 .6 .1],'LineWidth',.2);hold on
plot(mean_y,'k','LineWidth',2)
plot(mean_y+std_y,'k--','LineWidth',1)
plot(mean_y-std_y,'k--','LineWidth',1)
axis padded
ylim([-3 3]);title([VarNames(3)])

yr = pr4;
mean_y = mean(yr');std_y = std(yr');
figure;
plot(yr,'-','color', [1 .6 .1],'LineWidth',.2);hold on
plot(mean_y,'k','LineWidth',2)
plot(mean_y+std_y,'k--','LineWidth',1)
plot(mean_y-std_y,'k--','LineWidth',1)
axis padded
ylim([-3 3]);title([VarNames(4)])

yr = pr5;
mean_y = mean(yr');std_y = std(yr');
figure;
plot(yr,'-','color', [1 .6 .1],'LineWidth',.2);hold on
plot(mean_y,'k','LineWidth',2)
plot(mean_y+std_y,'k--','LineWidth',1)
plot(mean_y-std_y,'k--','LineWidth',1)
axis padded
ylim([-3 3]);title([VarNames(5)])

yr = pr6;
mean_y = mean(yr');std_y = std(yr');
figure;
plot(yr,'-','color', [1 .6 .1],'LineWidth',.2);hold on
plot(mean_y,'k','LineWidth',2)
plot(mean_y+std_y,'k--','LineWidth',1)
plot(mean_y-std_y,'k--','LineWidth',1)
axis padded
ylim([-3 3]);title([VarNames(6)])

yr = pr7;
mean_y = mean(yr');std_y = std(yr');
figure;
plot(yr,'-','color', [1 .6 .1],'LineWidth',.2);hold on
plot(mean_y,'k','LineWidth',2)
plot(mean_y+std_y,'k--','LineWidth',1)
plot(mean_y-std_y,'k--','LineWidth',1)
axis padded
ylim([-3 3]);title([VarNames(7)])




