clear

data = readtable("Nordsvan_etal.xlsx");

 De = 1; % 1 = depth, 2 = width, 3 = PCA

 if De == 1
     txt = {'Depth'};
 elseif De==2
     txt = {'Width'};
 elseif De==3
     txt = {'PCA'};
 end

VarNames = {'Sediment supply','Wave energy','Tidal range',...
    'Age','Current velocity','Lithology','Sea-level'};

xArray = mat2cell(table2array(data(:, 1:10)), 11 * ones(16, 1), 10); % Split into 16 cell arrays

NumPerm = 200; % how many models
PDS = 1; Inputs = 7;
importance = zeros(NumPerm,Inputs);

r2 = zeros(NumPerm,1);MSE = zeros(NumPerm,1);MAE = zeros(NumPerm,1);
r2T = zeros(NumPerm,1);MSET = zeros(NumPerm,1);MAET = zeros(NumPerm,1);
r2Tr = zeros(NumPerm,1);MSETr = zeros(NumPerm,1);MAETr = zeros(NumPerm,1);
Predictions = zeros(NumPerm,16);
pValues = zeros(Inputs, NumPerm);Tweights = zeros(Inputs, NumPerm);Aweights = zeros(Inputs, NumPerm);
MseM =  zeros(1, NumPerm);maeM =  zeros(1, NumPerm);rSquaredM = zeros(1, NumPerm);
MseM1 =  zeros(1, NumPerm);maeM1 =  zeros(1, NumPerm);rSquaredM1 = zeros(1, NumPerm);
KF1 = cell(NumPerm,1);

for i = 1:NumPerm

    k = 1; Num = 11; g = Num;
    List = (1:16)';
    RandomList = List(randperm(length(List))); % Prepares the data, splits into random order on each loop.
    for j = RandomList'

        x = xArray{j};

        X = x(:,[5 6 8 9]);
        X(:,5) =  x(:,10);
        Mdl(k:g,[2 3 5 6 7]) = bootstrp(Num,@median,X);
        YX(k:g,1) = bootstrp(Num,@median,x(:,De));
        for dol = [4 7]
            X = x(:,dol);
            Twe = 5*median(X)/100;
            Aa = X-Twe; Bb = X+Twe;
            Cc = rand(1,Num)';
            Sam = Aa(1) + (Bb(1)-Aa(1))*Cc;

            if dol == 4
                Mdl(k:g,1) =  x(:,dol); %Sam;

            elseif dol == 7
                Mdl(k:g,4) = x(:,dol); % Sam;
            end
        end

        List(k:g,:) = j;

        k = k+Num;
        g = g+Num;

    end

    Y = normalize(YX);
    X = normalize(Mdl);

    [Sta,~] = size(X);
    Test = zeros(Sta,1);
    Num2 = Num*3;
    Test(Sta-Num2+1:Sta,1) = ones(Num2,1);
    part = cvpartition("CustomPartition", Test);
    X1 = X(part.training, :);
    Y1 = Y(part.training, :);
    X2 = X(part.test, :);
    Y2 = Y(part.test, :);
    
    % Optermize the primary model
    svmModel = fitrsvm(X, Y,'OptimizeHyperparameters','all',...
        'HyperparameterOptimizationOptions',struct('ShowPlots',false,'CVPartition',part...
        ,'UseParallel',true,'MaxObjectiveEvaluations',300,'Verbose',0));

    YPredT = predict(svmModel, X2);
    DO = 1 - sum((Y2 - YPredT).^2) / sum((Y2 - mean(Y2)).^2); % Test if the models failed

    if DO > 0.1

        % These are for the PDP plots
        [pr1(:,PDS),x1(:,PDS),~] = partialDependence(svmModel,1); 
        [pr2(:,PDS),x2(:,PDS),~] = partialDependence(svmModel,2);
        [pr3(:,PDS),x3(:,PDS),~] = partialDependence(svmModel,3);
        [pr4(:,PDS),x4(:,PDS),~] = partialDependence(svmModel,4);
        [pr5(:,PDS),x5(:,PDS),~] = partialDependence(svmModel,5);
        [pr6(:,PDS),x6(:,PDS),~] = partialDependence(svmModel,6);
        [pr7(:,PDS),x7(:,PDS),~] = partialDependence(svmModel,7);

        PDS = PDS+1;

        YPred = predict(svmModel, X);r2(i,:) = 1 - sum((Y - YPred).^2) / sum((Y - mean(Y)).^2);
        MSE(i,:) = mean((YPred - Y ).^2);MAE(i,:) = mean(abs(YPred - Y));
        clear p1
        p1 = zeros(16,1);

            % reorder the sites and make the predictions
        Calcs = [YPred-Y List]; Calcs = sortrows(Calcs, 2); 
        d1=1; f1=Num;
        for r = 1:16 
            p1(r,:) = mean(Calcs(d1:f1,1));
            d1 = d1+Num; f1 = f1+Num;
        end
        Predictions(i,:) = p1';

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
        numPermutations = 1000; % number of permutations for the uncertainty and the p-value
        numVars = size(X, 2);permutedWeights = zeros(numPermutations, numVars);
        TestWeights = zeros(numPermutations, numVars); mseJ = zeros(numPermutations, 1);
        maeJ = zeros(numPermutations, 1) ; rSquaredJ = zeros(numPermutations, 1);
        mseJ1 = zeros(numPermutations, 1); maeJ1 = zeros(numPermutations, 1) ;
        rSquaredJ1 = zeros(numPermutations, 1);AllWeights = zeros(numPermutations, numVars);

        for j = 1:numPermutations
            permutedY = Y1(randperm(length(Y1))); % Randomly permute the labels
            permutedModel = fitrsvm(X1, permutedY, 'KernelFunction', KF); % Pvalue model with random outputs
            MyV = randperm(length(X1),round(length(X1)/2))';
            TestModel = fitrsvm(X1(MyV,:), Y1(MyV,1), 'KernelFunction',KF); % Half data
            AllDATA = fitrsvm(X1, Y1, 'KernelFunction',KF); % All data
            YPred = predict(svmModel,X(MyV,:));
            YPreda = predict(svmModel,X);
            mse = mean((YPred - Y(MyV,1)).^2);
            mae = mean(abs(YPred - Y(MyV,1)));
            rSquaredJ(j,1) = 1 - sum((YPred - mean(Y(MyV,1))).^2)   / sum((Y(MyV,1) - mean(Y(MyV,1))).^2);
            TestWeights(j,:) = TestModel.SupportVectors' * TestModel.Alpha ./ norm(TestModel.SupportVectors' * TestModel.Alpha);
            AllWeights(j,:) = AllDATA.SupportVectors' * AllDATA.Alpha ./ norm(AllDATA.SupportVectors' * AllDATA.Alpha);
            permutedWeights(j,:) = permutedModel.SupportVectors' * permutedModel.Alpha ./ norm(permutedModel.SupportVectors' * permutedModel.Alpha); % Get the weights for the permuted model
            mseJ(j,1) = mse;
            maeJ(j,1) = mae;
        end
        pValues(:,i) = arrayfun(@(k) sum(abs(permutedWeights(:,k)) >= abs(weights(k))) / numPermutations, 1:numVars); % Compute the p-values
        Tweights(:,i) = mean(TestWeights);
        Aweights(:,i) = mean(AllWeights);
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
TImportance = mean(Tweights, 2);TStdImportance = std(Tweights,0,2);
AImportance = mean(Aweights,2);AStdImportance = std(Aweights,0,2);
meanPValues = mean(pValues, 2);ErrorI = std([TImportance AImportance MeanImportance],0,2);

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
xlabel('Variable Index');
ylabel('P-value');
sortedPValues = sortedPValues';
text(1:length(sortedPValues),sortedPValues,num2str(sortedPValues'),'vert','bottom','horiz','center');
title('SVMG-P-values');
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
Opt = [MeanImportance(IDS) ErrorI(IDS) meanPValues(IDS)];

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

xr = x1;yr = pr1;
mean_y = mean(yr');med_y = mean(yr');
m_y  = smooth(med_y);std_y = std(yr');
mean_x = mean(xr');std_x = std(xr');figure;
plot(xr, yr,'-','color', [1 .6 .1],'LineWidth',.2);hold on
plot(mean_x,mean_y,'k','LineWidth',2)
plot(mean_x+std_x,mean_y+std_y,'k--','LineWidth',1)
plot(mean_x-std_x,mean_y-std_y,'k--','LineWidth',1)
axis padded
ylim([-3 3]);title([VarNames(1)])

xr = x2;yr = pr2;
mean_y = mean(yr');med_y = mean(yr');
m_y  = smooth(med_y);std_y = std(yr');
mean_x = mean(xr');std_x = std(xr');figure;
plot(xr, yr,'-','color', [1 .6 .1],'LineWidth',.2);hold on
plot(mean_x,mean_y,'k','LineWidth',2)
plot(mean_x+std_x,mean_y+std_y,'k--','LineWidth',1)
plot(mean_x-std_x,mean_y-std_y,'k--','LineWidth',1)
axis padded
ylim([-3 3]);title([VarNames(2)])

xr = x3;yr = pr3;
mean_y = mean(yr');med_y = mean(yr');
m_y  = smooth(med_y);std_y = std(yr');
mean_x = mean(xr');std_x = std(xr');figure;
plot(xr, yr,'-','color', [1 .6 .1],'LineWidth',.2);hold on
plot(mean_x,mean_y,'k','LineWidth',2)
plot(mean_x+std_x,mean_y+std_y,'k--','LineWidth',1)
plot(mean_x-std_x,mean_y-std_y,'k--','LineWidth',1)
axis padded
ylim([-3 3]);title([VarNames(3)])

xr = x4;yr = pr4;
mean_y = mean(yr');med_y = mean(yr');
m_y  = smooth(med_y);std_y = std(yr');
mean_x = mean(xr');std_x = std(xr');figure;
plot(xr, yr,'-','color', [1 .6 .1],'LineWidth',.2);hold on
plot(mean_x,mean_y,'k','LineWidth',2)
plot(mean_x+std_x,mean_y+std_y,'k--','LineWidth',1)
plot(mean_x-std_x,mean_y-std_y,'k--','LineWidth',1)
axis padded
ylim([-3 3]);title([VarNames(4)])

xr = x5;yr = pr5;
mean_y = mean(yr');med_y = mean(yr');
m_y  = smooth(med_y);std_y = std(yr');
mean_x = mean(xr');std_x = std(xr');figure;
plot(xr, yr,'-','color', [1 .6 .1],'LineWidth',.2);hold on
plot(mean_x,mean_y,'k','LineWidth',2)
plot(mean_x+std_x,mean_y+std_y,'k--','LineWidth',1)
plot(mean_x-std_x,mean_y-std_y,'k--','LineWidth',1)
axis padded
ylim([-3 3]);title([VarNames(5)])

xr = x6;yr = pr6;
mean_y = mean(yr');med_y = mean(yr');
m_y  = smooth(med_y);std_y = std(yr');
mean_x = mean(xr');std_x = std(xr');figure;
plot(xr, yr,'-','color', [1 .6 .1],'LineWidth',.2);hold on
plot(mean_x,mean_y,'k','LineWidth',2)
plot(mean_x+std_x,mean_y+std_y,'k--','LineWidth',1)
plot(mean_x-std_x,mean_y-std_y,'k--','LineWidth',1)
axis padded
ylim([-3 3]);title([VarNames(6)])

xr = x7;yr = pr7;
mean_y = mean(yr');med_y = mean(yr');
m_y  = smooth(med_y);std_y = std(yr');
mean_x = mean(xr');std_x = std(xr');figure;
plot(xr, yr,'-','color', [1 .6 .1],'LineWidth',.2);hold on
plot(mean_x,mean_y,'k','LineWidth',2)
plot(mean_x+std_x,mean_y+std_y,'k--','LineWidth',1)
plot(mean_x-std_x,mean_y-std_y,'k--','LineWidth',1)
axis padded
ylim([-3 3]);title([VarNames(7)])

