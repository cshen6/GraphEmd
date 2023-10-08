optional=1;

dataURL = "http://quantum-machine.org/data/qm7.mat";
outputFolder = fullfile(tempdir,"qm7Data");
dataFile = fullfile(outputFolder,"qm7.mat");

if ~exist(dataFile,"file")
    mkdir(outputFolder);
    disp("Downloading QM7 data...");
    websave(dataFile, dataURL);
    disp("Done.")
end

data = load(dataFile)
coulombData = double(permute(data.X, [2 3 1]));
atomData = sort(data.Z,2,'descend');
adjacencyData = coulomb2Adjacency(coulombData,atomData);

if optional==1
    figure
    tiledlayout("flow")

    for i = 1:9
        % Extract unpadded adjacency matrix.
        atomicNumbers = nonzeros(atomData(i,:));
        numNodes = numel(atomicNumbers);
        A = adjacencyData(1:numNodes,1:numNodes,i);

        % Convert adjacency matrix to graph.
        G = graph(A);

        % Convert atomic numbers to symbols.
        symbols = atomicSymbol(atomicNumbers);

        % Plot graph.
        nexttile
        plot(G,NodeLabel=symbols,Layout="force")
        title("Molecule " + i)
    end

    figure
    histogram(categorical(atomicSymbol(atomData)))
    xlabel("Node Label")
    ylabel("Frequency")
    title("Label Counts")
end

numObservations = size(adjacencyData,3);
[idxTrain,idxValidation,idxTest] = trainingPartitions(numObservations,[0.8 0.1 0.1]);

adjacencyDataTrain = adjacencyData(:,:,idxTrain);
adjacencyDataValidation = adjacencyData(:,:,idxValidation);
adjacencyDataTest = adjacencyData(:,:,idxTest);

coulombDataTrain = coulombData(:,:,idxTrain);
coulombDataValidation = coulombData(:,:,idxValidation);
coulombDataTest = coulombData(:,:,idxTest);

atomDataTrain = atomData(idxTrain,:);
atomDataValidation = atomData(idxValidation,:);
atomDataTest = atomData(idxTest,:);

[ATrain,XTrain,labelsTrain] = GCNpreprocessData(adjacencyDataTrain,coulombDataTrain,atomDataTrain);
[AValidation,XValidation,labelsValidation] = GCNpreprocessData(adjacencyDataValidation,coulombDataValidation,atomDataValidation);


muX = mean(XTrain);
sigsqX = var(XTrain,1);

XTrain = (XTrain - muX)./sqrt(sigsqX);
XValidation = (XValidation - muX)./sqrt(sigsqX);
parameters = struct;

numHiddenFeatureMaps = 32;
numInputFeatures = size(XTrain,2);

sz = [numInputFeatures numHiddenFeatureMaps];
numOut = numHiddenFeatureMaps;
numIn = numInputFeatures;
parameters.mult1.Weights = initializeGlorot(sz,numOut,numIn,"double");

sz = [numHiddenFeatureMaps numHiddenFeatureMaps];
numOut = numHiddenFeatureMaps;
numIn = numHiddenFeatureMaps;
parameters.mult2.Weights = initializeGlorot(sz,numOut,numIn,"double");

classes = categories(labelsTrain);
numClasses = numel(classes);

sz = [numHiddenFeatureMaps numClasses];
numOut = numClasses;
numIn = numHiddenFeatureMaps;
parameters.mult3.Weights = initializeGlorot(sz,numOut,numIn,"double");

numEpochs = 1500;
learnRate = 0.01;
validationFrequency = 300;
figure
C = colororder;
lineLossTrain = animatedline(Color=C(2,:));
lineLossValidation = animatedline( ...
    LineStyle="--", ...
    Marker="o", ...
    MarkerFaceColor="black");
ylim([0 inf])
xlabel("Epoch")
ylabel("Loss")
grid on

trailingAvg = [];
trailingAvgSq = [];
XTrain = dlarray(XTrain);
XValidation = dlarray(XValidation);
if canUseGPU
    XTrain = gpuArray(XTrain);
end

TTrain = onehotencode(labelsTrain,2,ClassNames=classes);
TValidation = onehotencode(labelsValidation,2,ClassNames=classes);

%monitor = trainingProgressMonitor( ...
 %   Metrics=["TrainingLoss","ValidationLoss"], ...
  %  Info="Epoch", ...
  %  XLabel="Epoch");

%groupSubPlot(monitor,"Loss",["TrainingLoss","ValidationLoss"])
start = tic;

for epoch = 1:numEpochs
    % Evaluate the model loss and gradients.
    [loss,gradients] = dlfeval(@GCNmodelLoss,parameters,XTrain,ATrain,TTrain);

    % Update the network parameters using the Adam optimizer.
    [parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients, ...
        trailingAvg,trailingAvgSq,epoch,learnRate);

    % Update the training progress plot.
    D = duration(0,0,toc(start),Format="hh:mm:ss");
    title("Epoch: " + epoch + ", Elapsed: " + string(D))
    loss = double(loss);
    addpoints(lineLossTrain,epoch,loss)
    drawnow

    % Display the validation metrics.
    if epoch == 1 || mod(epoch,validationFrequency) == 0
        YValidation = GCNmodel(parameters,XValidation,AValidation);
        lossValidation = crossentropy(YValidation,TValidation,DataFormat="BC");

        lossValidation = double(lossValidation);
        addpoints(lineLossValidation,epoch,lossValidation)
        drawnow
    end
end

[ATest,XTest,labelsTest] = GCNpreprocessData(adjacencyDataTest,coulombDataTest,atomDataTest);
XTest = (XTest - muX)./sqrt(sigsqX);
XTest = dlarray(XTest);
YTest = GCNmodel(parameters,XTest,ATest);
YTest = onehotdecode(YTest,classes,2);
accuracy = mean(YTest == labelsTest);

Z=GraphEncoder(adjacencyData,YTrn);
ZTrn=Z(trn,:);ZTsn=Z(tsn,:);
mdl=fitcknn(ZTrn,YTrn,'Distance','Euclidean','NumNeighbors',5);
tt=predict(mdl,ZTsn);
accuracy2=mean(YTest==tt);

if optional==1
figure
cm = confusionchart(labelsTest,YTest, ...
    ColumnSummary="column-normalized", ...
    RowSummary="row-normalized");
title("GCN QM7 Confusion Chart");
end