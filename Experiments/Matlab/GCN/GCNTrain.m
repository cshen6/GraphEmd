function time=GCNTrain(A,Y,X)

%[A,Y]=simGenerate(101,1000,20);
if nargin<3
    n=size(A,1);
    X=unifrnd(0.5,1,n,1);%as there is no vertex attribute, using a random 1-d attribute to measure the best possible time. 
end

numEpochs = 100;
learnRate = 0.01;
validationFrequency = 100;

[idxTrain,idxValidation,idxTest] = trainingPartitions(n,[0.8 0.1 0.1]);
ATrain=A(idxTrain,idxTrain);XTrain=X(idxTrain);labelsTrain=Y(idxTrain);
AValidation=A(idxValidation,idxValidation);XValidation=X(idxValidation);labelsValidation=Y(idxValidation);

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

%classes = categories(labelsTrain);
% classes = labelsTrain;
numClasses = max(Y);

sz = [numHiddenFeatureMaps numClasses];
numOut = numClasses;
numIn = numHiddenFeatureMaps;
parameters.mult3.Weights = initializeGlorot(sz,numOut,numIn,"double");

trailingAvg = [];
trailingAvgSq = [];
XTrain = dlarray(XTrain);
XValidation = dlarray(XValidation);
if canUseGPU
    XTrain = gpuArray(XTrain);
end

TTrain = onehotencode(categorical(labelsTrain),2);
TValidation = onehotencode(categorical(labelsValidation),2);

tic
for epoch = 1:numEpochs
    % Evaluate the model loss and gradients.
    [~,gradients] = dlfeval(@GCNmodelLoss,parameters,XTrain,ATrain,TTrain);

    % Update the network parameters using the Adam optimizer.
    [parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients, ...
        trailingAvg,trailingAvgSq,epoch,learnRate);

    % Update the training progress plot.
    %D = duration(0,0,toc(start),Format="hh:mm:ss");
    %title("Epoch: " + epoch + ", Elapsed: " + string(D))
%     loss = double(loss);
    %addpoints(lineLossTrain,epoch,loss)
    %drawnow

    % Display the validation metrics.
%     if epoch == 1 || mod(epoch,validationFrequency) == 0
%         YValidation = GCNmodel(parameters,XValidation,AValidation);
%         lossValidation = crossentropy(YValidation,TValidation,DataFormat="BC");

    %    lossValidation = double(lossValidation);
     %   addpoints(lineLossValidation,epoch,lossValidation)
      %  drawnow
%     end
end
time=toc;