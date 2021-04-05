function [mdl,filter]=GraphNN(X,Y,M,option)

% [~,ind]=checkDist(X);
% if ind==0
%     X=DCorInput(X,'euclidean');
% end
if nargin<3
    M=0;
end
if nargin<4
    option=2;
end
[Z,filter]=GraphFilter(X,Y);
if option==2
    X=Z;
end
if size(M,1)==size(X,1)
    X=[X,M];
end
net = patternnet(30); % number of neurons
net.trainParam.showWindow = false;
net.trainParam.epochs=30;
%net.layers{1}.transferFcn='poslin';
mdl = train(net,X',filter');

% not-used:
% layers = [ ...
%     sequenceInputLayer(k,'Name','seq1')
%     fullyConnectedLayer(30)
%     %reluLayer
%     %dropoutLayer(0.5)
%     fullyConnectedLayer(k)
%     softmaxLayer
%     classificationLayer];
% options = trainingOptions('sgdm', ... % stochastic gradient descent
%     'MaxEpochs',30,... % maximum number of iteration%'ValidationData',{XTsn',YTsn'}, ...%'ValidationFrequency',100, ...
%     'InitialLearnRate',0.01, ... % learning rate: slow training if too small, suboptimal or diverge result if too large
%     'Verbose',false);
% mdl = trainNetwork(Z',categorical(Y)',layers,options);