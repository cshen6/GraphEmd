function tmp=AttributeEvaluate(X, Y, indices)
if nargin<3
    indices=crossvalind('Kfold',Y,10);
end
tmp=zeros(2,1);
% if size(X,2)>100
%     netGNN = patternnet(30,'trainscg','crossentropy'); % number of neurons, Scaled Conjugate Gradient, cross entropy
%     netGNN.layers{1}.transferFcn = 'poslin';
%     netGNN.trainParam.showWindow = false;
%     netGNN.trainParam.epochs=100;
%     netGNN.divideParam.trainRatio = 0.9;
%     netGNN.divideParam.valRatio   = 0.1;
%     netGNN.divideParam.testRatio  = 0/100;
% end
rep=max(indices);
for j=1:rep
    tsn = (indices == j); % tst indices
    trn = ~tsn; % trning indices
    ZTrn=X(trn,:);
    ZTsn=X(tsn,:);
    YTrn=Y(trn);
    YTsn=Y(tsn);

%     if size(X,2)>100
%         Y2=onehotencode(categorical(YTrn),2)';
%         mdl3 = train(netGNN,ZTrn',Y2);
%         classes = mdl3(ZTsn'); % class-wise probability for tsting data
%         %acc_NN = perform(mdl3,Y2Tsn',classes);
%         tt = vec2ind(classes)'; % this gives the actual class for each observation
%     else
    tic
        mdl=fitcknn(ZTrn,YTrn,'NumNeighbors',5);
        tt=predict(mdl,ZTsn);
        %     t_AEE_NN(i)=tmp1;
%     end
    ttt=toc;
    tmp(1)=tmp(1)+mean(YTsn~=tt)/rep;
    tmp(2)=tmp(2)+ttt/rep;
end