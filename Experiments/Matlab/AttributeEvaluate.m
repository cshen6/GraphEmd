function tmp=AttributeEvaluate(X, Y, indices, eval)
if nargin<3
    indices=crossvalind('Kfold',Y,10);
end
if nargin<4
    eval=1;
end
tmp=zeros(2,1);
discrimType='pseudoLinear';
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
    if eval==1;
%         mdl=fitcknn(ZTrn,YTrn,'NumNeighbors',5);
        mdl=fitcdiscr(ZTrn,YTrn,'discrimType',discrimType);
        tt=predict(mdl,ZTsn);
        err=mean(YTsn~=tt);
    end
    if eval==2;
        mdl=fitrensemble(ZTrn,YTrn);
%         mdl=fitrnet(ZTrn,YTrn);
        tt=predict(mdl,ZTsn);
        err =sum((YTsn-tt).^2)/sum((YTsn-mean(YTsn)).^2);
    end
        %     t_AEE_NN(i)=tmp1;
%     end
    ttt=toc;
    tmp(1)=tmp(1)+err/rep;
    tmp(2)=tmp(2)+ttt/rep;
end