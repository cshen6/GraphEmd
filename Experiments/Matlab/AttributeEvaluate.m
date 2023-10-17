function tmp=AttributeEvaluate(X, Y, indices, eval,layer)
if nargin<3
    indices=crossvalind('Kfold',Y,10);
end
if nargin<4
    eval=1;
end
if nargin<5
    layer=20;
end
tmp=zeros(4,1);
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
%     if eval==3
%         Z=EncoderNN(X*ZTrn(1:szz,:)',YTrn(1:szz));
%         ZTrn=Z(trn,:);
%         ZTsn=Z(tsn,:);
%     end

%     if size(X,2)>100
%         Y2=onehotencode(categorical(YTrn),2)';
%         mdl3 = train(netGNN,ZTrn',Y2);
%         classes = mdl3(ZTsn'); % class-wise probability for tsting data
%         %acc_NN = perform(mdl3,Y2Tsn',classes);
%         tt = vec2ind(classes)'; % this gives the actual class for each observation
%     else
    if eval==1
        tic
%         mdl=fitcknn(ZTrn,YTrn,'NumNeighbors',5);
        mdl=fitcdiscr(ZTrn,YTrn,'discrimType',discrimType);
        tt=predict(mdl,ZTsn);
        err=mean(YTsn~=tt);
        tmp(2)=tmp(2)+toc/rep;
        tmp(1)=tmp(1)+err/rep;

        tic
        mdl=fitcknn(ZTrn,YTrn,'Distance','Euclidean','NumNeighbors',5);
        tt=predict(mdl,ZTsn);
        err=mean(YTsn~=tt);
        tmp(4)=tmp(4)+toc/rep;
        tmp(3)=tmp(3)+err/rep;
    end
    if eval==2
        tic
        mdl=fitcensemble(ZTrn,YTrn,'Method','Bag','NumLearningCycles',20);
%         mdl=fitrnet(ZTrn,YTrn);
        tt=predict(mdl,ZTsn);
        err=mean(YTsn~=tt);
        tmp(2)=tmp(2)+toc/rep;
        tmp(1)=tmp(1)+err/rep;
    end
    if eval==3
        tic
        mdl=fitcnet(ZTrn,YTrn,'LayerSizes',layer);
        tt=predict(mdl,ZTsn);
        err=mean(YTsn~=tt);
        tmp(2)=tmp(2)+toc/rep;
        tmp(1)=tmp(1)+err/rep;
    end
        %     t_AEE_NN(i)=tmp1;
end