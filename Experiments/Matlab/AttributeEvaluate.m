function Error=AttributeEvaluate(X, Y, indices, eval,layer)
if nargin<3
    indices=crossvalind('Kfold',Y,5);
end
if nargin<4
    eval=1;
end
if nargin<5
    layer=20;
end
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
optGraph=1;
if iscell(X)
    G=X{2};
    X=X{1};
    optGraph=3;
end
ZTrn=cell(3,optGraph);
ZTsn=cell(3,optGraph);
Error=zeros(4,optGraph);
opts1=struct('Refine',0); 
for j=1:rep
    tsn = (indices == j); % tst indices
    trn = ~tsn; % trning indices
    ZTrn{1}=X(trn,:);
    ZTsn{1}=X(tsn,:);

    YTrn=Y(trn);
    YTsn=Y(tsn);
    if optGraph==3
        Y2=Y;Y2(tsn)=0;
        Z2=GraphEncoder(G,Y2,opts1);
        ZTrn{2}=Z2(trn,:); ZTrn{3}=[X(trn,:),Z2(trn,:)];
        ZTsn{2}=Z2(tsn,:); ZTsn{3}=[X(tsn,:),Z2(tsn,:)];
    end
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
        for i=1:optGraph
            tic
            %         mdl=fitcknn(ZTrn,YTrn,'NumNeighbors',5);
            mdl=fitcdiscr(ZTrn{i},YTrn,'discrimType',discrimType);
            tt=predict(mdl,ZTsn{i});
            err=mean(YTsn~=tt);
            Error(2,i)=Error(2,i)+toc/rep;
            Error(1,i)=Error(1,i)+err/rep;

%             tic
%             mdl=fitcknn(ZTrn{i},YTrn,'Distance','Euclidean','NumNeighbors',5);
%             tt=predict(mdl,ZTsn{i});
%             err=mean(YTsn~=tt);
%             Error(4,i)=Error(4,i)+toc/rep;
%             Error(3,i)=Error(3,i)+err/rep;
        end
    end
     if eval==2
        for i=1:optGraph
            tic
                    mdl=fitcknn(ZTrn{i},YTrn,'NumNeighbors',5);
            % mdl=fitcdiscr(ZTrn{i},YTrn,'discrimType',discrimType);
            tt=predict(mdl,ZTsn{i});
            err=mean(YTsn~=tt);
            Error(2,i)=Error(2,i)+toc/rep;
            Error(1,i)=Error(1,i)+err/rep;

%             tic
%             mdl=fitcknn(ZTrn{i},YTrn,'Distance','Euclidean','NumNeighbors',5);
%             tt=predict(mdl,ZTsn{i});
%             err=mean(YTsn~=tt);
%             Error(4,i)=Error(4,i)+toc/rep;
%             Error(3,i)=Error(3,i)+err/rep;
        end
    end
    if eval==4
        for i=1:optGraph
            tic
            mdl=fitcensemble(ZTrn{i},YTrn,'Method','Bag','NumLearningCycles',20);
            %         mdl=fitrnet(ZTrn,YTrn);
            tt=predict(mdl,ZTsn{i});
            err=mean(YTsn~=tt);
            Error(2,i)=Error(2,i)+toc/rep;
            Error(1,i)=Error(1,i)+err/rep;
        end
    end
    if eval==3
        for i=1:optGraph
            tic
            mdl=fitcnet(ZTrn{i},YTrn,'LayerSizes',layer);
            tt=predict(mdl,ZTsn{i});
            err=mean(YTsn~=tt);
            Error(2,i)=Error(2,i)+toc/rep;
            Error(1,i)=Error(1,i)+err/rep;
        end
    end
    %     t_AEE_NN(i)=tmp1;
end