function [error,time]=RefineEvaluate(X,Y,indices,err)

if nargin<3
    indices=crossvalind('Kfold',Y,5);
end
if nargin<4
    err=0;
end
cvf=max(indices);
error=zeros(cvf,4);
time=zeros(cvf,4);
normalize=1;
classifier=1;
opts1=struct('Normalize',normalize,'RefineK',0,'RefineY',0); 
opts2=struct('Normalize',normalize,'RefineK',3,'RefineY',3); %best!
opts3=struct('Normalize',normalize,'RefineK',5,'RefineY',5); %best?
opts4=struct('Normalize',normalize,'RefineK',10,'RefineY',10);
opts={opts1,opts2,opts3,opts4};
K=max(Y);
for j=1:cvf
    tsn = (indices == j); % tst indices
    trn = ~tsn; % trn indices
    YTsn=Y(tsn);
    Y2=Y; Y2(tsn)=0;
    if err>0
        nrr=rand(length(trn),1)<err;
        Y(trn(nrr))=randi(K,sum(trn(nrr)),1);
    end

    for i=1:4
        tic
        [Z1,out]=RefinedGEE(X,Y2,opts{i});
        % if iscell(Z1)
        % Z1=horzcat(Z1{:});
        % end
        % if i>1
        % YNew=out.Y; idx=out.idx;
        % YNew(idx)=YNew(idx)+max(YNew); 
        % % YNew(idx)=max(YNew):max(YNew)+sum(idx)-1;
        % [Z1,out]=GraphEncoder(X,YNew,opts1);
        % end

        % Z1=[Z1,out.norm];
%         Z1=X*Z1;
        % dimClass=out.dimClass;
        if classifier==1
%             Z2 = normalize(Z1,2,'norm');
%             Z2(isnan(Z2))=0;
            % [~,Z1]=pca(Z1,'NumComponents',min(100,size(Z1,2)));
            mdl=fitcdiscr(Z1(trn,:),Y(trn),'discrimType','pseudoLinear');
            % mdl=fitcnet(Z1(trn,:),Y(trn),'LayerSizes',10*max(Y));
            YVal=predict(mdl,Z1);
            error(j,i)=mean(YVal(tsn)~=YTsn);
        else
            [~,YVal]=max(Z1,[],2);
            YVal=dimClass(YVal);
            % for k=1:max(Y) %same as above, but also considering the original class
            %     tmpZ=sum(Z1(:,dimClass==k),2);
            %     Z1(:,k)=tmpZ;
            % end
            % Z1=Z1(:,1:max(Y));
            % [~,YVal]=max(Z1,[],2);
            error(j,i)=mean(YVal(tsn)~=YTsn);
        end
        time(j,i)=toc;
    end
end
% mean(error)
% mean(time)
% mean(tmp)