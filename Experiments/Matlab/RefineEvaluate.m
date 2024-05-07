function [error,time]=RefineEvaluate(X,Y,indices)

if nargin<3
    indices=crossvalind('Kfold',Y,5);
end
cvf=max(indices);
error=zeros(cvf,3);
time=zeros(cvf,3);
normalize=1;
classifier=1;sof=0;
opts1=struct('Normalize',normalize,'Refine',0,'Softmax',sof); 
opts2=struct('Normalize',normalize,'Refine',1,'Softmax',sof); 
opts3=struct('Normalize',normalize,'Refine',10,'Softmax',sof); 
opts={opts1,opts2,opts3};
for j=1:cvf
    tsn = (indices == j); % tst indices
    trn = ~tsn; % trn indices
    YTsn=Y(tsn);
    Y2=Y; Y2(tsn)=0;

    for i=1:3
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
        dimClass=out.dimClass;
        if classifier==1
%             Z2 = normalize(Z1,2,'norm');
%             Z2(isnan(Z2))=0;
            mdl=fitcdiscr(Z1(trn,:),Y(trn),'discrimType','pseudoLinear');
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