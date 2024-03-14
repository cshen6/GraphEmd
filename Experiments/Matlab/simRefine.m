% load('Cora.mat');X=Edge;Y=Label;simRefine(X,Y);
% load('citeseer.mat');X=Edge;Y=Label;simRefine(X,Y);
% load('email.mat'); X=AdjOri;simRefine(X,Y);
% load('Gene.mat'); X=AdjOri;simRefine(X,Y);
% load('IIP.mat'); X=Adj;simRefine(X,Y);
% load('lastfm.mat'); X=AdjOri;simRefine(X,Y);
% load('polblogs.mat');X=Adj;simRefine(X,Y);
% load('CElegans.mat');X=Ac;Y=vcols;simRefine(X,Y);
% load('Wiki_Data.mat'); Y=Label;simRefine(TE,Y);simRefine(TF,Y);simRefine(GEAdj,Y);simRefine(GFAdj,Y);simRefine({TE,TF},Y);simRefine({TE,GEAdj},Y);simRefine({TE,GEAdj,TF,GFAdj},Y);
% [X,Y]=simGenerate(400,5000,3,0); Y2=Y;Y2(Y2==3)=1;[error,tmp]=simRefine(X,Y);[error,tmp]=simRefine(X,Y2);
% [X,Y]=simGenerate(401,5000,4,0); Y2=Y;Y2(Y2==3)=1;Y2(Y2==4)=2;[error,tmp]=simRefine(X,Y);[error,tmp]=simRefine(X,Y2);
% [X,Y]=simGenerate(402,5000,6,0); Y2=Y;Y2(Y2<=3)=1;Y2(Y2==5)=4;[error,tmp]=simRefine(X,Y);[error,tmp]=simRefine(X,Y2);

function [error,tmp]=simRefine(X,Y);

indices = crossvalind('Kfold',Y,10);
K=max(Y);
error=zeros(10,3);
tmp=zeros(10,2);
norma1=1;
norma=1;
opts1=struct('Normalize',norma1,'Refine',0); 
opts2=struct('Normalize',norma1,'Refine',1); 
opts3=struct('Normalize',norma1,'Refine',10); 
opts={opts1,opts2,opts3};
for j=1:10
    tsn = (indices == j); % tst indices
    trn = ~tsn; % trn indices
    YTsn=Y(tsn);
    Y2=Y; Y2(tsn)=0;

    for i=1:3
        Z1=GraphEncoder(X,Y2,opts{i});
        if norma==1
            Z2 = normalize(Z1,2,'norm');
            Z2(isnan(Z2))=0;
            mdl=fitcdiscr(Z2(trn,:),Y(trn),'discrimType','pseudoLinear');
            YVal=predict(mdl,Z2);
            error(j,i)=mean(YVal(tsn)~=YTsn);
        else
            [~,YVal]=max(Z1,[],2);
            error(j,i)=mean(YVal(tsn)~=YTsn);
        end
    end
end
mean(error)
% mean(tmp)