function simScreening

d=100;n=1000;dd=1;
[A,Y]=simGenerate(121,n,d,1);

opts = struct('DiagA',false,'Normalize',true);
% [Z,Y]=GraphEncoder(A,Y,opts);
% 
% pcorr = corr(Z, Y)
% [~,ind]=sort(pcorr,'descend');
% 
% [dcorInd,dcorNum,pval] = DCorScreening(Z,Y)

indices = crossvalind('Kfold',Y,10);
kfold=10;
acc1=zeros(kfold,1);acc2=zeros(kfold,1);time1=zeros(kfold,1);time2=zeros(kfold,1);
for i = 1:kfold
    tsn = (indices == i); % tst indices
    trn = ~tsn; % trning indices

    YT=Y;
    YT(tsn)=-1;
    YTrn=Y(trn);
    YTsn=Y(tsn);
    [Z,YT,~,indT]=GraphEncoder(A,YT,opts);

    tic
    mdl=fitcdiscr(Z(trn,:),YTrn,'DiscrimType','pseudoLinear');
    %                mdl=fitcknn(Z(indT,:),YTrn,'Distance','euclidean','NumNeighbors',5);
    YSTesn=predict(mdl,Z(tsn,:));
    acc1(i)=mean(YSTesn~=YTsn);
    time1(i)=toc;

    tic
    pcorr = corr(Z, Y);
    [~,ind]=sort(pcorr,'descend');

%     [dcorInd,dcorNum,pval] = DCorScreening(Z,Y);
%     [~,ind]=sort(dcorNum,'descend');
    mdl=fitcdiscr(Z(trn,ind(1:dd)),YTrn,'DiscrimType','pseudoLinear');
    %                mdl=fitcknn(Z(indT,:),YTrn,'Distance','euclidean','NumNeighbors',5);
    YSTesn=predict(mdl,Z(tsn,ind(1:dd)));
    acc2(i)=mean(YSTesn~=YTsn);
    time2(i)=toc;
end
mean(acc1)
mean(acc2)
mean(time1)
mean(time2)