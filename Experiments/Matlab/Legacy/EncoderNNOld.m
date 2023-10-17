function [Z,output]=EncoderNNOld(G,Y,opts)
warning ('off','all');
if nargin<2
    Y={2};
end
if nargin<4
    opts = struct('Normalize',false,'DiagAugment',true,'Laplacian',false,'Refine',0,'Directed',0,'MaxIter',30,'MaxIterK',3,'Replicates',3,'Dimension',false,'PCA',false);
end
if ~isfield(opts,'Normalize'); opts.Normalize=false; end
if ~isfield(opts,'DiagAugment'); opts.DiagAugment=true; end
if ~isfield(opts,'Laplacian'); opts.Laplacian=false; end
if ~isfield(opts,'Refine'); opts.Refine=0; end
if ~isfield(opts,'MaxIter'); opts.MaxIter=30; end
if ~isfield(opts,'MaxIterK'); opts.MaxIterK=3; end
if ~isfield(opts,'Replicates'); opts.Replicates=3; end
if ~isfield(opts,'Directed'); opts.Directed=0; end
if ~isfield(opts,'Dimension'); opts.Dimension=false; end
if ~isfield(opts,'PCA'); opts.PCA=false; end
% opts.Normalize=true;
% process input graph
[G,numG]=ProcessGraph(G,Y); 

% initialize outputs
Z=cell(numG,1);
ZStd=cell(numG,1);
classMean=cell(numG,1);
classStd=cell(numG,1);
dimScore=cell(numG,1);
% dimChoice=cell(numG,1);
% YNew=cell(numG,1);

for i=1:numG
    % take the output 
    tmpG=G{1,i};
    tmpY=G{2,i};
    tmpnk=G{3,i};
    tmpK=G{4,i};
    % Encoder embedding
    ZY=GraphEncoderEmbed(tmpG,tmpY,tmpnk,opts); 
    ZStd{i}=std(ZY);
    %[classMean{i},classStd{i},dimScore{i}]=GraphEncoderSummary(ZY,tmpY,tmpK);
    %mu=classMean{i};
    %ZY=ZY;%*(mu-mean(mu));
    if opts.Normalize==true
        ZY = normalize(ZY,2,'norm');
        %     ZY=ZY./sum(ZY,1);
        ZY(isnan(ZY))=0;
    end
    Z{i}=ZY;
end
%Y=YNew;

if size(Z,2)==1
    Z=Z(:,1);
    if size(Z,1)==1
        Z=Z{1};
    end
end
output=0;
%output=struct('Y',G{2},'mk',G{3}, 'Std',ZStd,'ClassMean',classMean,'ClassStd',classStd,'DimScore',dimScore);

%% Encoder Embedding Function
function Z=GraphEncoderEmbed(G,Y,mk,opts)
if size(Y,2)>1
    K=size(Y,2);
else
    K=max(Y);
end
[~,m]=size(G);
W=zeros(m,K);
for i=1:K
    ind=(Y==i);
    W(ind,i)=1/mk(i);
end
Z=G*W;
% for i=1:n
%     tmpL=Y(i);
%     if tmpL>0
%         Z(i,tmpL)=Z(i,tmpL)*mk(tmpL)/(mk(tmpL)-1);
%     end
% end

if opts.Normalize==true
    Z = normalize(Z,2,'norm');
%     ZY=ZY./sum(ZY,1);
    Z(isnan(Z))=0;
end

% function [tmpMean,tmpStd,tmpScore,Z]=GraphEncoderSummary(Z,tmpY,K)
% [~,sz2]=size(Z);
% tmpMean=zeros(K,sz2);
% tmpStd=zeros(K,sz2);
% %             if length(tmpY)>sz1
% %                 tmpY=tmpY(1:sz1);
% %             end
% for ii=1:K
%     size(Z)
%     tmp=(tmpY==ii);
%     size(tmpY)
%     tmpMean(ii,:)=mean(Z(tmp,:));
%     tmpStd(ii,:)=std(Z(tmp,:));
% end
% tmpScore=(max(tmpMean,[],1)-min(tmpMean,[],1));
% indTmp=(tmpScore>0);
% tmp=tmpScore./max(tmpStd,[],1);
% tmpScore(indTmp)=tmp(indTmp);
% tmpScore=tmpScore';

% if opts.Normalize==true
%     tmpMean = normalize(tmpMean,2,'norm');
% %     ZY=ZY./sum(ZY,1);
%     tmpMean(isnan(tmpMean))=0;
% end
% tmpScore=(max(tmpMean)-min(tmpMean))./stdZ;
% if opts.Normalize==true
%     Z = normalize(Z,2,'norm');
% %     ZY=ZY./sum(ZY,1);
%     Z(isnan(Z))=0;
% end

%% pre-precess input to s*3 then diagonal augment
function [G2,numG,n]=ProcessGraph(G,Y)
if iscell(G)
    numG=length(G);
else
    G={G};
    numG=1;
end
G2=cell(3,numG);
G2(1,:)=G;

n=zeros(numG,1);
m=zeros(numG,1);
for i=1:numG
    tmpG=G{i};
    [n(i),m(i)]=size(tmpG);
end
if std(n)>0
    disp('Input sample sizes do not match across different graphs');
else
    n=mean(n);
end

if iscell(Y)
    numY=length(Y);
    if numG~=numY
        disp('The number of labels not match the number of graphs');
        return;
    end
else
    Y={Y};
    numY=1;
end

for i=1:numG
    if i==1 || numY==numG
        tmpY=Y{i};
        indT=(tmpY>0);
        YTrn=tmpY(indT);
        [tmp,~,YTrn]=unique(YTrn);
        K=length(tmp);
        tmpY(indT)=YTrn;
        mk=zeros(K,1);
        for j=1:K
            mk(j)=sum(tmpY==j);
        end
    end
    if length(tmpY)~=m(i)
        disp('The input label size does not match input sample size');
        return;
    end
    G2{2,i}=tmpY;
    G2{3,i}=mk;
    G2{4,i}=K;
end

%% Compute the GEE clustering score (the minimal rank index)
function tmpGCS=calculateGCS(Z,Y,n,K)
D=zeros(n,K);
for i=1:K
    D(1:n,i)=sum((Z-repmat(mean(Z(Y==i,:),1),n,1)).^2,2);
end
[~,tmpIdx]=min(D,[],2);
tmpGCS=mean(tmpIdx~=Y);