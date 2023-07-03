%% Compute the Graph Encoder Embedding.
%% Running time is O(nK+s) where s is number of edges, n is number of vertices, and K is number of class.
%% Reference: C. Shen and Q. Wang and C. E. Priebe, "One-Hot Graph Encoder Embedding", 2022.
%%
%% @param X is either n*n adjacency, s*2 or s*3  edge list, or a cell of edgelists that share same vertex set.
%% @param Y can be either an n*1 class label vector, or a positive integer for number of clusters, or a cell array of multiple labels and multiple cluster choice.
%%        In case of partial known labels, Y should be a n*1 vector with unknown labels set to <=0 and known labels being >0.
%%        When there is no known label, set Y to be the number of clusters or a range of clusters, Y={2,3,4};
%% @param U is an n*d node attributes
%% @param opts specifies options:
%%        Normalize specifies whether to normalize each embedding by L2 norm;
%%        Laplacian specifies whether to uses graph Laplacian or adjacency matrix;
%%        Refinement specififies whether the labels are refined by classification or clustering, 
%%                   default 0 means no refinement, 1 for refinement at current dimension, and other integers for refinement into another dimension.
%%        Directed specifices whether to output directed embedding: 0 means overall embedding, 1 means sender embedding, 2 means receiver embedding.
%%        Three integers for clustering refinement: Replicates denotes the number of replicates for clustering,
%%                                       MaxIter denotes the max iteration within each replicate for encoder embedding,
%%                                       MaxIterK denotes the max iteration used within kmeans.
%%
%% @return The n*k Encoder Embedding Z and the n*1 label vector Y. 
%% @return The n*1 boolean vector indT denoting known labels.
%% @return The GEE Clustering Score (called Minimal Rank Index in paper): ranges in [0,1] and the smaller the better (only for clustering);
%%         In case of multiple graphs, all outputs become are cell array.
%%
%% @export
%%

function [Z,output]=GraphEncoder(G,Y,U,opts)
warning ('off','all');
if nargin<2
    Y={2};
end
if nargin<3
    U=0;
end
if nargin<4
    opts = struct('Normalize',true,'Laplacian',false,'Refine',0,'Directed',0,'MaxIter',30,'MaxIterK',3,'Replicates',3,'Dimension',false,'PCA',false);
end
if ~isfield(opts,'Normalize'); opts.Normalize=true; end
% if ~isfield(opts,'DiagAugment'); opts.DiagAugment=false; end
if ~isfield(opts,'Laplacian'); opts.Laplacian=false; end
if ~isfield(opts,'Refine'); opts.Refine=0; end
if ~isfield(opts,'MaxIter'); opts.MaxIter=30; end
if ~isfield(opts,'MaxIterK'); opts.MaxIterK=3; end
if ~isfield(opts,'Replicates'); opts.Replicates=3; end
if ~isfield(opts,'Directed'); opts.Directed=0; end
if ~isfield(opts,'Dimension'); opts.Dimension=false; end
if ~isfield(opts,'PCA'); opts.PCA=false; end
% opts.PCA=true; % this option is for internal testing only. Should always be set to false unless to test PCA. 
% if ~isfield(opts,'Weight'); opts.Weight=1; end
% opts.neuron=20;
% opts.activation='poslin';
% opts.Directed=1;
% opts.Refine=1;
% opts.DiagAugment=true;
% opts.Normalize=false;
% opts.Laplacian=true;

% if length(opts.Weight)~=numG
%     opts.Weight=ones(numG,1);
% end
[G,numG,n]=ProcessGraph(G,opts); % process input graph
if iscell(Y)
    numY=length(Y);
else
    if Y==0
        numY=0;
    else
        Y={Y};
        numY=1;
    end
end
Z=cell(numG,numY);
ZStd=cell(numG,numY);
classMean=cell(numG,numY);
classStd=cell(numG,numY);
dimScore=cell(numG,numY);
dimChoice=cell(numG,numY);
YNew=cell(numG,numY);
indT=cell(1,numY);
ClusterScore=cell(numG,numY);
thres1=0.01;% known label percentage less than thres will trigger ensemble clustering
thres2=1;

for j=1:numY
    tmpY=Y{j};
    [tmpY,indT{j},K,n]=ProcessLabel(tmpY,n); % process label
    ratio=sum(indT{j})/n;
    for i=1:numG
        tmpG=G{i};
        ZY=GraphEncoderEmbed(tmpG,tmpY,n,opts); % graph encoder embed
        tmpS=0;
        if ratio<=thres1
            [ZY,tmpY,tmpS]=GraphEncoderCluster(ZY,tmpY,tmpG,K,n,opts);
        end
        %         if refine<=1 % no refinement
        %             Z{i,j}=ZY;Score(i,j)=tmpS;YNew{i,j}=tmpY;
        %         else
        %             [Z{i,j},YNew{i,j},Score(i,j)]=GraphEncoderCluster(ZY,tmpY,tmpG,K,n,opts);
        %         end
        ZStd{i,j}=std(ZY);
        [classMean{i,j},classStd{i,j},dimScore{i,j}]=GraphEncoderSummary(ZY,tmpY,K,opts);
        if opts.Dimension==true
            if opts.PCA==true
                [~,tmpZ,latent]=pca(ZY);
                tmp=getElbow(latent,2);
                dimChoice{i,j}=1:tmp(end);
                ZY=tmpZ(:,1:tmp(end));
            else
                tmp=(dimScore{i,j}>thres2);
                dimChoice{i,j}=tmp;
                if sum(tmp)~=0
                    ZY=ZY(:,tmp);
                else
                    ZY=sum(ZY(:,~tmp),2);
                end
            end
        end
        Z{i,j}=ZY;ClusterScore{i,j}=tmpS;YNew{i,j}=tmpY; 
    end
end
Y=YNew;

n1=size(U,1);
if n1==n
    ZU=cell(numG,1);
    for i=1:numG
        ZU{i}=GraphFeatureEmbed(G{i},U,n,opts); % embed any feature
        ZU{i}=[ZU{i},U];
    end
    Z=[Z;ZU]; % concatenate the encoder and feature embedding
end

if size(Z,2)==1
%     if numY>0
%         indT=indT{1};
%         Y=Y(:,1);
%     end
    Z=Z(:,1);
    if size(Z,1)==1
        Z=Z{1};
%         classMean=classMean{1};classStd=classStd{1};ClusterScore=ClusterScore{1};dimScore=dimScore{1};
%         if numY>0
%         Y=Y{1};
%         end
    end
end
output=struct('Y',Y,'KnownIndices',indT,'Std',ZStd,'ClassMean',classMean,'ClassStd',classStd,'DimScore',dimScore,'ClusterScore',ClusterScore,'DimChoice',dimChoice);

%% Encoder Embedding Function
function Z=GraphEncoderEmbed(G,Y,n,opts)
if nargin<4
    opts = struct('Normalize',true,'Directed',0);
end
% sparse=true;
% prob=false;
sender=(opts.Directed < 2);
receiver=(opts.Directed ~=1);

s=size(G,1);
% if s==n
%     sparse=false;
% end
if size(Y,2)>1
    K=size(Y,2);
    %     prob=true;
else
    K=max(Y);
end
nk=zeros(1,K);
Z=zeros(n,K);
% indS=zeros(n,k);
% if prob==true && sparse==false
%     nk=sum(Y);
%     W=Y./repmat(nk,n,1);
% else
%     if sparse==false
%         W=zeros(n,K);
%         for i=1:K
%             ind=(Y==i);
%             nk(i)=sum(ind);
%             W(ind,i)=nk(i);
%         end
%     else
W=Y;
for i=1:K
    nk(i)=sum(W==i);
end
%     end
% end

% Matrix Version
% if s==n
%     Z=G*W;
% else
    % Edge List Version in O(s)
    for i=1:s
        a=G(i,1);
        b=G(i,2);
        e=G(i,3);
        %         if prob==true && sparse==false
        %             for j=1:K
        %                 Z(a,j)=Z(a,j)+e/W(b,j);
        %                 Z(b,j)=Z(b,j)+e/W(a,j);
        %             end
        %         else
        c=Y(a);
        d=Y(b);
        if d>0 && sender
            %                 if sparse==false
            %                     if c==d
            %                         Z(a,d)=Z(a,d)+e/(W(b,d)-1);
            %                     else
            %                         Z(a,d)=Z(a,d)+e/W(b,d);
            %                     end
            %                 else
            if c==d
                Z(a,d)=Z(a,d)+e/(nk(d)-1);
            else
                Z(a,d)=Z(a,d)+e/nk(d);
            end
            %                 end
        end
        if c>0 && receiver %&& a~=b
            %                 if sparse==false
            %                     if c==d
            %                         Z(b,c)=Z(b,c)+e/(W(a,c)-1);
            %                     else
            %                         Z(b,c)=Z(b,c)+e/W(a,c);
            %                     end
            %                 else
            if c==d
                Z(b,c)=Z(b,c)+e/(nk(c)-1);
            else
                Z(b,c)=Z(b,c)+e/nk(c);
            end
            %                 end
        end
        %         end
    end
% end

if opts.Normalize==true
    Z = normalize(Z,2,'norm');
    Z(isnan(Z))=0;
end

function [tmpMean,tmpStd,tmpScore]=GraphEncoderSummary(ZY,tmpY,K,opts)
[~,sz2]=size(ZY);
if opts.Normalize==false
    ZY = normalize(ZY,2,'norm');
    ZY(isnan(ZY))=0;
end
tmpMean=zeros(K,sz2);
tmpStd=zeros(K,sz2);
%             if length(tmpY)>sz1
%                 tmpY=tmpY(1:sz1);
%             end
for ii=1:K
    tmp=(tmpY==ii);
    tmpMean(ii,:)=mean(ZY(tmp,:));
    tmpStd(ii,:)=std(ZY(tmp,:));
end
tmpScore=(max(tmpMean,[],2)-min(tmpMean,[],2));
indTmp=(tmpScore>0);
tmp=tmpScore./max(tmpStd,[],2);
tmpScore(indTmp)=tmp(indTmp);
tmpScore=tmpScore';
% tmpScore=(max(tmpMean)-min(tmpMean))./stdZ;

%% Feature Embedding Function
function Z=GraphFeatureEmbed(G,U,n,opts)
if nargin<4
    opts = struct('Normalize',true,'Directed',0);
end
sender=(opts.Directed < 2);
receiver=(opts.Directed ~=1);

s=size(G,1);
K=size(U,2);
Z=zeros(n,K);

% Edge List Version in O(s)
for i=1:s
    a=G(i,1);
    b=G(i,2);
    e=G(i,3);
    for j=1:K
        if sender
            Z(a,j)=Z(a,j)+e*U(a,j);
        end
        if receiver %&& a~=b
            Z(b,j)=Z(b,j)+e*U(b,j);
        end
    end
end

if K >=2 && opts.Normalize==true
    Z = normalize(Z,2,'norm');
    Z(isnan(Z))=0;
end

%% Clustering Function
function [Z,Y,Score]=GraphEncoderCluster(Z,Y,G,K,n,opts)

% if nargin<4
%     opts = struct('Normalize',true,'MaxIter',50,'MaxIterK',5,'Replicates',3,'Directed',1,'Dim',0);
% end
Score=1;
ens=0;
Zt=Z;
Z=0;
Y(Y<1)=K+1;% set to K+1 for RandIndex function to work
for rep=1:opts.Replicates
    for r=1:opts.MaxIter
        Y1 = kmeans(Zt, K,'MaxIter',opts.MaxIterK,'Replicates',1,'Start','plus');
        %[Y3] = kmeans(Zt*WB, K,'MaxIter',opts.MaxIterK,'Replicates',1,'Start','plus');
        %gmfit = fitgmdist(Z,k, 'CovarianceType','diagonal');%'RegularizationValue',0.00001); % Fitted GMM
        %Y3 = cluster(gmfit,Z); % Cluster index
        if RandIndex(Y,Y1)==1
            break;
        else
            Y=Y1;
        end
        Zt=GraphEncoderEmbed(G,Y(:,1),n,opts);
    end
    % Compute GCS for each replicate
    tmpGCS=calculateGCS(Zt,Y1,n,K);
    if tmpGCS==Score
        Z=Z+Zt;
        ens=ens+1;
    end
    if tmpGCS<Score
        Z=Zt;Score=tmpGCS;Y=Y1;ens=1;
    end
end
% If more than one optimal solution, used the ensemble embedding for another
% k-means clustering
if ens>1
    Y = kmeans(Z, K,'MaxIter',opts.MaxIterK,'Replicates',1,'Start','plus');
    Z = GraphEncoderEmbed(G,Y(:,1),n,opts);
    Score=calculateGCS(Z,Y,n,K);
end

%% pre-precess input to s*3 then diagonal augment
function [G,numG,n]=ProcessGraph(G,opts)
if iscell(G)
    numG=length(G);
else
    G={G};
    numG=1;
end
n=1;
for i=1:numG
    tmpG=G{i};
    [s,t]=size(tmpG);
    if s==t % convert adjacency matrix to edgelist
        [tmpG,s,n]=adj2edge(tmpG);
%         n=s;
    else
        if t==2 % enlarge the edgelist to s*3
            tmpG=[tmpG,ones(s,1)];
        end
        n=max(max(max(G{i}(:,1:2))),n);
    end
%     if opts.DiagAugment==true
%         XNew=[1:n;1:n;ones(1,n)]';
%         tmpG=[tmpG;XNew];
%         s=s+n;
%     end
    if opts.Laplacian==true % convert the edge weight from raw weight to Laplacian
        D=zeros(n,1);
        for j=1:s
            a=tmpG(j,1);
            b=tmpG(j,2);
            c=tmpG(j,3);
            D(a)=D(a)+c;
            if a~=b
                D(b)=D(b)+c;
            end
        end
        D=D.^-0.5;
        for j=1:s
            tmpG(j,3)=tmpG(j,3)*D(tmpG(j,1))*D(tmpG(j,2));
        end
    end
    G{i}=tmpG;
end

%% process labels
function [Y,indT,K,n]=ProcessLabel(Y,n)
[numN,numY]=size(Y);
if numN==1 && numY==1
    if Y<2 || floor(Y)~=Y
        disp('The input dimension range is either not a integer, or smaller than 2, or bigger than vertex size');
        return;
    end
    n=max(Y,n);
    indT=0;
    K=Y;
    Y=randi(Y,n,1);
else
    if numN <n
        disp('The input label does not match input graph size');
        return;
    end
    n=max(numN,n);
    indT=(Y>0);
    YTrn=Y(indT);
    [tmp,~,YTrn]=unique(YTrn);
    K=length(tmp);
    Y(indT)=YTrn;
end

%% Adj to Edge Function
function [Edge,s,n]=adj2edge(Adj)
if size(Adj,2)<=3
    Edge=Adj;
    return;
end
n=size(Adj,1);
Edge=zeros(sum(sum(Adj>0)),3);
s=1;
for i=1:n
    for j=1:n
        if Adj(i,j)>0
            Edge(s,1)=i;
            Edge(s,2)=j;
            Edge(s,3)=Adj(i,j);
            s=s+1;
        end
    end
end
s=s-1;

%% Compute the GEE clustering score (the minimal rank index)
function tmpGCS=calculateGCS(Z,Y,n,K)
D=zeros(n,K);
for i=1:K
    D(1:n,i)=sum((Z-repmat(mean(Z(Y==i,:),1),n,1)).^2,2);
end
[~,tmpIdx]=min(D,[],2);
tmpGCS=mean(tmpIdx~=Y);
% tmp=zeros(K,1);
% for i=1:K
%     tmp(i)=sum(D(Y3==i,i));
% end
%     tmpCount=accumarray(Y3,1);
%  %   [tmpDist,tmpIdx]=mink(sum(D),2,2);
% %     tmpDist=tmpDist(:,2);tmpIdx=tmpIdx(:,2);
% %     tmp=mean(tmp(:,1)./tmp(:,2))
%   %%tmp=tmp./tmpCount./tmpDist.*(tmpCount(tmpIdx)).*tmpCount/n;
%     tmp=tmp./tmpCount./(sum(D)'-tmp).*(n-tmpCount).*tmpCount/n;
% % %2.    tmp=tmp.*(tmpCount/n);
%     %tmpRI=sum(sum(tmp));
%     tmpGCS=mean(tmp)+2*std(tmp);