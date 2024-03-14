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

function [Z,output]=GraphEncoder(G,Y,opts)
warning ('off','all');
if nargin<3
    opts = struct('Normalize',true,'Refine',0,'Principal',0,'Laplacian',false);
end
if ~isfield(opts,'Normalize'); opts.Normalize=true; end
if ~isfield(opts,'Refine'); opts.Refine=0; end
if ~isfield(opts,'Principal'); opts.Principal=0; end
if ~isfield(opts,'Laplacian'); opts.Laplacian=false; end
% opts.PCA=true; % this option is for internal testing only. Should always be set to false unless to test PCA. 
% if ~isfield(opts,'Weight'); opts.Weight=1; end
% opts.neuron=20;
% opts.activation='poslin';
% opts.Directed=1;
% opts.Refine=1;
% opts.Dimension=true;
% opts.PCA=true;
% opts.Laplacian=true;
% opts.Matrix=true;
% if length(opts.Weight)~=numG
%     opts.Weight=ones(numG,1);
% end
% opts.Normalize=false;

% Pre-Processing
[G,numG,n]=ProcessGraph(G,opts); % process input graph
if length(Y)~=n
    disp('Input Sample Size does not match Input Label Size');
    return;
end

% Pre-Processing
Z=cell(numG,1);
% YNew=cell(numG,1);

%[Y,indT,K,n,nk,indK]=ProcessLabel(Y,n); % process label

for i=1:numG
    X=G{i};
    [ZY,output,Y,idx]=GraphEncoderMain(X,Y,Y,n,opts);
    if opts.Refine>0
        K=size(ZY,2);Y2=Y;
        for r=1:opts.Refine
            Y2=Y2+idx*K;
            [ZY2,output2,Y2,idx2]=GraphEncoderMain(X,Y2,Y,n,opts);
            if sum(idx) <= sum(idx2)
                break;
            else
                ZY=ZY2;output=output2;idx=idx2;
            end
        end
    end
    Z{i}=ZY;%YNew{i}=tmpY;
end
%Y=YNew;
Z=horzcat(Z{:});

function [Z,dimMajor,Y,idx]=GraphEncoderMain(X,Y,YOri,n,opts)
[Y,indTrn,nk,indKN,indK]=ProcessLabel(Y,n);
[s,t]=size(X);
if s==t
    Z=X*indKN;
else
    Z=EdgeEncoderEmbed(X,Y,n,nk);
end
if opts.Normalize==true
    Z = normalize(Z,2,'norm');
    Z(isnan(Z))=0;
end
Z=EncoderClassifier(Z,nk,indK);

[~,YVal]=max(Z,[],2);
dimMajor=zeros(size(Z,2),1);
for d=1:size(Z,2)
    dimMajor(d)=mode(YOri( (YVal==d) & indTrn));
end

idx=(dimMajor(YVal)~=YOri);
idx= (indTrn & idx);
% tmpS=0;
% ZStd{i}=std(ZY);
% [classMean{i},classStd{i},comScore{i},comChoice{i}]=GraphEncoderSummary(ZY,indK,K,opts);
% ZY=ZY(:,comChoice{i});



%% Encoder Embedding Function
function Z=EdgeEncoderEmbed(X,Y,n,nk)

Z=zeros(n,size(nk,1));
s=size(X,1);
% edgelist version of GEE,
for i=1:s
    a=X(i,1);
    b=X(i,2);
    e=X(i,3);
    c=Y(a);
    d=Y(b);
    if d>0
        if c==d
            Z(a,d)=Z(a,d)+e/(nk(d)-1);
        else
            Z(a,d)=Z(a,d)+e/nk(d);
        end
        %                 end
    end
    if c>0
        if c==d
            Z(b,c)=Z(b,c)+e/(nk(c)-1);
        else
            Z(b,c)=Z(b,c)+e/nk(c);
        end
    end
end


%% LDA transform function
function [Z,U,V]=EncoderClassifier(Z,mk,indK)
m=sum(mk);
K=length(mk);
[~,p]=size(Z);
mu=zeros(K,p);
Sigma=zeros(p,p);
for j=1:K
    tmp=indK(:,j);
    mu(j,:)=mean(Z(tmp,:));
    Sigma=Sigma+cov(Z(tmp,:))*(mk(j)-1)/(m-K);
end
U=pinv(Sigma);
V=-(diag(mu*U*mu')*0.5-log(mk/m))';
U=U*mu';
Z=Z*U+V;

function [GEEMean,GEEStd,comScore,comChoice]=GraphEncoderSummary(Z,indK,K,opts)
thres2=0.7;
[~,sz2]=size(Z);
GEEMean=zeros(K,sz2);
GEEStd=zeros(K,sz2);
for j=1:K
    tmp=indK(:,j);
    GEEMean(j,:)=mean(Z(tmp,:));
    GEEStd(j,:)=std(Z(tmp,:));
end
comScore=(max(GEEMean,[],1)-min(GEEMean,[],1));
indTmp=(comScore>0);
tmp=comScore./max(GEEStd,[],1);
comScore(indTmp)=tmp(indTmp);
comScore=comScore';

if opts.Principal>0
    tmp=sort(comScore,'descend');
    tmp(tmp==Inf)=100;
    tmp2=getElbow(tmp,opts.Principal);
    tmp=(comScore>=max(tmp(tmp2(end)),thres2));
    comChoice=tmp';
%     if sum(tmp)~=0
%         Z=Z(:,tmp);
%     end
else
    comChoice=true(1,K);
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
    X=G{i};
    [s,t]=size(X);
    if s==t % graph is matrix input
        n=max(s,n);
        if opts.Laplacian==true 
            D=sum(X,2);
            D=D.^-0.5;
            X=D'*X*D;
        end
    else % graph is edgelist input
        if t==2
            X=[X,ones(s,1)];
        end
        n=max(max(max(X(:,1:2))),n);
        if opts.Laplacian==true % convert the edge weight from raw weight to Laplacian
            D=zeros(n,1);
            for j=1:s
                a=X(j,1);
                b=X(j,2);
                c=X(j,3);
                D(a)=D(a)+c;
                if a~=b
                    D(b)=D(b)+c;
                end
            end
            D=D.^-0.5;
            for j=1:s
                X(j,3)=X(j,3)*D(X(j,1))*D(X(j,2));
            end
        end
    end
    %     if opts.DiagAugment==true
    %         XNew=[1:n;1:n;ones(1,n)]';
    %         tmpG=[tmpG;XNew];
    %         s=s+n;
    %     end
    G{i}=X;
end

%% Process Label into unique labels
function [Y,indTrn,mk,indKN,indK]=ProcessLabel(Y,m)
% Identify training labels, and unique label
indTrn=(Y>0);   %index of training indices, i.e., non-zero labels
YTrn=Y(indTrn);
[tmp,~,YTrn]=unique(YTrn); %extract the unique labels.
K=length(tmp);
Y(indTrn)=YTrn;

% Process labels into one-hot matrices
indK=false(m,K); %one-hot encoder matrix
for j=1:K
    indK(:,j)=(Y==j);
end
mk=sum(indK)';  %number of samples per class
indKN=zeros(m,K); %normalized one-hot encoder
for j=1:K
    indKN(indK(:,j),j)=1/mk(j);
end