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
    opts = struct('Normalize',true,'Refine',0,'Principal',0,'Laplacian',false,'Discriminant',true,'Softmax',false);
end
if ~isfield(opts,'Normalize'); opts.Normalize=true; end
if ~isfield(opts,'Refine'); opts.Refine=0; end
if ~isfield(opts,'Principal'); opts.Principal=0; end
if ~isfield(opts,'Laplacian'); opts.Laplacian=false; end
if ~isfield(opts,'Discriminant'); opts.Discriminant=true; end
if ~isfield(opts,'Softmax'); opts.Softmax=false; end

% Pre-Processing
[G,n]=ProcessGraph(G,opts); % process input graph
if length(Y)~=n
    disp('Input Sample Size does not match Input Label Size');
    return;
end

% Initial Graph Encoder Embedding
[Z,dimMajor,Y,idx,comChoice]=GraphEncoderMain(G,Y,Y,n,opts);

% Refined Graph Encoder Embedding
if opts.Refine>0
    K=size(Z,2);Y2=Y;
    for r=1:opts.Refine
        Y2=Y2+idx*K;
        [Z2,dimMajor2,Y2,idx2,comChoice]=GraphEncoderMain(G,Y2,Y,n,opts);
        if sum(idx) <= sum(idx2)
            break;
        else
            Z=Z2;dimMajor=dimMajor2;idx=idx2;
        end
    end
end
%     Z{i}=Z;%YNew{i}=tmpY;
%Y=YNew;
output={dimMajor,comChoice{1},comChoice{2}};

%% Graph Encoder Embedding
function [Z,dimMajor,Y,idx,comChoice]=GraphEncoderMain(G,Y,YOri,n,opts)
[Y,indTrn,nk,indKN,indK]=ProcessLabel(Y,n);

numG=length(G);Z=cell(numG,1);
for i=1:numG
    X=G{i};
    [s,t]=size(X);
    if s==t
        tmpZ=X*indKN;
    else
        tmpZ=EdgeEncoderEmbed(X,Y,n,nk);
    end
    if opts.Normalize
        tmpZ = normalize(tmpZ,2,'norm');
        tmpZ(isnan(tmpZ))=0;
    end
    Z{i}=tmpZ;
end
Z=horzcat(Z{:});

% Apply a linear discriminant to identify which dimension corresponds to
% which class
[Z2,~,~,S]=EncoderDiscriminant(Z,nk,indK,opts);
[~,YVal]=max(Z2,[],2);
dimMajor=zeros(size(Z2,2),1);
for d=1:size(Z2,2)
    dimMajor(d)=mode(YOri( (YVal==d) & indTrn));
end
% The training indices that is mis-classfied by LDA
idx=(dimMajor(YVal)~=YOri);
idx= (indTrn & idx);
if opts.Discriminant
    Z=Z2;
end

% Find principal communities
if opts.Principal
    comChoice=EncoderDimension(S);
    % if there is only one graph, and no discriminant transformation is
    % done, keep only the principal dimensions
    if ~opts.Discriminant
        Z=Z(:,comChoice{2});
    end
else
    comChoice={0,0};
end
% tmpS=0;
% ZStd{i}=std(ZY);
% [classMean{i},classStd{i},comScore{i},comChoice{i}]=GraphEncoderSummary(ZY,indK,K,opts);
% ZY=ZY(:,comChoice{i});

%% LDA transform function
function [Z,U,V,S]=EncoderDiscriminant(Z,mk,indK,opts)
m=sum(mk);
K=length(mk);
[~,p]=size(Z);
mu=zeros(K,p);
Sigma=zeros(p,p);
Std=zeros(K,p);
for j=1:K
    tmp=indK(:,j);
    mu(j,:)=mean(Z(tmp,:));
    Std(j,:)=std(Z(tmp,:));
    Sigma=Sigma+cov(Z(tmp,:))*(mk(j)-1)/(m-K);
end
U=pinv(Sigma);
V=-(diag(mu*U*mu')*0.5-log(mk/m))';
U=U*mu';
Z=Z*U+V;
if opts.Softmax
    Z=softmax(Z')';
end
S={mu,Std};

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


function comChoice=EncoderDimension(S)
thres2=0.7;

mu=S{1};Std=S{2};
comScore=(max(mu,[],1)-min(mu,[],1));
indTmp=(comScore>0);
tmp=comScore./max(Std,[],1);
comScore(indTmp)=tmp(indTmp);
comScore=comScore';

tmp=sort(comScore,'descend');
tmp(tmp==Inf)=100;
tmp2=getElbow(tmp,3);
tmp=(comScore>=max(tmp(tmp2(end)),thres2));
comChoice={comScore,tmp};
%     if sum(tmp)~=0
%         Z=Z(:,tmp);
%     end


%% pre-precess input to s*3 then diagonal augment
function [G,n]=ProcessGraph(G,opts)
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

% Given a decreasingly sorted vector, return the given number of elbows
% dat: a input vector (e.g. a vector of standard deviations), or a input feature matrix.
% n: the number of returned elbows.
% q: a vector of length n. Typically 1st or 2nd elbow suffices
% Reference: Zhu, Mu and Ghodsi, Ali (2006), "Automatic dimensionality selection from the scree plot via the use of profile likelihood", Computational Statistics & Data Analysis, Volume 51 Issue 2, pp 918-930, November, 2006.
function q=getElbow(d, n)
if nargin<2
    n=3;
end
p=length(d);
q=getElbow2(d);
for i=2:n
    if q(i-1)>=p
        break;
    else
        q=[q,q(i-1)+getElbow2(d(q(i-1)+1:end))];
    end
end
if length(q)<n
    q=[q,q(end)*ones(1,n-length(q))];
end

function q=getElbow2(d)
p=length(d);
lq=zeros(p,1);
for i=1:p
    mu1 = mean(d(1:i));
    mu2 = mean(d(i+1:end));
    sigma2 = (sum((d(1:i) - mu1).^2) + sum((d(i+1:end) - mu2).^2)) / (p - 1 - (i < p));
    lq(i) = sum( log(normpdf(  d(1:i), mu1, sqrt(sigma2)))) + sum( log(normpdf(  d(i+1:end), mu2, sqrt(sigma2))));
end
[~,q]=max(lq);