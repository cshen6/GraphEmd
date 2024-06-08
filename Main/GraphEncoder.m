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
    opts = struct('Normalize',1,'Unbiased',0,'DiagAugment',0,'Principal',0,'Laplacian',0,'Discriminant',1);
end
if ~isfield(opts,'Normalize'); opts.Normalize=1; end
if ~isfield(opts,'Unbiased'); opts.Unbiased=0; end
if ~isfield(opts,'DiagAugment'); opts.DiagAugment=0; end
if ~isfield(opts,'Principal'); opts.Principal=0; end
if ~isfield(opts,'Laplacian'); opts.Laplacian=0; end
if ~isfield(opts,'Discriminant'); opts.Discriminant=1; end
% if ~isfield(opts,'Softmax'); opts.Softmax=false; end
% if ~isfield(opts,'BenchY'); opts.BenchY=Y; end
% if opts.Softmax
%     opts.Discriminant=true;
% end
if opts.Discriminant
    opts.Principal=0;
end

% opts.Refine=5;
% Pre-Processing
[G,n]=ProcessGraph(G,opts); % process input graph
if length(Y)~=n
    disp('Input Sample Size does not match Input Label Size');
    return;
end

[Y,indTrn,nk,indKN,indK]=ProcessLabel(Y,n);
numG=length(G);Z=cell(numG,1);
for i=1:numG
    X=G{i};
    [s,t]=size(X);
    if s==t
        tmpZ=X*indKN;
        if opts.Unbiased
            for j=1:length(nk)
                tmpZ(indK(:,j),j)=tmpZ(indK(:,j),j)*nk(j)/(nk(j)-1);
            end
        end
    else
        tmpZ=EncoderEmbedEdge(X,Y,n,nk);
    end
    if opts.Normalize
        [tmpZ,~,normZ] = normalize(tmpZ,2,'norm');
        tmpZ(isnan(tmpZ))=0;
    else
        normZ=0;
    end
    Z{i}=tmpZ;
end
Z=horzcat(Z{:});
outTransform={0,0};YVal=Y;idx=0;mu=0;Std=0;

if opts.Discriminant || opts.Principal>0
    % Apply a linear discriminant to identify which dimension corresponds to
    % which class
    [Z2,outTransform,mu,Std]=EncoderTransform(Z,nk,indK,opts);
end

if opts.Discriminant
    [~,YVal]=max(Z2,[],2);
    idx=(YVal~=Y);
    idx= (indTrn & idx);
    YVal(~indTrn)=0;
    Z=Z2;
end
if opts.Principal>0
    Z=Z2;
end

output=struct('out1',outTransform{1},'out2',outTransform{2},'mu',mu,'Std',Std,'Y',Y,'YVal',YVal,'norm',normZ,'idx',idx,'indK',indK,'nk',nk);


% thres=0.95;
% idx= (indTrn & (ZMax<thres)); %all training data where embedding probability is less than thres
% for i=1:max(opts.BenchY) %same as above, but also considering the original class
%     tmpZ=sum(Z3(:,dimClass==i),2);
%     idx= (idx & (tmpZ<thres));
% end

% K=size(indK,2);
% for i=1:K
%     tmpIdx=find(indK(:,i) & idx);
%     if length(tmpIdx)==nk(i) && nk(i)>5
%         try
%             tmpG=X(tmpIdx,tmpIdx)+0.001;
% %             (sum(tmpG,2)*sum(tmpG)).^0.5
% %             tmpG=tmpG./(sum(tmpG,2)).^0.5;
%             tmpY=kmeans(tmpG,2,'Distance','Cosine','MaxIter',30);
%             tmpY=(tmpY==1);
%             idx(tmpIdx(tmpY))=0;
%         catch
%             idx(tmpIdx)=0;
%         end
%     end
% end

%% LDA transform function + Principal Dimension Reduction
function [Z,outTransform,mu,Std]=EncoderTransform(Z,mk,indK,opts)
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
outTransform={0,0};

% Dimension reduction via Principal Community
if opts.Principal>0
    outTransform=EncoderDimension({mu,Std},opts.Principal);
    tmp=outTransform{1};
    Sigma=Sigma(tmp,tmp);
    mu=mu(:,tmp);
    Z=Z(:,tmp);
    if opts.Normalize
        Z = normalize(Z,2,'norm');
        Z(isnan(Z))=0;
    end
end

if opts.Discriminant
    U=pinv(Sigma);
    V=-(diag(mu*U*mu')*0.5-log(mk/m))';
    U=U*mu';
    Z=Z*U+V;
    outTransform={U,V};
end

%% Encoder Embedding Function
function Z=EncoderEmbedEdge(X,Y,n,nk)

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

function comChoice=EncoderDimension(S,elbow)
thres2=0.7;

mu=S{1};Std=S{2};
comScore=(max(mu,[],1)-min(mu,[],1));
indTmp=(comScore>0);
tmp=comScore./max(Std,[],1);
comScore(indTmp)=tmp(indTmp);
comScore=comScore';

tmp=sort(comScore,'descend');
tmp(tmp==Inf)=100;
tmp2=getElbow(tmp,elbow);
tmp=(comScore>=max(tmp(tmp2(end)),thres2));
comChoice={tmp,comScore};
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
        if opts.DiagAugment
            X=X+eye(s);
        end
        if opts.Laplacian
            D=sum(X,2);
            D=D.^-0.5;
            X=D'*X*D;
        end
    else % graph is edgelist input
        if t==2
            X=[X,ones(s,1)];
        end
        n=max(max(max(X(:,1:2))),n);
        if opts.Laplacian % convert the edge weight from raw weight to Laplacian
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
        if opts.DiagAugment
            XNew=[1:n;1:n;ones(1,n)]';
            X=[X;XNew];
        end
    end
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