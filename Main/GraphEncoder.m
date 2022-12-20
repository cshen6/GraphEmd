%% Compute the Graph Encoder Embedding.
%% Running time is O(nK+s) where s is number of edges, n is number of vertices, and K is number of class.
%% Reference: C. Shen and Q. Wang and C. E. Priebe, "One-Hot Graph Encoder Embedding", 2022.
%%
%% @param X is either n*n adjacency, or s*3 edge list. Vertex size should be >10.
%%        Adjacency matrix can be weighted or unweighted, directed or undirected. It will be converted to s*3 edgelist.
%%        Edgelist input can be either s*2 or s*3, and complexity in O(s).
%% @param Y is either an n*1 class label vector, or a positive integer for number of clusters, or a range of potential cluster size, i.e., [2,10].
%%        In case of partial known labels, Y should be a n*1 vector with unknown labels set to <=0 and known labels being >0.
%%        When there is no known label, set Y to be the number of clusters or a range of clusters. 
%%        The maximum K should not be too large relative to n (ideally n/max(K)>15).
%% @param opts specifies options:
%%        DiagA = true means adding 1 to all diagonal entries (i.e., add self-loop to edgelist), which can help sparse graphs. However, if graph weights are very close to 0, this option can introduce significant within-group bias.
%%        Normalize specifies whether to normalize each embedding by L2 norm;
%%        Laplacian specifies whether to uses graph Laplacian or adjacency matrix;
%%        Directed specifices whether to output directed embedding: 1 means overall embedding, 2 means sender and receiver embedding, 3 has all 3 embedding.
%%        Three integers for clustering: Replicates denotes the number of replicates for clustering,
%%                                       MaxIter denotes the max iteration within each replicate for encoder embedding,
%%                                       MaxIterK denotes the max iteration used within kmeans.
%%
%% @return The n*k Encoder Embedding Z; the n*k Encoder Transformation: W; the n*1 label vector: Y;
%% @return The n*1 boolean vector for known label: indT (only for classification);
%% @return The GEE Clustering Score (called Minimal Rank Index in paper): ranges in [0,1] and the smaller the better (only for clustering);
%%
%% @export
%%

function [Z,Y,W,indT,GCS]=GraphEncoder(X,Y,opts)
warning ('off','all');
if nargin<2
    Y=2:5;
end
if nargin<3
    opts = struct('DiagA',true,'Normalize',true,'Laplacian',false,'Learner',1,'LearnIter',0,'MaxIter',30,'MaxIterK',3,'Replicates',3,'Attributes',0,'Directed',1,'Dim',0,'Weight',1,'Sparse',false);
end
if ~isfield(opts,'DiagA'); opts.DiagA=true; end
if ~isfield(opts,'Normalize'); opts.Normalize=true; end
if ~isfield(opts,'Laplacian'); opts.Laplacian=false; end
if ~isfield(opts,'Learner'); opts.Learner=1; end
if ~isfield(opts,'LearnIter'); opts.LearnIter=0; end
if ~isfield(opts,'MaxIter'); opts.MaxIter=20; end
if ~isfield(opts,'MaxIterK'); opts.MaxIterK=2; end
if ~isfield(opts,'Replicates'); opts.Replicates=1; end
if ~isfield(opts,'Attributes'); opts.Attributes=0; end
if ~isfield(opts,'Directed'); opts.Directed=1; end
if ~isfield(opts,'Dim'); opts.Dim=0; end
if ~isfield(opts,'Weight'); opts.Weight=1; end
% if ~isfield(opts,'Sparse'); opts.Sparse=false; end
opts.neuron=20;
elbN=2;
opts.activation='poslin';
U=opts.Attributes;
% opts.Directed=1;
directed=opts.Directed;
GCS=1;
% if ~isfield(opts,'distance'); opts.distance='Normalize'; end
% opts.DiagA=false;
% opts.Normalize=false;
% opts.Laplacian=true;
% opts.Sparse=true;

%% pre-precess input to s*3 then diagonal augment
if iscell(X)
    num=length(X);
else
    X={X};
    num=1;
end
if length(opts.Weight)~=num
    opts.Weight=ones(num,1);
end

for i=1:num
    [s,t]=size(X{i});
    if s==t % convert adjacency matrix to edgelist
        [X{i},s,n]=adj2edge(X{i});
    else
        if t==2 % enlarge the edgelist to s*3
            X{i}=[X{i},ones(s,1)];
            %         n=max(max(X{i}));
            %         t=3;
        end
        n=max(max(X{i}));
    end
    %     n=max(max(X{1}(:,1:2)));
    if opts.DiagA==true
        XNew=[1:n;1:n;ones(1,n)]';
        X{i}=[X{i};XNew];
        s=s+n;
    end
    if opts.Laplacian==true
        D=zeros(n,1);
        for j=1:s
            a=X{i}(j,1);
            b=X{i}(j,2);
            c=X{i}(j,3);
            D(a)=D(a)+c;
            if a~=b
                D(b)=D(b)+c;
            end
        end
        D=D.^-0.5;
        for j=1:s
            X{i}(j,3)=X{i}(j,3)*D(X{i}(j,1))*D(X{i}(j,2));
        end
    end
end
if size(U,1)==n
    attr=true;
else
    attr=false;
end
%% partial or full known labels when label size matches vertex size, do embedding / classification directly
if length(Y)==n
    indT=(Y>0);
    YTrn=Y(indT);
    [~,~,YTrn]=unique(YTrn);
    Y(indT)=YTrn;
    YTrn2=onehotencode(categorical(YTrn),2)';
    K=max(Y);
    dim=K;
    if opts.Dim>0
        dim=min(opts.Dim,dim);
    end

%     if opts.Sparse==false
        Z=zeros(n,dim*num,directed);
%     else
%         Z=sparse(n,dim*num,directed);
%     end
    W=cell(1,num);
    if opts.Learner==2
        % initialize NN
        netGNN = patternnet(max(opts.neuron,dim),'trainscg','crossentropy'); % number of neurons, Scaled Conjugate Gradient, cross entropy
        %         netGNN.layers{1}.transferFcn = opts.activation;
        netGNN.trainParam.showWindow = false;
        netGNN.trainParam.epochs=100;
        netGNN.divideParam.trainRatio = 0.9;
        netGNN.divideParam.valRatio   = 0.1;
        netGNN.divideParam.testRatio  = 0;
    end
    if opts.LearnIter<1
        for i=1:num
            [tmpZ,W{i}]=GraphEncoderEmbed(X{i},Y,n,opts);
            Z(:,(i-1)*dim+1:i*dim,:)=tmpZ*opts.Weight(i);
        end
%         %% Simple PCA
%         if opts.Dim>0 && opts.Dim<=K
%             if opts.Sparse==true
%                Z=full(Z);
%             end
%             for i=1:directed
%                 [coeff,Z]=pca(Z(:,:,i),'NumComponents',opts.Dim);
%                 if opts.Dim==K
%                     elb=getElbow(diag(coeff),elbN);
%                     Z=Z(:,1:elb(elbN),i);
%                 end
%             end
%         end
        if attr==true
            Z=[Z,repmat(U,1,directed)];
        end
        if sum(indT)<n
            if opts.Learner==1
                mdl=fitcdiscr(Z(indT,:,directed),YTrn,'DiscrimType','pseudoLinear');
                %                mdl=fitcknn(Z(indT,:),YTrn,'Distance','euclidean','NumNeighbors',5);
                Y(~indT)=predict(mdl,Z(~indT,:,directed));
            else
                mdl = train(netGNN,Z(indT,:,directed)',YTrn2);
                prob=mdl(Z(~indT,:,directed)');
                Y(~indT)=vec2ind(prob)';
                %             [~,Y(~indT)] = max(prob,[],1); % class-wise probability for tsting data
            end
        end
    else
        %
        %         indNew=indT;
        GCS=0;Y1=Y;
        for rep=1:opts.Replicates
            tmp=randi([1,K],[sum(~indT),1]);
            Y1(~indT)=tmp;
            Y2=onehotencode(categorical(Y1),2);
            %             Y3=double(Y1);
            for r=1:opts.LearnIter
                %             i
                for i=1:num
                    [tmpZ,W{i}]=GraphEncoderEmbed(X{i},Y1,n,opts); % discrete label version
                    Z(:,(i-1)*K+1:i*K,directed)=tmpZ*opts.Weight(i);
                    %[Z(:,(i-1)*K+1:i*K),W{i}]=GraphEncoderEmbed(X{i},Y2,n,opts); % probability version
                end
%                 %% Simple PCA
%                 if opts.Dim>0 && opts.Dim<=K
%                     if opts.Sparse==true
%                         Z=full(Z);
%                     end
%                     [coeff,Z]=pca(Z,'NumComponents',opts.Dim*directed);
%                     if opts.Dim==K
%                         elb=getElbow(diag(coeff),elbN);
%                         Z=Z(:,1:elb(elbN));
%                     end
%                 end
                %                 [Z,W]=GraphEncoderEmbed(X,Y2,n,opts);
                if attr==true && r==1
                    Z=[Z,repmat(U,1,directed)];
                end
                if opts.Learner==1
                    mdl=fitcdiscr(Z(indT,:,directed),YTrn,'DiscrimType','pseudoLinear');
                    [class,prob] = predict(mdl,Z(~indT,:,directed));
                    prob1=max(prob,[],2);
                else
                    mdl = train(netGNN,Z(indT,:,directed)',YTrn2);
                    prob=mdl(Z(~indT,:,directed)');
                    %                     class=vec2ind(prob)';
                    [prob1,class] = max(prob,[],1); % class-wise probability for tsting data
                end
                if RandIndex(Y1(~indT),class)==1
                    break;
                else
                    Y2(~indT,:)=prob';
                    %                     Y3(~indT)=prob1;
                    Y1(~indT)=class;
                end
            end
            minP=mean(prob1)-3*std(prob1);
            if minP>GCS
                GCS=minP;Y=Y1;
            end
        end
    end
else
    %% otherwise do clustering
    indT=zeros(n,1);
    K=Y;
    if n/max(K)<10
        disp('Too many clusters at maximum range. Result may bias towards large K when n/max(K) <10.')
    end
    %% when a given cluster size is specified
    if length(K)==1
        [Z,Y,W,GCS]=GraphEncoderCluster(X,K,n,num,attr,opts);
    else
        %% when a range of cluster size is specified
        K=sort(K); % ensure increasing K
        if length(K)<n/2 && max(K)<max(n/2,10)
            tmpGCS=1;Z=0;W=0;GCS=zeros(length(K),1);
            for r=1:length(K)
                [Zt,Yt,Wt,tmp]=GraphEncoderCluster(X,K(r),n,num,attr,opts);
                GCS(r)=tmp;
                if tmp<=tmpGCS
                    tmpGCS=tmp;Y=Yt;Z=Zt;W=Wt;
                end
            end
        end
    end
end

%% Clustering Function
function [Z,Y,W,GCS]=GraphEncoderCluster(X,K,n,num,attr,opts)

% if nargin<4
%     opts = struct('Normalize',true,'MaxIter',50,'MaxIterK',5,'Replicates',3,'Directed',1,'Dim',0);
% end
elbN=2;
directed=opts.Directed;
GCS=1;
% if opts.Sparse==false
    Zt=zeros(n,K*num,directed);
%     ZtC=zeros(n,K*num,directed);
% else
%     Zt=sparse(n,K*num*directed);
% end

Wt=cell(1,num);
W=cell(1,num);
dim=K;
if opts.Dim>0
    dim=min(opts.Dim,dim);
end

ens=0;
for rep=1:opts.Replicates
    Y2=randi([1,K],[n,1]);Y2=repmat(Y2,1,directed);Y3=randi([1,K],[n,directed]);
    for r=1:opts.MaxIter
        for i=1:num
            [tmpZ,Wt{i}]=GraphEncoderEmbed(X{i},Y2(:,1),n,opts);
            Zt(:,(i-1)*K+1:i*K,:)=tmpZ*opts.Weight(i);
        end
        %% Simple PCA
%         if opts.Dim>0 && opts.Dim<=K
%             if opts.Sparse==true
%                Zt=full(Zt);
%             end
%             [coeff,Zt]=pca(Zt,'NumComponents',opts.Dim*directed);
%             if opts.Dim==K
%                 elb=getElbow(diag(coeff),elbN);
%                 Zt=Zt(:,1:elb(elbN));
%             end
%         end
        if attr==true && r==1
            Zt=[Zt,repmat(U,1,directed)];
        end
        for i=1:directed
            Y3(:,i) = kmeans(Zt(:,:,i), K,'MaxIter',opts.MaxIterK,'Replicates',1,'Start','plus');
            %[Y3] = kmeans(Zt*WB, K,'MaxIter',opts.MaxIterK,'Replicates',1,'Start','plus');
            %gmfit = fitgmdist(Z,k, 'CovarianceType','diagonal');%'RegularizationValue',0.00001); % Fitted GMM
            %Y3 = cluster(gmfit,Z); % Cluster index
            if RandIndex(Y2(:,i),Y3(:,i))==1
                break;
            else
                Y2(:,i)=Y3(:,i);
            end
        end
    end
    % Re-Embed using final label Y3
    for i=1:num
        for j=1:directed
            [tmpZ,Wt{i}]=GraphEncoderEmbed(X{i},Y3(:,j),n,opts);
            Zt(:,(i-1)*K+1:i*K,j)=tmpZ(:,:,j)*opts.Weight(i);
        end
    end
    % Compute GCS for each replicate
    tmpGCS=calculateGCS(Zt(:,:,directed),Y3(:,directed),n,K);
    if tmpGCS==GCS
        Z=Z+Zt;
        ens=ens+1;
    end
    if tmpGCS<GCS
        Z=Zt;W=Wt;GCS=tmpGCS;Y=Y3;ens=1;
    end
end
% If more than one optimal solution, used the ensemble embedding for another
% k-means clustering
if ens>1
    Y = kmeans(Z(:,:,directed), K,'MaxIter',opts.MaxIterK,'Replicates',1,'Start','plus');
    % Final embed
    for i=1:num
        [tmpZ,W{i}]=GraphEncoderEmbed(X{i},Y3,n,opts);
        Z(:,(i-1)*K+1:i*K,:)=tmpZ*opts.Weight(i);
    end
    GCS=calculateGCS(Z(:,:,directed),Y,n,K);
end

%% Embedding Function
function [Z,W]=GraphEncoderEmbed(X,Y,n,opts)
if nargin<4
    opts = struct('Normalize',true,'Directed',1);
end
prob=false;
directed=opts.Directed;

s=size(X,1);
if size(Y,2)>1
    K=size(Y,2);
    prob=true;
else
    K=max(Y);
end
nk=zeros(1,K);
% if opts.Sparse==false
    W=zeros(n,K);
    Z=zeros(n,K*directed);
% else
%     W=sparse(n,K);
%     Z=sparse(n,K*directed);
% end
% indS=zeros(n,k);
if prob==true
    nk=sum(Y);
    W=Y./repmat(nk,n,1);
else
    for i=1:K
        ind=(Y==i);
        nk(i)=sum(ind);
        W(ind,i)=1/nk(i);
        %         indS(:,i)=ind;
    end
end
% num=size(X,3);

% Edge List Version in O(s)
for i=1:s
    a=X(i,1);
    b=X(i,2);
    e=X(i,3);
    if prob==true
        for j=1:K
            Z(a,j)=Z(a,j)+W(b,j)*e;
%             if a~=b
                tmp=j+(directed>1)*K;
                Z(b,tmp)=Z(b,tmp)+W(a,j)*e;
%             end
        end
    else
        c=Y(a);
        d=Y(b);
        if d>0
            Z(a,d)=Z(a,d)+W(b,d)*e;
        end
        if c>0 %&& a~=b
            tmp=c+(directed>1)*K;
            Z(b,tmp)=Z(b,tmp)+W(a,c)*e;
        end
    end
end
if directed==3
    Z(:,2*K+1:3*K)=Z(:,K+1:2*K)+Z(:,1:K);
end

if opts.Normalize==true
    for i=1:directed
        Z(:,(i-1)*K+1:i*K) = normalize(Z(:,(i-1)*K+1:i*K),2,'norm');
    end
    Z(isnan(Z))=0;
end
if directed>1
    Z=reshape(Z,n,K,directed);
end
% [~,Z]=pca(Z);
% Z=sum(Z,2);
% W=W(:,1:min(opts.Dim,K));

% % Z=reshape(Z,n,size(Z,2)*num);
% B=zeros(k,k);
% for j=1:k
%     tmp=(indS(:,j)==1);
%     B(j,:)=mean(Z(tmp,:));
% end

%% Compute the GEE clustering score (the minimal rank index)
function tmpGCS=calculateGCS(Zt,Y3,n,K)
D=zeros(n,K);
for i=1:K
    D(1:n,i)=sum((Zt-repmat(mean(Zt(Y3==i,:),1),n,1)).^2,2);
end
[~,tmpIdx]=min(D,[],2);
tmpGCS=mean(tmpIdx~=Y3);
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