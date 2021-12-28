%% Compute the Adjacency Encoder Embedding.
%% Running time is O(s) where s is number of edges.
%% Reference: C. Shen and Q. Wang and C. E. Priebe, "Graph Encoder Embedding", 2021.
%%            C. Shen et.al., "Graph Encoder Clustering", in preparation.
%%
%% @param X is either n*n adjacency, or s*3 edge list. Vertex size should be >10.
%%        Adjacency matrix can be weighted or unweighted, directed or undirected. It will be converted to s*3 edgelist.
%%        Edgelist input can be either s*2 or s*3, and complexity in O(s).
%% @param Y is either an n*1 class label vector, or a positive integer for number of clusters, or a range of potential cluster size, i.e., [2,10].
%%        In case of partial known labels, Y should be a n*1 vector with unknown labels set to <=0 and known labels being >0. 
%%        When there is no known label, set Y to be the number of clusters or a range of clusters.
%% @param opts specifies options: 
%%        DiagA = true means adding 1 to all diagonal entries (i.e., add self-loop to edgelist);
%%        Correlation specifies whether to use angle metric or Euclidean metric;
%%        Laplacian specifies whether to uses graph Laplacian or adjacency matrix; 
%%        Three integers for clustering: Replicates denotes the number of replicates for clustering, 
%%                                       MaxIter denotes the max iteration within each replicate for encoder embedding,
%%                                       MaxIterK denotes the max iteration used within kmeans.
%%
%% @return The n*k Encoder Embedding Z; the n*k Encoder Transformation: W; the n*1 label vector: Y;
%% @return The n*1 boolean vector for known label: indT (only for classification);
%% @return The meanSS criterion, the smaller the better (only for clustering);
%%
%% @export
%%

function [Z,Y,W,indT,meanSS]=GraphEncoder(X,Y,opts)
warning ('off','all');
if nargin<2
    Y=2:5;
end
if nargin<3
    opts = struct('DiagA',true,'Correlation',true,'Laplacian',false,'Learn',false,'MaxIter',100,'MaxIterK',10,'Replicates',1);
end
if ~isfield(opts,'DiagA'); opts.DiagA=true; end
if ~isfield(opts,'Correlation'); opts.Correlation=true; end
if ~isfield(opts,'Laplacian'); opts.Laplacian=false; end
if ~isfield(opts,'Learn'); opts.Learn=false; end
if ~isfield(opts,'MaxIter'); opts.MaxIter=50; end
if ~isfield(opts,'MaxIterK'); opts.MaxIterK=10; end
if ~isfield(opts,'Replicates'); opts.Replicates=3; end
% if ~isfield(opts,'distance'); opts.distance='correlation'; end
% opts.DiagA=false;
% opts.Laplacian=true;
% opts.Correlation=false;

%% pre-precess input to s*3 then diagonal augment
[s,t]=size(X);
if s==t % convert adjacency matrix to edgelist
    [X]=adj2edge(X);
end
if t==2 % enlarge the edgelist to s*3
    X=[X,ones(s,1)];
end
n=max(max(X(:,1:2)));
if opts.DiagA==true
    XNew=[1:n;1:n;ones(1,n)]';
    X=[X;XNew];
end

%% partial or full known labels when label size matches vertex size, do embedding / classification directly
if length(Y)==n 
    [Z,Y,W,indT]=GraphEncoderEmbed(X,Y,n,opts);
%     mdl=fitcdiscr(Z(indT,:),Y(indT));
%     Y(~indT)=predict(mdl,Z(~indT,:));        
    Mdl = fitsemiself(Z(indT,:),Y(indT),Z(~indT,:),'Learner','discriminant','IterationLimit',10,'ScoreThreshold',0.1);
    Y(~indT)=Mdl.FittedLabels;
    meanSS=0;
else 
    %% otherwise do clustering
    indT=zeros(n,1);
    K=Y;
    if n/max(K)<30
        disp('Too many clusters at maximum. Result may bias towards large K. Please make sure n/Kmax >30.')
    end
    %% when a given cluster size is specified
    if length(K)==1
        [Z,Y,W,meanSS]=GraphEncoderCluster(X,K,n,opts);
    else
        %% when a range of cluster size is specified
        if length(K)<n/2 && max(K)<max(n/2,10)
            minSS=-1;Z=0;W=0;meanSS=zeros(length(K),1);
            for r=1:length(K)
                [Zt,Yt,Wt,tmp]=GraphEncoderCluster(X,K(r),n,opts);
                meanSS(r)=tmp;
%                 a_counts = accumarray(Yt,1);
                if minSS==-1 || tmp<minSS
                    minSS=tmp;Y=Yt;Z=Zt;W=Wt;
                end
            end
        end
    end
end
%     if opts.Learn==true && sum(indT)<length(Y)
%         t1=indT;
%         t2=~indT;
%         thres=0.5;
%         
%         opttt=2;
%         if opttt==1
%             mdl=fitsemiself(Z(t1,:),Y(t1),Z(t2,:),'Learner','discriminant','IterationLimit',opts.maxIter,'ScoreThreshold',thres);
%             tmp=(max(abs(mdl.LabelScores),[],2)>thres);
%             tmp1=find(t2>0);
%             Y(tmp1(tmp))=mdl.FittedLabels(tmp);
%             [Z,Y,W,indT,B]=GraphEncoderMain(X,Y,opts);
%         else
%             for i=1:opts.maxIter
%                 mdl=fitsemiself(Z(t1,:),Y(t1),Z(t2,:),'Learner','discriminant','IterationLimit',1,'ScoreThreshold',thres);
%                 tmp=(max(abs(mdl.LabelScores),[],2)>thres);
%                 if (sum(tmp)==0)
%                     break;
%                 else
%                     tmp1=find(t2>0);
%                     Y(tmp1(tmp))=mdl.FittedLabels(tmp);
%                     t1=find(Y>=0);
%                     t2=~t1;
%                     [Z,Y,W,indT,B]=GraphEncoderMain(X,Y,opts);
%                 end
%             end
%         end
%     end

%% Clustering Function
function [Z,Y,W,minSS]=GraphEncoderCluster(X,K,n,opts)

if nargin<4
    opts = struct('Correlation',true,'Laplacian',false,'MaxIter',50,'MaxIterK',5,'Replicates',3);
end
minSS=-1;
for rep=1:opts.Replicates
    Y2=randi([1,K],[n,1]);
    for r=1:opts.MaxIter
        [Zt,~,~,Wt]=GraphEncoderEmbed(X,Y2,n,opts);
        [Y3,~,tmp,D] = kmeans(Zt, K,'MaxIter',opts.MaxIterK,'Replicates',1,'Start','plus');
        %gmfit = fitgmdist(Z,k, 'CovarianceType','diagonal');%'RegularizationValue',0.00001); % Fitted GMM
        %Y = cluster(gmfit,Z); % Cluster index
        if RandIndex(Y2,Y3)==1
            break;
        else
            Y2=Y3;
        end
    end
%     tmp
    tmpCount=accumarray(Y3,1);
%         tmp./tmpCount
%     tmp./tmpCount./sum(D)'*n
    tmp=tmp./tmpCount./(sum(D)'-tmp).*(n-tmpCount).*tmpCount/n;
    tmp=mean(tmp)+2*std(tmp);
    %tmp=max(tmp./tmpCount./sum(D)'*n);
%     tmp=median(tmp./tmpCount./sum(D)'*n)
%         tmp=max(tmp)/sum(sum(D))*K*n;
%     sum(D)/n
%     tmp=max(tmp./sum(D)');
%     tmp=max(tmp)/min(sum(D));
%     tmp=max(tmp)/sum(sum(D))*K*n;
%     tmp=sum(sum(D));
%     tmp=tmp/sum(sum(D));
%     tmp=sum(tmp)/sum(sum(D));
    if minSS==-1 || tmp<minSS
        Z=Zt;W=Wt;minSS=tmp;Y=Y3;
    end
end

%% Embedding Function
function [Z,Y,W,indT,B]=GraphEncoderEmbed(X,Y,n,opts)
if nargin<4
    opts = struct('Correlation',true,'Laplacian',false);
end

indT=(Y>0);
Y1=Y(indT);
s=size(X,1);
[tmp,~,Ytmp]=unique(Y1);
Y(indT)=Ytmp;
k=length(tmp);

nk=zeros(1,k);
W=zeros(n,k);
indS=zeros(n,k);
for i=1:k
    ind=(Y==i);
    nk(i)=sum(ind);
    W(ind,i)=1/nk(i);
    indS(:,i)=ind;
end
% num=size(X,3);
        
if opts.Laplacian==true
    D=zeros(n,1);
    for i=1:s
        a=X(i,1);
        b=X(i,2);
        c=X(i,3);
        D(a)=D(a)+c;
        if a~=b
            D(b)=D(b)+c;
        end
    end
    D=D.^-0.5;
    for i=1:s
        X(i,3)=X(i,3)*D(X(i,1))*D(X(i,2));
    end
end

% Edge List Version in O(s)
Z=zeros(n,k);
for i=1:s
    a=X(i,1);
    b=X(i,2);
    c=Y(a);
    d=Y(b);
    e=X(i,3);
    if d>0
        Z(a,d)=Z(a,d)+W(b,d)*e;
    end
    if c>0 && a~=b
        Z(b,c)=Z(b,c)+W(a,c)*e;
    end
end

if opts.Correlation==true
    Z = normalize(Z,2,'norm');
    Z(isnan(Z))=0;
end

% Z=reshape(Z,n,size(Z,2)*num);
B=zeros(k,k);
for j=1:k
    tmp=(indS(:,j)==1);
    B(j,:)=mean(Z(tmp,:));
end

%% Adj to Edge Function
function [Edge]=adj2edge(Adj)
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