%% Compute the Adjacency Encoder Embedding.
%% Running time is O(nK+s) where s is number of edges, n is number of vertices, and K is number of class.
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
    opts = struct('DiagA',true,'Correlation',true,'Laplacian',false,'Learner',2,'LearnIter',0,'MaxIter',20,'MaxIterK',2,'Replicates',1,'Attributes',0,'Directed',1);
end
if ~isfield(opts,'DiagA'); opts.DiagA=true; end
if ~isfield(opts,'Correlation'); opts.Correlation=true; end
if ~isfield(opts,'Laplacian'); opts.Laplacian=false; end
if ~isfield(opts,'Learner'); opts.Learner=2; end
if ~isfield(opts,'LearnIter'); opts.LearnIter=0; end
if ~isfield(opts,'MaxIter'); opts.MaxIter=20; end
if ~isfield(opts,'MaxIterK'); opts.MaxIterK=2; end
if ~isfield(opts,'Replicates'); opts.Replicates=1; end
if ~isfield(opts,'Attributes'); opts.Attributes=0; end
if ~isfield(opts,'Directed'); opts.Directed=1; end
opts.neuron=20;
opts.activation='poslin';
U=opts.Attributes;
% opts.Directed=3;
di=opts.Directed;
% if ~isfield(opts,'distance'); opts.distance='correlation'; end
% opts.DiagA=true;
% opts.Correlation=true;
% opts.Laplacian=false;

%% pre-precess input to s*3 then diagonal augment
if iscell(X)
    num=length(X);
else
    X={X};
    num=1;
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
    Z=zeros(n,di*K*num);
    W=cell(1,num);
    if opts.Learner==2
        % initialize NN
        netGNN = patternnet(max(opts.neuron,K),'trainscg','crossentropy'); % number of neurons, Scaled Conjugate Gradient, cross entropy
%         netGNN.layers{1}.transferFcn = opts.activation;
        netGNN.trainParam.showWindow = false;
        netGNN.trainParam.epochs=100;
        netGNN.divideParam.trainRatio = 0.9;
        netGNN.divideParam.valRatio   = 0.1;
        netGNN.divideParam.testRatio  = 0;
    end
    if opts.LearnIter<1
        for i=1:num
            [Z(:,(i-1)*K*di+1:i*K*di),W{i}]=GraphEncoderEmbed(X{i},Y,n,opts);
        end
        if attr==true
            Z=[Z,U];
        end
        if sum(indT)<n
            if opts.Learner==1
                mdl=fitcdiscr(Z(indT,:),YTrn,'DiscrimType','pseudoLinear');
                Y(~indT)=predict(mdl,Z(~indT,:));
            else
                mdl = train(netGNN,Z(indT,:)',YTrn2);
                prob=mdl(Z(~indT,:)');
                Y(~indT)=vec2ind(prob)';
                %             [~,Y(~indT)] = max(prob,[],1); % class-wise probability for tsting data
            end
        end
    else
        %
        %         indNew=indT;
        meanSS=0;Y1=Y;
        for rep=1:opts.Replicates
            tmp=randi([1,K],[sum(~indT),1]);
            Y1(~indT)=tmp;
            Y2=onehotencode(categorical(Y1),2);
%             Y3=double(Y1);
            for r=1:opts.LearnIter
                %             i
                for i=1:num
                    [Z(:,(i-1)*K*di+1:i*K*di),W{i}]=GraphEncoderEmbed(X{i},Y1,n,opts); % discrete label version
                    %[Z(:,(i-1)*K+1:i*K),W{i}]=GraphEncoderEmbed(X{i},Y2,n,opts); % probability version
                end
%                 [Z,W]=GraphEncoderEmbed(X,Y2,n,opts);
                if attr==true
                    Z=[Z,U];
                end
                if opts.Learner==1
                    mdl=fitcdiscr(Z(indT,:),YTrn,'DiscrimType','pseudoLinear');
                    [class,prob] = predict(mdl,Z(~indT,:));
                    prob1=max(prob,[],2);
                else
                    mdl = train(netGNN,Z(indT,:)',YTrn2);
                    prob=mdl(Z(~indT,:)');
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
            if minP>meanSS
                meanSS=minP;Y=Y1;
            end
        end
    end
else 
    %% otherwise do clustering
    indT=zeros(n,1);
    K=Y;
    if n/max(K)<30
        disp('Too many clusters at maximum. Result may bias towards large K. Please make sure n/Kmax >30.')
    end
    %% when a given cluster size is specified
    if length(K)==1
        [Z,Y,W,meanSS]=GraphEncoderCluster(X,K,n,num,attr,opts);
    else
        %% when a range of cluster size is specified
        if length(K)<n/2 && max(K)<max(n/2,10)
            minSS=100;Z=0;W=0;meanSS=zeros(length(K),1);
            for r=1:length(K)
                [Zt,Yt,Wt,tmp]=GraphEncoderCluster(X,K(r),n,num,attr,opts);
                meanSS(r)=tmp;
                if tmp<minSS
                    minSS=tmp;Y=Yt;Z=Zt;W=Wt;
                end
            end
        end
    end
end

%% Clustering Function
function [Z,Y,W,minSS]=GraphEncoderCluster(X,K,n,num,attr,opts)

if nargin<4
    opts = struct('Correlation',true,'MaxIter',50,'MaxIterK',5,'Replicates',3,'Directed',1);
end
di=opts.Directed;
minSS=100;
Z=zeros(n,K*num);
Wt=cell(1,num);
W=cell(1,num);              
for rep=1:opts.Replicates
    Y2=randi([1,K],[n,1]);
    for r=1:opts.MaxIter
        for i=1:num
            [Zt(:,(i-1)*K*di+1:i*K*di),Wt{i}]=GraphEncoderEmbed(X{i},Y2,n,opts);
        end
        if attr==true
            Zt=[Zt,U];
        end
        [Y3,~,tmp,D] = kmeans(Zt, K,'MaxIter',opts.MaxIterK,'Replicates',1,'Start','plus');
%         [Y3,~,tmp,D] = kmeans(Zt*WB, K,'MaxIter',opts.MaxIterK,'Replicates',1,'Start','plus');
        %gmfit = fitgmdist(Z,k, 'CovarianceType','diagonal');%'RegularizationValue',0.00001); % Fitted GMM
        %Y = cluster(gmfit,Z); % Cluster index
        if RandIndex(Y2,Y3)==1
            break;
        else
            Y2=Y3;
        end
    end
    tmpCount=accumarray(Y3,1);
    tmp=tmp./tmpCount./(sum(D.^0.5)'-tmp).*(n-tmpCount).*tmpCount/n;
    tmp=mean(tmp)+2*std(tmp);
    if tmp<minSS
        Z=Zt;W=Wt;minSS=tmp;Y=Y3;
    end
end

%% Embedding Function
function [Z,W]=GraphEncoderEmbed(X,Y,n,opts)
if nargin<4
    opts = struct('Correlation',true,'Directed',1);
end
prob=false;
di=opts.Directed;

s=size(X,1);
if size(Y,2)>1
    K=size(Y,2);
    prob=true;
else
    K=max(Y);
end
nk=zeros(1,K);
W=zeros(n,K);
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
Z=zeros(n,K*di);
for i=1:s
    a=X(i,1);
    b=X(i,2);
    e=X(i,3);
    if prob==true
        for j=1:K
            Z(a,j)=Z(a,j)+W(b,j)*e;
            if a~=b
               tmp=j+(di>1)*K;
               Z(b,tmp)=Z(b,tmp)+W(a,j)*e;
            end
        end
    else
        c=Y(a);
        d=Y(b);
        if d>0
            Z(a,d)=Z(a,d)+W(b,d)*e;
        end
        if c>0 && a~=b
            tmp=c+(di>1)*K;
            Z(b,tmp)=Z(b,tmp)+W(a,c)*e;
        end
    end
end
if di==3
    Z(:,2*K+1:3*K)=Z(:,K+1:2*K)+Z(:,1:K);
end
if opts.Correlation==true
    Z = normalize(Z,2,'norm');
    Z(isnan(Z))=0;
end

% % Z=reshape(Z,n,size(Z,2)*num);
% B=zeros(k,k);
% for j=1:k
%     tmp=(indS(:,j)==1);
%     B(j,:)=mean(Z(tmp,:));
% end

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