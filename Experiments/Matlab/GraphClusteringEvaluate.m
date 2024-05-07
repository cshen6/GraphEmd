function [result]=GraphClusteringEvaluate(X,Y,opts)

if nargin < 3
    opts = struct('Adjacency',1,'Laplacian',1,'Spectral',0,'NN',0,'Dist','sqeuclidean','normalize',0,'dmax',30,'dimGEE',0); % default parameters
end
if ~isfield(opts,'Adjacency'); opts.Adjacency=1; end
if ~isfield(opts,'Laplacian'); opts.Laplacian=1; end
if ~isfield(opts,'Spectral'); opts.Spectral=0; end
if ~isfield(opts,'NN'); opts.NN=0; end
if ~isfield(opts,'Dist'); opts.Dist='sqeuclidean'; end
if ~isfield(opts,'dimGEE'); opts.dimGEE=0; end
% if ~isfield(opts,'deg'); opts.deg=0; end
% if ~isfield(opts,'maxIter'); opts.maxIter=20; end
if ~isfield(opts,'normalize'); opts.normalize=0; end
if ~isfield(opts,'dmax'); opts.dmax=30; end
d=opts.dmax;
n=length(Y);
[~,~,Y]=unique(Y);
% n=length(Y);
K=max(Y);
opts.Dim = min(opts.dimGEE,K);

ARI_AEE=0;t_AEE=0;ARI_LEE=0;t_LEE=0;ARI_AEE_GNN=0;t_AEE_GNN=0;ARI_ASE=0;t_ASE=0;ARI_LSE=0;t_LSE=0;
minSSL=0;minSS=0;
% tic
% opts2=opts;opts2.deg=0;
% [ind_AEE,~]=GraphClustering(X,K,opts2);
% t_AEE=toc;
% ARI_AEE=RandIndex(Y,ind_AEE);
if iscell(X)
    opts.Spectral=0;
end
% if opts.Spectral==1
    if size(X,2)<=3
        X=X-min(min(X))+1;
        n=size(Y,1);
        Adj=zeros(n,n);
        for i=1:n
            Adj(X(i,1),X(i,2))=1;
            Adj(X(i,2),X(i,1))=1;
        end
    else
        Adj=X;
        X=adj2edge(Adj);
    end
% end

if opts.Adjacency==1
    tic
    oot=opts;
    oot.Laplacian=false;
    [~,ind_AEE]=UnsupGEE(X,K,n,oot);
    t_AEE=toc;
    ARI_AEE=RandIndex(Y,ind_AEE);
    
    if opts.Spectral==1
        ARI_ASE=zeros(d,1);t_ASE=zeros(d,1);
        tic
        [U,S,~]=svds(Adj,d);
        t1=toc;
        for j=1:d
            tic
            Z_ASE=U(:,1:j)*S(1:j,1:j)^0.5;
            ind_ASE = kmeans(Z_ASE, K,'Distance',opts.Dist);
            t_ASE(j)=t1+toc;
            ARI_ASE(j)=RandIndex(Y,ind_ASE);
        end
        [ARI_ASE,ind]=max(ARI_ASE);
        t_ASE=t_ASE(ind);
    end
end

if opts.NN==1
    [ind_AEE]=GraphClusteringNN(X,K);
    t_AEE_GNN=toc;
    ARI_AEE_GNN=RandIndex(Y,ind_AEE);
end

if opts.Laplacian==1
    tic
    [~,ind_LEE]=UnsupGEE(X,K,n,opts);
    t_LEE=toc;
    ARI_LEE=RandIndex(Y,ind_LEE);
    
    if opts.Spectral==1
        ARI_LSE=zeros(d,1);t_LSE=zeros(d,1);
        tic
        D=max(sum(Adj,1),1).^(0.5);
        AdjT=Adj;
        for j=1:n
            AdjT(:,j)=AdjT(:,j)/D(j)./D';
        end
        [U,S,~]=svds(AdjT,d);
        t1=toc;
        for j=1:d
            tic
            Z_LSE=U(:,1:j)*S(1:j,1:j)^0.5;
            ind_LSE = kmeans(Z_LSE, K,'Distance',opts.Dist);
            t_LSE(j)=t1+toc;
            ARI_LSE(j)=RandIndex(Y,ind_LSE);
        end
        [ARI_LSE,ind]=max(ARI_LSE);
        t_LSE=t_LSE(ind);
    end
end

accN=[ARI_AEE,ARI_ASE,ARI_AEE_GNN,ARI_LEE,ARI_LSE];
MDRI=[minSS,0,0,minSSL,0];
time=[t_AEE,t_ASE,t_AEE_GNN,t_LEE,t_LSE];

result = array2table([accN; time;MDRI], 'RowNames', {'ARI', 'time','MDRI'},'VariableNames', {'AEE','ASE','AEE_GNN','LEE','LSE'});
% result = array2table([accN; time], 'RowNames', {'ARI', 'time'},'VariableNames', {'AEE', 'AEE_Deg','AEE_NN', 'ASE','LSE'});

%% Adj to Edge Function
function [Edge,s,n]=adj2edge(Adj)
if size(Adj,2)<=3
    Edge=Adj;
    return;
end
n=size(Adj,1);
ind=ones(n,1);
Edge=zeros(sum(sum(Adj>0)),3);
s=1;
for i=1:n
%     if sum(Adj(i,:))+sum(Adj(:,i))==0
%         ind(i)=0;
%     end
    for j=1:n
        if Adj(i,j)>0
            Edge(s,1)=i;
            Edge(s,2)=j;
            Edge(s,3)=Adj(i,j);
            s=s+1;
        end
    end
end
Edge(s,1)=n;Edge(s,2)=n;Edge(s,3)=1;
% s=s-1;