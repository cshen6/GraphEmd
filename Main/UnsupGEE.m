%% Compute the unsupervised Graph Encoder Embedding.

function [Z,YNew]=UnsupGEE(G,Y,n,opts)
warning ('off','all');
if nargin<4
    opts = struct('MaxIter',10,'Replicates',10,'Normalize',true,'Refine',0,'Metric',0,'Principal',0,'Laplacian',false,'Discriminant',false,'SeedY',0,'Transformer',1);
end
if ~isfield(opts,'MaxIter'); opts.MaxIter=10; end
if ~isfield(opts,'Replicates'); opts.Replicates=10; end
if ~isfield(opts,'Normalize'); opts.Normalize=true; end
if ~isfield(opts,'Laplacian'); opts.Laplacian=false; end
%
if ~isfield(opts,'Metric'); opts.Metric=0; end
if ~isfield(opts,'Refine'); opts.Refine=0; end
if ~isfield(opts,'Principal'); opts.Principal=0; end
if ~isfield(opts,'Discriminant'); opts.Discriminant=false; end
if ~isfield(opts,'SeedY'); opts.SeedY=0; end
if ~isfield(opts,'Transformer'); opts.Transformer=1; end

numY=length(Y);
Score=1;
Z=0;YNew=0;
if opts.Transformer
    K=Y(1);
    tmpY=randi(K,n,opts.Replicates);
    for j=1:opts.Replicates
        for r=1:opts.MaxIter-1
            tmpZ=GraphEncoder(G,tmpY(:,j),opts);
            % tmpY2=kmeans(tmpZ, K,'MaxIter',10,'Replicates',1,'Start','plus');
            % Z(:,(j-1)*K+1:j*K)=tmpZ;
            [~,tmpY2]=max(tmpZ,[],2);
            if RandIndex(tmpY2,tmpY(:,j))==1
                tmpY(:,j)=tmpY2;
                break;
            else
                tmpY(:,j)=tmpY2;
            end
        end
    end
    Z=zeros(n,K*opts.Replicates);
    for j=1:opts.Replicates
        tmpZ=GraphEncoder(G,tmpY(:,j),opts);
        Z(:,(j-1)*K+1:(j-1)*K+size(tmpZ,2))=tmpZ;
    end
    YNew=kmeans(Z,K);
    Z=GraphEncoder(G,YNew,opts);
else
    for i=1:numY
        K=Y(i);
        for rep=1:opts.Replicates
            if rep==1 && length(opts.SeedY)==n
                tmpY=opts.SeedY;
            else
                tmpY=randi(K,n,1);
            end
            [tmpZ,out]=GraphEncoder(G,tmpY,opts);
            % mu = normalize(out.mu,2,'norm');
            for r=1:opts.MaxIter
                % size(out.mu)
                % size(tmpZ)
                mu = normalize(out.mu,2,'norm');
                switch opts.Metric
                    case 0
                        tmpZmu=tmpZ*mu';
                    case 1
                        tmpZmu=pdist2(tmpZ,mu,'euclidean');
                    case 2
                        tmpZmu=1-pdist2(tmpZ,mu,'spearman');
                    case 3
                        tmpZmu=1-pdist2(tmpZ,mu,'cosine');
                end
                [~,tmpY1]=max(tmpZmu,[],2);
                if RandIndex(tmpY,tmpY1)==1
                    break;
                else
                    tmpY=tmpY1;
                end
                [tmpZ,out]=GraphEncoder(G,tmpY,opts);
            end
            tmpScore=ClusteringScore(tmpZ,tmpY,n,K);
            if tmpScore<Score
                Z=tmpZ;Score=tmpScore;YNew=tmpY;
            end
        end
    end
end


% % If more than one optimal solution, used the ensemble embedding for another
% % k-means clustering
% if ens>1
%     Y = kmeans(Z, K,'MaxIter',opts.MaxIterKMeans,'Replicates',1,'Start','plus');
%     Z = GraphEncoderEmbed(G,Y(:,1),n,nk,opts);
%     Score=ClusteringScore(Z,Y,n,K);
% end


%% Compute the GEE clustering score (the minimal rank index)
function score=ClusteringScore(Z,Y,n,K)
D=zeros(n,K);
for i=1:K
    D(1:n,i)=sum((Z-repmat(mean(Z(Y==i,:),1),n,1)).^2,2);
end
[~,tmpIdx]=min(D,[],2);
score=mean(tmpIdx~=Y);