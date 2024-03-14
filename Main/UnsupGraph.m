%% Compute the unsupervised Graph Encoder Embedding.

function [Z,YNew]=UnsupGraph(G,Y,n,opts)
warning ('off','all');
if nargin<4
    opts = struct('MaxIter',30,'MaxIterKMeans',3,'Replicates',3,'Normalize',true,'Refine',0,'Principal',0,'Laplacian',false,'Discriminant',false,'Softmax',false);
end
if ~isfield(opts,'MaxIter'); opts.MaxIter=30; end
if ~isfield(opts,'MaxIterKMeans'); opts.MaxIterKMeans=3; end
if ~isfield(opts,'Replicates'); opts.Replicates=3; end
if ~isfield(opts,'Normalize'); opts.Normalize=true; end
if ~isfield(opts,'Laplacian'); opts.Laplacian=false; end
%
if ~isfield(opts,'Refine'); opts.Refine=0; end
if ~isfield(opts,'Principal'); opts.Principal=0; end
if ~isfield(opts,'Discriminant'); opts.Discriminant=false; end
if ~isfield(opts,'Softmax'); opts.Softmax=false; end

numY=length(Y);
Score=1;
Z=0;YNew=0;
for i=1:numY
    K=Y(i);
    for rep=1:opts.Replicates
        tmpY=randi(K,n,1);
        tmpZ=GraphEncoder(G,tmpY,opts);
        for r=1:opts.MaxIter
            tmpY1 = kmeans(tmpZ, K,'MaxIter',opts.MaxIterKMeans,'Replicates',1,'Start','plus');
            %[Y3] = kmeans(Zt*WB, K,'MaxIter',opts.MaxIterK,'Replicates',1,'Start','plus');
            %gmfit = fitgmdist(Z,k, 'CovarianceType','diagonal');%'RegularizationValue',0.00001); % Fitted GMM
            %Y3 = cluster(gmfit,Z); % Cluster index
            if RandIndex(tmpY,tmpY1)==1
                break;
            else
                tmpY=tmpY1;
            end
            tmpZ=GraphEncoder(G,tmpY,opts);
        end
        % Compute clustering score for each replicate
        tmpScore=ClusteringScore(tmpZ,tmpY,n,K);
%         if tmpScore==Score
%             Z=Z+Zt;
%             ens=ens+1;
%         end
        if tmpScore<Score
            Z=tmpZ;Score=tmpScore;YNew=tmpY;
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