%% Compute the Adjacency Encoder Embedding.
%% Running time is O(s) where s is number of edges.
%% Reference: C. Shen and Q. Wang and C. E. Priebe, "Graph Encoder Embedding", 2021. 
%%
%% @param X is either n*n adjacency, or s*3 edge list.
%%        Adjacency matrix can be weighted or unweighted, directed or undirected. Complexity in O(n^2).
%%        Edgelist input can be either s*2 or s*3, and complexity in O(s).
%% @param Y is either an n*1 class label vector, or a positive integer for number of classes. 
%%        Y should be a n*1 vector when some labels are known. Unknown labels shall be set to <=0 and known labels being >0. 
%%        When there is no known label, set Y to be the number of classes. 
%% @param opts specifies two options: Laplacian being 1 uses graph Laplacian, otherwise uses adjacency matrix; 
%%        then maxIter denotes the max iteration when there is no known label, and not used otherwise.
%%
%% @return The n*k Encoder Embedding Z
%% @return The n*k Encoder Transformation W
%%
%% @export
%%

function [Z,Y,W,indT,B]=GraphEncoder(X,Y,opts)

if nargin<3
    opts = struct('Laplacian',false,'MaxIter',100,'Learn',false,'DiagA',true,'Correlation',true);
end
if ~isfield(opts,'Laplacian'); opts.Laplacian=false; end
if ~isfield(opts,'Learn'); opts.Learn=false; end
if ~isfield(opts,'MaxIter'); opts.MaxIter=100; end
if ~isfield(opts,'DiagA'); opts.DiagA=true; end
if ~isfield(opts,'Correlation'); opts.Correlation=true; end
% if ~isfield(opts,'distance'); opts.distance='correlation'; end
% opts.DiagA=false;
% opts.Laplacian=true;
% opts.Correlation=false;

[s,t]=size(X);
if t<=3
    n=max(max(X));
    if t==2
        X=[X,ones(s,1)];
%         t=3;
    end
    if opts.DiagA==true
        XNew=[1:n;1:n;ones(1,n)]';
        X=[X;XNew];
%         s=size(X,1);
    end
else
    n=size(X,1);
    if opts.DiagA==true
        X=X+1*eye(n,n);
    end
end

if length(Y)==1
    k=Y;
    indT=0; 
    Y2=randi([1,k],[n,1]);   
    warning ('off','all');
    for r=1:opts.MaxIter
        [Z,~,~,W,B]=GraphEncoderMain(X,Y2,opts);
%         try
            Y = kmeans(Z, k,'MaxIter',10,'Replicates',1,'Start','plus');
%             Y = kmeans(Z, k,'MaxIter',3,'Replicates',1,'Start','plus','Distance',opts.distance);
%             gmfit = fitgmdist(Z,k, 'CovarianceType','diagonal');%'RegularizationValue',0.00001); % Fitted GMM
%             Y = cluster(gmfit,Z); % Cluster index
            if RandIndex(Y2,Y)==1
                break;
            else
                Y2=Y;
            end
%         catch
%             r=1;
%             Y2=randi([1,k],[n,1]); %%% re-initialize
%         end
    end
else
    [Z,Y,W,indT,B]=GraphEncoderMain(X,Y,opts);
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
end


function [Z,Y,W,indT,B]=GraphEncoderMain(X,Y,opts)
if nargin<3
    opts = struct('Laplacian',false,'Correlation',true);
end
if ~isfield(opts,'Laplacian'); opts.Laplacian=false; end
if ~isfield(opts,'Correlation'); opts.Correlation=true; end

n=length(Y);
indT=(Y>0);
Y1=Y(indT);
[s,t]=size(X);
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
num=size(X,3);
        
if opts.Laplacian==true
    if t==3
        D=zeros(n,1);
        for i=1:s
            a=X(i,1);
            b=X(i,2);
            D(a)=D(a)+X(i,3);
            if a~=b
                D(b)=D(b)+X(i,3);
            end
        end
        D=D.^-0.5;
        for i=1:s
            X(i,3)=X(i,3)*D(X(i,1))*D(X(i,2));
        end
    else
        D=max(sum(X,1),1).^(0.5);
        for i=1:n
            X(:,i)=X(:,i)/D(i)./D';
        end
    end
end

% Adjacency matrix version in O(n^2)
if s==n && t==n   
    Z=zeros(n,k,num);
    for r=1:num
        Z(:,:,r)=X(:,:,r)*W;
    end
end

% Edge List Version in O(s) (thus more efficient for large sparse graph)
if t==3 && num==1
    Z=zeros(n,k);
    for i=1:s
        a=X(i,1);
        b=X(i,2);
        c=Y(a);
        d=Y(b);
        e=X(i,3);
        Z(a,d)=Z(a,d)+W(b,d)*e;
        if a~=b
            Z(b,c)=Z(b,c)+W(a,c)*e;
        end
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