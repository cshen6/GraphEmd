function [Z,YW,err]=GraphEncoderR(Edge,Y,indT,U,opts)

if nargin<3
    indT=ones(size(Y));
    indTrn=1;
end
if nargin<4
    U=0;
end
if nargin<5
    opts = struct('Normalize',true,'DiagAugment',false,'Laplacian',false,'Refine',0,'Directed',0,'MaxIter',30,'MaxIterK',3,'Replicates',3,'Elbow',0,'dim',30);
end
if ~isfield(opts,'Normalize'); opts.Normalize=true; end
if ~isfield(opts,'DiagAugment'); opts.DiagAugment=true; end
if ~isfield(opts,'Laplacian'); opts.Laplacian=false; end
if ~isfield(opts,'Refine'); opts.Refine=0; end
if ~isfield(opts,'MaxIter'); opts.MaxIter=30; end
if ~isfield(opts,'MaxIterK'); opts.MaxIterK=3; end
if ~isfield(opts,'Replicates'); opts.Replicates=3; end
if ~isfield(opts,'Directed'); opts.Directed=0; end
if ~isfield(opts,'Elbow'); opts.Elbow=0; end
if ~isfield(opts,'dim'); opts.Dim=30; end

if length(indT)~= length(Y)
    if indT>0 && indT<=1
        indTrn=indT;
        indT=(rand(size(Y))<=indT);
    else
        return;
    end
else
    indTrn=mean(indT);
end
mean(indT)
YW=zeros(size(Y));
YW(indT)=kmeans(Y(indT),opts.dim);
Z=GraphEncoder(Edge,YW,0,opts);
if size(U,1)==size(Z,1)
    Z=[Z,U];
end
err=0;
% Z=sum(Z,2);
if indTrn<1
    mdl=fitrensemble(Z(indT,:),Y(indT));
    %         mdl=fitrnet(ZTrn,YTrn);
    tt=predict(mdl,Z(~indT,:));
    YTsn=Y(~indT);
    err =sum((YTsn-tt).^2)/sum((YTsn-mean(YTsn)).^2);
end
%split, output testing data and SSE, can take a model to predict