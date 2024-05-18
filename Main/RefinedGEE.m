%% A refined graph encoder embedding

function [Z,output]=RefinedGEE(G,Y,opts)
warning ('off','all');
if nargin<3
    opts = struct('Normalize',true,'RefineK',5,'RefineY',5,'eps',0.3,'epsn',5);
end
if ~isfield(opts,'Normalize'); opts.Normalize=true; end
if ~isfield(opts,'RefineK'); opts.RefineK=5; end
if ~isfield(opts,'RefineY'); opts.RefineY=5; end
if ~isfield(opts,'eps'); opts.eps=0.3; end
if ~isfield(opts,'epsn'); opts.epsn=5; end
opts.Discriminant = true;
opts.Principal=0;
version=2;

[Z,output]=GraphEncoder(G,Y,opts);

% Refined on K
if version==2 % new version 
    if opts.RefineK>0
        ZK=cell(opts.RefineK,1);
        output1=output;idx=output1.idx;
        for rK=1:opts.RefineK
            indK=output1.indK;K=size(indK,2);
            tmp = sum(repmat(idx,1,K) & indK); %class-wise index that is mis-classified
            Y1=output1.YVal+idx*sum(tmp>0); %new class
            [~,output1]=GraphEncoder(G,Y1,opts);
            tmp2=sum(repmat(output1.idx,1,K) & indK); %check wrong index per-class
            idxK=(tmp>=tmp2+opts.epsn);
            if sum(idx)-sum(output1.idx & idx)< max(sum(idx)*opts.eps,opts.epsn)
                break;
            else
                Y1=output1.YVal+idx*sum(idxK);
                [Z1,output1]=GraphEncoder(G,Y1,opts);%idx=output1.idx;K=size(Z1,2);
                ZK{rK,1}=Z1;idx=output1.idx & idx;
            end
        end
        ZK=horzcat(ZK{:});
        Z=[Z,ZK];
    end
end
if version==1 % old version for paper, simpler and has similar vertex error in most cases
    K=size(Z,2);
    % Refined Graph Encoder Embedding
    if opts.RefineK>0
        ZK=cell(opts.RefineK,1);
        output1=output;idx=output1.idx;
        for rK=1:opts.RefineK
            Y1=output1.YVal+idx*K;
            [Z2,output2]=GraphEncoder(G,Y1,opts);
            if sum(idx)-sum(output2.idx & idx)< max(sum(idx)*opts.eps,opts.epsn)
                break;
            else
                ZK{rK,1}=Z2;output1=output2;idx=output2.idx & idx;
            end
        end
        ZK=horzcat(ZK{:});
        Z=[Z,ZK];
    end
end

% Refined on Y
if opts.RefineY>0
    ZY=cell(opts.RefineY,1);
    output1=output;idx=output1.idx;
    for r=1:opts.RefineY
        [Z2,output2]=GraphEncoder(G,output1.YVal,opts);
        if sum(idx)-sum(output2.idx & idx)< max(sum(idx)*opts.eps,opts.epsn)
            break;
        else
            ZY{r}=Z2;output1=output2;idx=output2.idx & idx;
        end
    end
    ZY=horzcat(ZY{:});
    Z=[Z,ZY];
end