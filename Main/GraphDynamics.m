%% Compute the Temporal Encoder Embedding and Dynamic measure for time-series graph.
%% Running time is O(nKT+ s) where s is total number of edges, n is number of vertices, K is number of class, T is the time steps.
%% Reference: C. Shen et al., "Discovering Communication Pattern Shifts in Large-Scale Networks using Encoder Embedding and Vertex Dynamics", 2023.
%%
%% @param X: needs to be a 1*T cells, each cell contain a s_t*3 edge list.
%% @param Y: is a n*1 class label vector of K groups.
%%        In case of partial known labels, Y should be a n*1 vector with unknown labels set to <=0 and known labels being >0.
%%        When there is no known label, set Y to be the number of desired clusters (which will use iterations and slower). 
%% @param opts specifies options:
%%        Common = true means vertices without any connectivity at any timestep will be excluded. By default false, so all verticecs are present.
%%        BenchTime is the reference time point to compute the dynamic measure. By default 1.
%%
%% @return Z: The n*k*T Encoder Embedding Z; the n*k Encoder Transformation: W; the n*1 label vector: Y;
%% @return Dynamic: 1*3 cell of Dynamic measures. Dynamic{1} is the n*T vertex dynamic, Dynamic{2} is the K*T community dynamic, 
%%         and Dynamic{3} is the 1*T graph dynamic.
%% @return Y: The 1*n label vector.
%% @return time: length 3 vector. First element is the embedding running time, second element is the common extraction running time (by default 0), 
%%         third element is the dynamic computation running time.
%%
%%
function [Z,Dynamic,Y,time]=GraphDynamics(X,Y,opts)

if nargin<3
    opts = struct('Common',false,'BenchTime',1,'Normalize',true);
end
if ~isfield(opts,'Common'); opts.Common=false; end
if ~isfield(opts,'BenchTime'); opts.BenchTime=1; end
if ~isfield(opts,'Normalize'); opts.Normalize=true; end

%%% Dynamic Encoder Embedding and Reshape
time=zeros(3,1);
tic
[Z,out]=GraphEncoder(X,Y,opts);
Y=out(opts.BenchTime).Y;
t=length(X);
if iscell(Z)
    Z=cell2mat(Z');
end
[n,Kt]=size(Z);
K=Kt/t;
Z=reshape(Z,n,K,t);
time(1)=toc;
t_b=opts.BenchTime;

if t>1
    %%% Extract common vertices only
    if opts.Common==true
        tic
        vnorm=zeros(size(Z,1),t);
        for i=1:t
            vnorm(:,i)=(vecnorm(Z(:,:,i),2,2)==1);
        end
        vnorm=sum(vnorm,2);
        ind=(vnorm==t);
        Z=Z(ind,:,:); %use common indices, then re-normalize
        for j=1:t
            Z(:,:,j) = normalize(Z(:,:,j),2,'norm');
        end
        time(2)=toc;
        n=size(Z,1);
        Y=Y(ind);
    end

    %%% Dynamic Vertex Embedding
    tic
    VD=zeros(n,t-t_b+1);CD=zeros(K,t-t_b+1);
    for i=2:t-t_b+1
%         VD(:,t_b+i-1)=vecnorm(Z(:,:,t_b+i-1)-Z(:,:,t_b),2,2);
        VD(:,t_b+i-1)=1-dot(Z(:,:,t_b+i-1),Z(:,:,t_b),2);
        for k=1:K
           CD(k,t_b+i-1)=mean(VD(Y==k,t_b+i-1));
        end
    end
    GD=mean(VD);
    Dynamic={VD,CD,GD};
    time(3)=toc;
end
%     for r=1:n
% %         tmpDist=zeros(t,t);
% %         tmpDist=zeros(t,1);
%         for i=1:t
%             Z_d(r,i)=norm(Z(r,:,i)-Z(r,:,t_b),'fro');
%         end
%         for i=1:t
%             for j=i+1:t
%                 tmpDist(i,j)=norm(Z(r,:,i)-Z(r,:,j),'fro');
%                 tmpDist(j,i)=tmpDist(i,j);
%             end
%         end
%         Z_d(r,:)=cmdscale(tmpDist,opts.d);
%           [U,S,~]=svds(tmpDist,1);
%           Z_d(r,:)=(U(:,1)*S^0.5)';
%     end
%     % Z_diff=max(Z_d,[],2)-min(Z_d,[],2);
%     % ind_out=find(Z_diff>1);
%     time(3)=toc;
