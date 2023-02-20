%%%%% MSFT Data Pre-processing

%%% Input: all edges, using table
a=a(2:end,:);
G={};
b=unique(a(:,1));
for i=1:size(b,1)
ind = find(strcmp(string(a{:,1}), string(b{i,1})) == 1);
tmp=a{ind,2:4};tmp(:,1:2)=tmp(:,1:2)+1;
G{i}=tmp;
end
%%% process labels, using matrix
a=a(2:end,:);
[b,cind,~]=unique(a(:,2));
n=length(cind);
label=a(cind,2);
%%% Embedding & re-org
[Z,Dynamic,Y,time]=GraphDynamics(G,label);
bet=1;
tic
ind = (vecnorm(Z(:,:,bet),2,2)==1);
Y2=kmeans(Z(ind,:,bet), K,'Distance','correlation');
toc
RandIndex(label(ind)+1,Y2+1)
%%%Figure 0: Youngser's figure
% Dist=zeros(24,24);
% for i=1:24
%     for j=i+1:24
%         Dist(i,j)=norm(Z(:,:,i)-Z(:,:,j),'fro');
%     end
% end
% Dist=Dist+Dist';
% Gy=mdscale(Dist,1);
% plot(1:24,Gy(:,1));
% Gy=mdscale(Dist,2);
% plot(Gy(:,1),Gy(:,2));


%%% Phone data
phone=unique([a{:,1};a{:,2}]);
num=length(phone);
G=zeros(size(a));
G(:,3)=table2array(a(:,3));
for i=1:num
    ind=find(strcmp(string(a{:,1}), string(phone{i,1})) == 1);
    G(ind,1)=i;
    ind=find(strcmp(string(a{:,2}), string(phone{i,1})) == 1);
    G(ind,2)=i;
end

% GEE & UMAP
load('smartphone.mat')
[Z]=GraphEncoder(G,label);
[X]=run_umap([Z,label],'label_column','end','contour_percent',0, 'verbose','none','randomize',false);
plotUMAP(X,label);fs=15;

% directed outlier
directed=3;opts = struct('Directed',directed,'Normalize',true);
tic
[ZD]=GraphEncoder(G,label,opts);
timeD=toc;
res=zeros(n,3);
for dd=1:3
ZN=reshape(ZD(:,:,dd),n,K,24);
i=1;j=2;
Z1=ZN(:,:,i);
Z2=ZN(:,:,j);
res(:,dd)=vecnorm(Z1-Z2,2,2);
end