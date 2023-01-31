%%%%% MSFT Data

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
%%% GEE Embedding
load('anonymized_msft.mat')
tic
opts = struct('DiagA',false);
[Z]=GraphEncoder(G,label,opts);
time_GEE=toc;
tic
Z=reshape(Z,n,K,24);
% [X]=run_umap([Z(:,:,1),label],'label_column','end','contour_percent',0, 'verbose','none','randomize',false);
% plotUMAP(X,label);fs=15;
%%% Find common index
vnorm=zeros(size(Z,1),24);
for i=1:24
    vnorm(:,i)=vecnorm(Z(:,:,i),2,2);
end
vnorm=sum(vnorm,2);
ind=(vnorm==24);
Z=Z(ind,:,:); %use common indices, then re-normalize
for j=1:24
      Z(:,:,j) = normalize(Z(:,:,j),2,'norm');
end
time_Common=toc;
%%% Dynamic Vertex Embedding
tic
Z_d=zeros(n_ind,24);
for r=1:n_ind
    tmpDist=zeros(24,24);
    for i=1:24
        for j=i+1:24
            tmpDist(i,j)=norm(Z(r,:,i)-Z(r,:,j),'fro');
            tmpDist(j,i)=tmpDist(i,j);
        end
    end
    Z_d(r,:)=cmdscale(tmpDist,1);
end
Z_diff=max(Z_d,[],2)-min(Z_d,[],2);
% ind_out=find(Z_diff>1);
time_Dist=toc;
% Dist=Dist+Dist';
%%% Visualize
% hist(Z_diff);
r=1; fs=24;thres=0.5;
subplot(2,2,1)
hold on
plot(1:24,Z_d(2,:),'b-.','LineWidth',2);
plot(1:24,Z_d(3,:),'g-.','LineWidth',2);
plot(1:24,Z_d(4,:),'r-.','LineWidth',2);
plot(1:24,Z_d(5,:),'c-.','LineWidth',2);
plot(1:24,Z_d(6,:),'m-.','LineWidth',2);
hold off
ylim([-0.5,1]);
xlim([1,24]);
xlabel('Month')
legend('Vertex 1','Vertex 2','Vertex 3','Vertex 4','Vertex 5','Location','NorthWest')
title('Vertex Dynamics')
set(gca,'FontSize',fs);
axis('square');
% xticks([1,13,24])
% Gy=mdscale(Dist,2);
% plot(Gy(:,1),Gy(:,2));
%outlier
%%%%%%%More
Dist=zeros(24,24);
for i=1:24
    for j=i+1:24
        Dist(i,j)=norm(Z(:,:,i)-Z(:,:,j),'fro');
    end
end
% Dist=Dist+Dist';
% Gy=mdscale(Dist,1);
% plot(1:24,Gy(:,1));
% Gy=mdscale(Dist,2);
% plot(Gy(:,1),Gy(:,2));
%outlier
subplot(2,2,2)
i=12;j=13;
Z1=Z(:,:,i);
Z2=Z(:,:,j);
res=vecnorm(Z1-Z2,2,2);
[~,indOut]=sort(res,'descend');
hist(res);
mean(res>thres)
xlim([0,1.414]);
title("Vertex Shifts from 2019.12 to 2020.1");
set(gca,'FontSize',fs);
set(gca,'YTickLabel',[])
axis('square');
subplot(2,2,3)
i=12;j=18;
Z1=Z(:,:,i);
Z2=Z(:,:,j);
res=vecnorm(Z1-Z2,2,2);
[~,indOut]=sort(res,'descend');
hist(res);
mean(res>thres)
xlim([0,1.414]);
title("Vertex Shifts from 2019.12 to 2020.6");
set(gca,'FontSize',fs);
set(gca,'YTickLabel',[])
axis('square');
subplot(2,2,4)
i=12;j=24;
Z1=Z(:,:,i);
Z2=Z(:,:,j);
res=vecnorm(Z1-Z2,2,2);
[~,indOut]=sort(res,'descend');
hist(res);
mean(res>thres)
xlim([0,1.414]);
title("Vertex Shifts from 2019.12 to 2020.12");
set(gca,'FontSize',fs);
set(gca,'YTickLabel',[])
axis('square');
F.fname='MSFT1';
F.wh=[8 8]*2;
%     F.PaperPositionMode='auto'; 
print_fig(gcf,F)

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