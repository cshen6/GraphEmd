function simClustering(choice)

if choice==1
%%% self-learning
n=2000;K=10;
opts = struct('Adjacency',1,'Laplacian',1,'Spectral',1,'LDA',1,'GFN',1,'GCN',0,'GNN',0,'knn',5,'dim',30,'neuron',10,'epoch',100,'training',0.8,'activation','poslin'); % default parameters
[Adj,Y]=simGenerate(10,n,K);
indices = crossvalind('Kfold',Y,5);error1=0;
for i=1:max(indices);
trn = (indices == i); % tst indices
tsn = ~trn; % trning indices
Y2=Y;Y2(tsn)=-1;
opts = struct('Laplacian',false,'maxIter',20,'Learn',true);
[Z,Y3,W,indT,B]=GraphEncoder(Adj,Y2,opts);
error1=error1+mean(Y3(tsn)==Y(tsn))/max(indices);
end
end

if choice==2;
%%%%
n=2000;K=10;
opts = struct('Adjacency',1,'Laplacian',1,'Spectral',1,'LDA',1,'GFN',1,'GCN',0,'GNN',0,'knn',5,'dim',30,'neuron',10,'epoch',100,'training',0.8,'activation','poslin'); % default parameters
[Adj,Y]=simGenerate(10,n);
GraphEncoderC(Adj,3,Y);
[Adj,Y]=simGenerate(11,n,5);
GraphEncoderC(Adj,5,Y);
[Adj,Y]=simGenerate(20,n);
GraphEncoderC(Adj,3,Y);
[Adj,Y]=simGenerate(21,n,5);
GraphEncoderC(Adj,5,Y);
[Adj,Y]=simGenerate(30,n);
GraphEncoderC(Adj,3,Y);
[Adj,Y]=simGenerate(31,n,5);
GraphEncoderC(Adj,5,Y);
end
%%% Basic Sims

if choice==3
n=2000;K=10;
opts = struct('Adjacency',1,'Laplacian',1,'Spectral',1,'LDA',1,'GFN',1,'GCN',0,'GNN',0,'knn',5,'dim',30,'neuron',10,'epoch',100,'training',0.8,'activation','poslin'); % default parameters
[Adj,Y]=simGenerate(10,n);
indices = crossvalind('Kfold',Y,5);
opts.indices=indices; 
% opts2=opts;
% opts2.deg=1;opts2.ASE=0;opts2.LSE=0;opts2.GCN=0;opts2.GNN=0; opts2.LDA=0;opts2.GFN=0;
SBM=GraphEncoderSemi(Adj,Y,opts);
% SBM1=GraphEncoderEvaluate(Adj,Y,opts2);
[Adj,Y]=simGenerate(11,n,K);
SBM2=GraphEncoderSemi(Adj,Y,opts);
% SBM3=GraphEncoderEvaluate(Adj,Y,opts2);
[Adj,Y]=simGenerate(30,n);
RDPG=GraphEncoderSemi(Adj,Y,opts);
% RDPG1=GraphEncoderEvaluate(Adj,Y,opts2);
[Adj,Y]=simGenerate(31,n,K);
RDPG2=GraphEncoderSemi(Adj,Y,opts);
% RDPG3=GraphEncoderEvaluate(Adj,Y,opts2);
% DC-SBM
[Adj,Y]=simGenerate(20,n);
DCSBM=GraphEncoderSemi(Adj,Y,opts);
% DCSBM1=GraphEncoderEvaluate(Adj,Y,opts2);
[Adj,Y]=simGenerate(21,n,K);
DCSBM2=GraphEncoderSemi(Adj,Y,opts);
% DCSBM3=GraphEncoderEvaluate(Adj,Y,opts2);
% Dist
opts.LSE=0;opts2.LSE=0;
[Adj,Y]=simGenerate(40,n);
Dist=GraphEncoderSemi(Adj,Y,opts);
% Dist1=GraphEncoderEvaluate(Adj,Y,opts2);
[Adj,Y]=simGenerate(41,n,K);
Dist2=GraphEncoderSemi(Adj,Y,opts);
% Dist3=GraphEncoderEvaluate(Adj,Y,opts2);
% Kernel
[Adj,Y]=simGenerate(50,n);
Kern=GraphEncoderSemi(Adj,Y,opts);
% Kern1=GraphEncoderEvaluate(Adj,Y,opts2);
[Adj,Y]=simGenerate(51,n,K);
Kern2=GraphEncoderSemi(Adj,Y,opts);
end

if choice==4
%%% Basic Sims
n=500;
% opts = struct('ASE',1,'LSE',1,'NN',0,'Dist','cosine','maxIter',20,'normalize',0,'deg',0,'dmax',30); % default parameters
K=10;rep=10; tt=3;
SBM=zeros(3,7,tt);
RDPG=zeros(3,7,tt);
DCSBM=zeros(3,7,tt);
for r=1:rep
[Adj,Y]=simGenerate(10,n,K);
% % Edge=Adj2Edge(Adj);
SBM(:,:,1)=SBM(:,:,1)+table2array(GraphClusteringEvaluate(Adj,Y))/rep;
[Adj,Y]=simGenerate(11,n,K);
% Edge=Adj2Edge(Adj);
SBM(:,:,2)=SBM(:,:,2)+table2array(GraphClusteringEvaluate(Adj,Y))/rep;
[Adj,Y]=simGenerate(12,n,K);
% Edge=Adj2Edge(Adj);
SBM(:,:,3)=SBM(:,:,3)+table2array(GraphClusteringEvaluate(Adj,Y))/rep;
% [Adj,Y]=simGenerate(30,n,k);
% % % Edge=Adj2Edge(Adj);
% RDPG(:,:,1)=RDPG(:,:,1)+table2array(GraphClusteringEvaluate(Adj,Y))/rep;
% [Adj,Y]=simGenerate(31,n,k);
% % Edge=Adj2Edge(Adj);
% RDPG(:,:,2)=RDPG(:,:,2)+table2array(GraphClusteringEvaluate(Adj,Y))/rep;
% DC-SBM
[Adj,Y]=simGenerate(20,n);
% % Edge=Adj2Edge(Adj);
DCSBM(:,:,1)=DCSBM(:,:,1)+table2array(GraphClusteringEvaluate(Adj,Y))/rep;
[Adj,Y]=simGenerate(21,n,K);
% Edge=Adj2Edge(Adj);
DCSBM(:,:,2)=DCSBM(:,:,2)+table2array(GraphClusteringEvaluate(Adj,Y))/rep;
[Adj,Y]=simGenerate(22,n,K);
% Edge=Adj2Edge(Adj);
DCSBM(:,:,3)=DCSBM(:,:,3)+table2array(GraphClusteringEvaluate(Adj,Y))/rep;
end
end
% 
% n=2000;K=3;
% [Adj,Y]=simGenerate(10,n,K);
% simClusteringReal(Adj,Y);
% 
% % Dist
% opts.LSE=0;opts2.LSE=0;
% [Adj,Y]=simGenerate(40,n);
% Dist=GraphClusteringEvaluate(Adj,Y);
% [Adj,Y]=simGenerate(41,n,K);
% Dist2=GraphClusteringEvaluate(Adj,Y);
% % Kernel
% [Adj,Y]=simGenerate(50,n);
% Kern=GraphClusteringEvaluate(Adj,Y);
% [Adj,Y]=simGenerate(51,n,K);
% Kern2=GraphClusteringEvaluate(Adj,Y);

% 
% 
% 
% 
% n=3000;
% [Adj,Y]=simGenerate(1,n);
% Dist='sqeuclidean'; %Dist='cosine';
% [SBM_RI_AEE,SBM_RI_AEN,SBM_RI_ASE,SBM_t_AEE,SBM_t_AEN,SBM_t_ASE,SBM_ind_AEE,SBM_ind_AEN,SBM_ind_ASE]=simClusteringReal(Adj,Y,Dist);
% % Dist='cosine';
% % [SBM_RI_AEE2,~,SBM_t_AEE2,~,SBM_ind_AEE2,~]=simClusteringReal(Adj,Y,Dist);
% [Adj,Y]=simGenerate(2,n);
% Dist='sqeuclidean';%Dist='cosine';
% [DCSBM_RI_AEE,DCSBM_RI_AEN,DCSBM_RI_ASE,DCSBM_t_AEE,DCSBM_t_AEN,DCSBM_t_ASE,DCSBM_ind_AEE,DCSBM_ind_AEN,DCSBM_ind_ASE]=simClusteringReal(Adj,Y,Dist);
% % Dist='cosine';
% % [DCSBM_RI_AEE2,~,DCSBM_t_AEE2,~,DCSBM_ind_AEE2,~]=simClusteringReal(Adj,Y,Dist);
% [Adj,Y]=simGenerate(3,n);
% Dist='sqeuclidean';%Dist='cosine';
% [RDPG_RI_AEE,RDPG_RI_AEN,RDPG_RI_ASE,RDPG_t_AEE,RDPG_t_AEN,RDPG_t_ASE,RDPG_ind_AEE,RDPG_ind_AEN,RDPG_ind_ASE]=simClusteringReal(Adj,Y,Dist);
% Dist='cosine';
% [RDPG_RI_AEE2,~,RDPG_t_AEE2,~,RDPG_ind_AEE2,~]=simClusteringReal(Adj,Y,Dist);

%% Real Data
% [AN_RI_AEE,AN_RI_AEN,AN_RI_ASE,AN_t_AEE,AN_t_AEE2,AN_t_AEN,AN_t_ASE]=simClusteringReal(Adj,Label);
% 

%%% 1. Corr better
if choice==5
n=3000;K=10;
[Adj,Y]=simGenerate(10,n);opts.Dim=K;
GraphClusteringEvaluate(Adj,Y,opts)
[Adj,Y]=simGenerate(11,n,K);opts.Dim=K;
GraphClusteringEvaluate(Adj,Y,opts)
[Adj,Y]=simGenerate(12,n,K);opts.Dim=K;
GraphClusteringEvaluate(Adj,Y,opts)
% [Adj,Y]=simGenerate(15,n,k);
% GraphClusteringEvaluate(Adj,Y)
% [Adj,Y]=simGenerate(16,n,k);
% GraphClusteringEvaluate(Adj,Y)
[Adj,Y]=simGenerate(20,n);opts.Dim=K;
GraphClusteringEvaluate(Adj,Y,opts)
[Adj,Y]=simGenerate(21,n,K);opts.Dim=K;
GraphClusteringEvaluate(Adj,Y,opts)
[Adj,Y]=simGenerate(22,n,K);opts.Dim=K;
GraphClusteringEvaluate(Adj,Y,opts)
%
t=0;r=100;K=3;
for i=1:r
    [Adj,Y]=simGenerate(13,450,K);
    [~,~,~,~,score]=GraphEncoder(Adj,[2:10]);
    [~,ind]=min(score);
    if ind==K-1
        t=t+1/r;
    end
end
end

if choice==8
    n=300;K=2; rep=5;lim=5;
    [A,Y]=simGenerate(11,n);
    tmpY=randi(K,n,rep);
    Z=zeros(n,K*rep);
    for r=1:lim
        for i=1:rep
            Z(:,(i-1)*K+1:i*K)=GraphEncoder(A,tmpY(:,i));
            [~,tmpY(:,i)]=max(Z(:,(i-1)*K+1:i*K),[],2);
        end
    end
    Y2=kmeans(Z,K);
    % [~,Y2]=max(Z,[],2);
    % Y2=mod(Y2,K)+1;
    RandIndex(Y,Y2)

    [~,Y1]=UnsupGEE(A,K,n); 
    RandIndex(Y,Y1)
end

if choice==9
    n=1000;K=2; rep=5;
    [A,Y]=simGenerate(1,n);
    tmpY=randi(K,n,rep);
    Z=zeros(n,K*rep);
    for i=1:rep
        Z(:,(i-1)*K+1:i*K)=GraphEncoder(A,tmpY(:,i));
    end
    Y2=kmeans(Z,K);
    % [~,Y2]=max(Z,[],2);
    % Y2=mod(Y2,K)+1;
    RandIndex(Y,Y2)
end

if choice >=10 && choice<20
    d=20;n=3000;K=3; 
    switch choice
        case 10 
            type=10;%?
        case 11 
            type=12;%?
        case 12 
            type=15;
        case 13 
            type=11;%?
        case 14 
            type=20;
        case 15 
            type=22;
        case 16 
            type=25;
        case 17 
            type=27;
        case 18 
            type=21;%?
    end
    [A,Y]=simGenerate(type,n);
    opts1 = struct('MaxIter',10,'Replicates',10,'Normalize',true,'Discriminant',0);% default parameter
    opts2 = struct('MaxIter',10,'Replicates',10,'Normalize',true,'Discriminant',0,'Transformer',1);% default parameter
    tic
    Z=ASE(A,d);
    Y0=kmeans(Z,max(Y));
    t0=toc;
    tic
    [~,Y1]=UnsupGEE(A,max(Y),length(Y),opts1); t1=toc;tic
    [~,Y2]=UnsupGEE(A,max(Y),length(Y),opts2);t2=toc;tic
    [c1,t4]=CPL(A,max(Y),Y);
    [RandIndex(Y,Y2),RandIndex(Y,Y1),RandIndex(Y,Y0),c1]
    [t2,t1,t0,t4]
end

if choice >=20 && choice<30
    opts = struct('Adjacency',1,'Laplacian',1,'Spectral',1,'NN',0,'Dist','sqeuclidean','normalize',0,'dmax',30); % default parameter
    opts1 = struct('MaxIter',20,'Replicates',20,'Normalize',true,'Refine',0,'Metric',0,'Principal',0,'Laplacian',false,'Discriminant',0,'SeedY',0); % default parameter
    opts2 = struct('MaxIter',20,'Replicates',20,'Normalize',true,'Refine',0,'Metric',0,'Principal',0,'Laplacian',false,'Discriminant',0,'SeedY',0,'Transformer',1);% default parameter
    load('n2v.mat');time=zeros(1,6);skip=1;
    switch choice
        case 21 %
            load('Adjnoun.mat');A=Adj; Z=AdjNoun; %AEL / GFN K=2
        % case 25
        %     load('citeseer.mat');A=edge2adj(Edge); Y=Label; Z=Citeseer;%AEL / GFN K=2
        % case 27
        %     load('Cora.mat'); A=edge2adj(Edge);ind=(sum(A)>0); Y=Label(ind);A=A(ind,ind);Z=Cora; 
        case 22 %
            load('email.mat');A=Adj; Z=email;%k=42
        % case 25
        %     load('Gene.mat');A=Adj; Z=Gene;
        % case 25 %
        %     load('IIP.mat');A=Adj; Z=IIP;
        case 25 %?
            load('lastfm.mat');A=Adj;Z=lastfm;
        case 23 %
            load('polblogs.mat');A=Adj; Z=polblogs; %
        % case 29 
        %     load('pubmed.mat');A=edge2adj(Edge); Y=Label;%AEL / GFN K=2
        % case 29 
        %     load('IMDB.mat');A=Edge2; Y=Label2;%AEL / GFN K=2
        case 24 %
            load('karate.mat');A=G; Z=karate;
        % case 27 %
        %     load('web-spam-detection.mat');A=Edge; Y=Label;%Z=karate;
        % case 28 %
        %     load('TerroristRel.mat');A=edge2adj(Edge); Y=Label;%Z=karate;
        % case 29 
        %     load('letter.mat');A=Edge1;Y=Label1; 
        % case 32 
        %     load('letter.mat');A=Edge2;Y=Label2; 
        % case 33
        %     load('letter.mat');A=Edge3;Y=Label3;
        % case 27
        %     load('Wiki_Data.mat');A=GEAdj;Y=Label;Z=WikiGE;LeidenY=GELeidenY;
        % case 28
        %     load('Wiki_Data.mat');A=GFAdj;ind=(sum(A)>0); Y=Label(ind);A=A(ind,ind);Z=WikiGF;LeidenY=GFLeidenY;
        % case 42
        %     load('Wiki_Data.mat');A=GFAdj;Y=Label;Z=WikiGF;
        % case 28
        %     load('CElegans.mat');A=Ac;Y=vcols;Z=CElegansAc;
        % case 29
        %     load('CElegans.mat');A=Ag;Y=vcols;Z=CElegansAg;
        % case 47
        %     load('CElegans.mat');A={Ac,Ag};Y=vcols;
    end
    if skip==0
    result=GraphClusteringEvaluate(A,Y,opts);
    time(4)=table2array(result(2,3));
    time(3)=table2array(result(2,2));
    Y2=kmeans(Z,max(Y));
    % opts.SeedY=Y;
    tic
    [~,Y3]=UnsupGEE(A,max(Y),length(Y),opts1);
    time(2)=toc;
    tic
    [~,Y4]=UnsupGEE(A,max(Y),length(Y),opts2);
    time(1)=toc;
    [c1,time(6)]=CPL(A,max(Y),Y);
    [RandIndex(Y,Y4),RandIndex(Y,Y3),table2array(result(1,2)),table2array(result(1,3)),RandIndex(Y,Y2),RandIndex(Y,LeidenY),c1]
    time
    else
        tic
        [~,Y4]=UnsupGEE(A,max(Y),length(Y),opts2); 
        t4=toc;
        [c1,t]=CPL(A,max(Y),Y);
        [RandIndex(Y,Y4),c1]
        [t4,t]
    end
end


%%% 3. No changce / similar
% load('Adjnoun.mat') %AEL / GFN K=2
% load('DD244.mat') %AEL / GFN K=2
% load('pubmedAdj.mat') %k=3;
% load('Wiki_Data.mat')
% load('graphCElegans.mat')
% % Simple Eval
% tic
% [~,Y1,~,~]=GraphEncoder(Adj,K);
% toc
% RandIndex(Y,Y1)
% tic
% [~,Y1,~,~,score]=GraphEncoder(Adj,[2:10]);
% toc
% RandIndex(Y,Y1)
% % Try K choice
% kmax=10;
% score=zeros(kmax,1);ari=zeros(kmax,1);
% for r=2:kmax
%     [~,Y2,~,~,score(r)]=GraphEncoder(Adj,r);
%     ari(r)=RandIndex(Y2,Y+1);
% end

%% fusion
if choice==51
n=1000;K=10;
[Adj,Y]=simGenerate(19,n);opts.Dim=K;
GraphClusteringEvaluate(Adj,Y)
n=1000;K=10;
[Adj,Y]=simGenerate(29,n);opts.Dim=K;
GraphClusteringEvaluate(Adj,Y)
%%
load('Wiki_Data.mat')
Label=Label+1;
opts.Weight=[1,0];
GraphClusteringEvaluate(GEAdj,Label,opts)
GraphClusteringEvaluate(GFAdj,Label,opts)
GraphClusteringEvaluate(TE,Label,opts)
GraphClusteringEvaluate(TF,Label,opts)
GraphClusteringEvaluate({TE,TF},Label,opts)
GraphClusteringEvaluate({TE,GE},Label,opts)
GraphClusteringEvaluate({GEAdj,GFAdj},Label,opts)
GraphClusteringEvaluate({TE,TF,GEAdj,GFAdj},Label,opts)
load('graphCElegans.mat')
GraphClusteringEvaluate(Ag,vcols)
GraphClusteringEvaluate(Ac,vcols)
GraphClusteringEvaluate({Ac,Ag},vcols)
% [PB_RI_AEE,PB_RI_AEN,PB_RI_ASE,PB_t_AEE,PB_t_AEE2,PB_t_AEN,PB_t_ASE]=simClusteringReal(Adj,Label);
end

%% n increase and mse

if choice==100
n=100;K=5;kmax=20;nmax=50;
score=zeros(kmax,nmax);
ari=zeros(kmax,nmax);
for i=1:nmax
    [Adj,Y]=simGenerate(11,i*n,K);
    for r=2:kmax
        [~,Y2,~,~,score(r,i)]=GraphEncoder(Adj,r);
        ari(r,i)=RandIndex(Y2,Y+1);
    end
end
% Try K choice
kmax=10;n=1000;K=5;type=11;rep=10;
score=zeros(kmax,1);
ari=zeros(kmax,1);
[Adj,Y]=simGenerate(type,n,K);
for i=1:rep
for r=2:kmax
    [~,Y2]=GraphEncoder(Adj,r);
    ari(r)=ari(r)+RandIndex(Y2,Y+1)/rep;
end
end

subplot(1,2,1);
plot(2:kmax,1-score(2:kmax),'r-','LineWidth',2)
title(strcat('1-MeanSS at K=',num2str(K)))
xlim([2,kmax]);
axis('square')
set(gca,'FontSize',15);
subplot(1,2,2);
plot(2:kmax,ari(2:kmax),'b-','LineWidth',2)
title(strcat('ARI at K=',num2str(K)))
xlim([2,kmax]);
axis('square')
set(gca,'FontSize',15);
end

function [c1,t1]=CPL(A,K,Y);

% scp method
compErr = @(c,e) compMuI(compCM(c,e,K));    % use mutual info as a measure of error/sim.

tic
init_opts = struct('verbose',false);
[e] = initLabel5b(A, K, 'scp', init_opts);
% t1=toc;
% c1 = compErr(Y, e);

% CPL method
T = 20;
% tic
cpl_opts = struct('verbose',false,'delta_max',0.000000001,'itr_num',T,'em_max',500,'track_err',true);
[c1] = cpl4c(A, K, e, Y, 'cpl', cpl_opts);
t1=toc;
% fprintf('%3.5fs\n',RT_cpl)
c1 = RandIndex(Y,c1); %compErr(Y, c1);