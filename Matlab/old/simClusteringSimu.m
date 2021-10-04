function [t_AEE,RI_AEE,t_ASE,RI_ASE,ind_AEE,ind_ASE,Y,RI]=simClusteringSimu(n,opts)

% n=10000;opts = struct('model','SBM','AEE',1,'ASE',1,'edgeV',1,'Dist','sqeuclidean'); % default parameters
% [t_AEE,RI_AEE,t_ASE,RI_ASE,ind_AEE,ind_ASE,Y,RI]=simClusteringSimu(n, opts)
% n=10000;opts = struct('model','DCSBM','AEE',1,'ASE',1,'edgeV',1,'Dist','cosine'); % default parameters
% [t_AEE,RI_AEE,t_ASE,RI_ASE,ind_AEE,ind_ASE,Y,RI]=simClusteringSimu(n, opts)
% n=10000;opts = struct('model','RDPG','AEE',1,'ASE',1,'edgeV',1,'Dist','sqeuclidean'); % default parameters
% [t_AEE,RI_AEE,t_ASE,RI_ASE,ind_AEE,ind_ASE,Y,RI]=simClusteringSimu(n, opts)

fpath = mfilename('fullpath');
fpath=strrep(fpath,'\','/');
findex=strfind(fpath,'/');
rootDir=fpath(1:findex(end));
pre=strcat(rootDir,'');% The folder to save figures
fs=15;

if nargin < 2
    opts = struct('model','SBM','AEE',1,'ASE',1,'plot',1,'edgeV',0,'Dist','sqeuclidean'); % default parameters
end
if ~isfield(opts,'model'); opts.model='SBM'; end
if ~isfield(opts,'AEE'); opts.ASE=1; end
if ~isfield(opts,'ASE'); opts.AEE=1; end
if ~isfield(opts,'plot'); opts.plot=1; end
if ~isfield(opts,'edgeV'); opts.edgeV=0; end
if ~isfield(opts,'Dist'); opts.Dist='sqeuclidean'; end
fs=15; K=3; d=3;
t_AEE=0;RI_AEE=0;t_ASE=0;RI_ASE=0;RI=0; 

if strcmpi(opts.model,'SBM')
    [Adj,Y]=simGenerate(1,n);
end
if strcmpi(opts.model,'DCSBM')
    [Adj,Y]=simGenerate(2,n);
end
if strcmpi(opts.model,'RDPG')
    [Adj,Y]=simGenerate(3,n);
end

if opts.edgeV==1
    n=size(Adj,1);r=1;Edge=0;
    for i=1:n
        for j=i+1:n
            if Adj(i,j)==1
                Edge(r,1)=i;
                Edge(r,2)=j;
                r=r+1;
            end
        end
    end
end

if opts.AEE==1
    tic
    if opts.edgeV==1
       [ind_AEE,Z_AEE]=GraphClustering(Edge,K);
    else
       [ind_AEE,Z_AEE]=GraphClustering(Adj,K);
    end
    t_AEE=toc;
    RI_AEE=RandIndex(Y,ind_AEE);
%     if opts.plot==1
        figure('units','normalized','Position',[0 0 1 1]);
        subplot(1,2,1)
        plot3(Z_AEE(ind_AEE==1,1),Z_AEE(ind_AEE==1,2),Z_AEE(ind_AEE==1,3),'o');
        hold on
        plot3(Z_AEE(ind_AEE==2,1),Z_AEE(ind_AEE==2,2),Z_AEE(ind_AEE==2,3),'o');
        plot3(Z_AEE(ind_AEE==3,1),Z_AEE(ind_AEE==3,2),Z_AEE(ind_AEE==3,3),'o');
        hold off
        title(strcat('GFN Clustering for ',{' '},opts.model),'FontSize',fs)
        xlabel(strcat('ARI = ',{' '}, num2str(round(RI_AEE*100)/100),{'; '}, 'Time = ',{' '},num2str(round(t_AEE*100)/100),{' '},'seconds'));
        set(gca,'FontSize',fs);
%     end
end

if opts.ASE==1
    tic
    [U,S,~]=svds(Adj,d);
    Z_ASE=U(:,1:d)*S(1:d,1:d)^0.5;
    ind_ASE = kmeans(Z_ASE, K);
    t_ASE=toc;
    RI_ASE=RandIndex(Y,ind_ASE);
    RI=RandIndex(ind_AEE,ind_ASE);
%     if opts.plot==1
        subplot(1,2,2)
        plot3(Z_ASE(ind_ASE==1,1),Z_ASE(ind_ASE==1,2),Z_ASE(ind_ASE==1,3),'o');
        hold on
        plot3(Z_ASE(ind_ASE==2,1),Z_ASE(ind_ASE==2,2),Z_ASE(ind_ASE==2,3),'o');
        plot3(Z_ASE(ind_ASE==3,1),Z_ASE(ind_ASE==3,2),Z_ASE(ind_ASE==3,3),'o');
        hold off
        title(strcat('ASE Clustering for ',{' '},opts.model),'FontSize',fs)
        xlabel(strcat('ARI = ',{' '}, num2str(round(RI_ASE*100)/100),{'; '}, 'Time = ',{' '},num2str(round(t_ASE*100)/100),{' '},'seconds'));
        set(gca,'FontSize',fs);
%     end
end

if opts.plot==1
F.fname=strcat(pre, 'Fig1',opts.model);
F.wh=[8 3]*2;
F.PaperPositionMode='auto';
print_fig(gcf,F)
end
