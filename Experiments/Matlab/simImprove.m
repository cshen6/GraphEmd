function []=simImprove(opt)

% Visualization
fpath = mfilename('fullpath');
fpath=strrep(fpath,'\','/');
findex=strfind(fpath,'/');
rootDir=fpath(1:findex(end));
pre=strcat(rootDir,'');% The folder to save figures
fs=30;
lw=3;
rep=100;
opts0 = struct('DiagA',true,'Normalize',false,'Laplacian',false,'Replicates',1);
opts1 = struct('DiagA',true,'Normalize',true,'Laplacian',false,'Replicates',1);
opts2 = struct('DiagA',true,'Normalize',true,'Laplacian',false,'Replicates',10);
map2 = brewermap(128,'PuBuGn'); % brewmap
% colormap(gca,map2);
if opt==0
%     n=500;K=5;
%     [Adj,Y]=simGenerate(11,n,K);
%     subplot(1,2,1)
%     ind1=[];
%     for i=1:K
%         ind1=[ind1;find(Y==i)];
%     end
%     heatmap(Adj(ind1,ind1),'GridVisible','off');
%     Ax = gca;
%     Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
%     Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
%     colorbar( 'off' )
%     colormap default
%     title('Dense Graph')
% %     axis('square')
%     set(gca,'FontSize',fs);

    n=1000;K=5; fs=15; 
    [Adj,Y]=simGenerate(25,n,2);
    subplot(1,3,1)
    ind1=[];
    for i=1:K
        ind1=[ind1;find(Y==i)];
    end
%     G=graph(Adj,'omitselfloops');
%     plot(G)
    heatmap(Adj(ind1,ind1),'ColorMap',map2,'GridVisible','off');
    Ax = gca;
    Ax.XDisplayLabels = nan(size(Ax.XDisplayData));
    Ax.YDisplayLabels = nan(size(Ax.YDisplayData));
    colorbar( 'off' )
%     colormap default
    title('Sparse Graph')
%     axis('square')
    set(gca,'FontSize',fs);

    subplot(1,3,2)
    Z=GraphEncoder(Adj,Y,opts0);
    hold on
    plot(Z(Y==1,1),Z(Y==1,2),'ro');
    plot(Z(Y==2,1),Z(Y==2,2),'bx');
    hold off
    Y2=kmeans(Z,2);
    title(strcat('ARI = ',{' '},num2str(floor(RandIndex(Y,Y2)*100)/100)));
%     axis('square')
    set(gca,'FontSize',fs);

    subplot(1,3,3)
    Z=GraphEncoder(Adj,Y,opts1);
    hold on
    plot(Z(Y==1,1),Z(Y==1,2),'ro');
    plot(Z(Y==2,1),Z(Y==2,2),'bx');
    hold off
    Y2=kmeans(Z,2);
    title(strcat('ARI = ',{' '},num2str(floor(RandIndex(Y,Y2)*100)/100)));
%     axis('square')
    set(gca,'FontSize',fs);

    currentFolder = pwd;
    F.fname=strcat(strcat(currentFolder,'\FigImprove0'));
    F.wh=[6 2]*2;
    F.PaperPositionMode='auto';
    print_fig(gcf,F)
end

% Normalize Experiment
if opt==1
    load('Improve\ClusteringChoice25.mat')
    subplot(1,3,1)
    hold on
    ln=1:lim;
    plot(ln,accMDRI(2,:),'-.','LineWidth',lw);
    load('Improve\ClusteringChoice20.mat')
    plot(ln,accMDRI(3,:),'-.','LineWidth',lw);
    load('Improve\ClusteringChoice22.mat')
    plot(ln,accMDRI(4,:),'-.','LineWidth',lw);
    load('Improve\ClusteringChoice21.mat')
    plot(ln,accMDRI(5,:),'-.','LineWidth',lw);
    xlim([1,lim])
    xticks([1,3,lim])
    xticklabels({'1000','3000','5000'})
    xlabel('Sample Size')
    ylabel('Estimation Accuracy')
    title('Cluster Choice via MRI');
    ylim([0,1.05])
    legend('Sim 1','Sim 2','Sim 3','Sim 4', 'Location','SouthEast');
    hold off
    axis('square')

    load('Improve\ClusteringChoice21.mat')
    subplot(1,3,2)
    bar(2:kmax,scoreMDRI(2:kmax,lim))
    xlim([2-0.5,kmax+0.5])
    xlabel('Cluster Size')
    ylabel('Minimum Rank Index')
    title('Cluster Choice via MRI for Sim 4');
    axis('square')

    subplot(1,3,3)
    bar(2:kmax,-scoreSI(2:kmax,lim))
    xlim([2-0.5,kmax+0.5])
    xlabel('Cluster Size')
    ylabel('Silhouette Score')
    title('Cluster Choice via SS for Sim 4');
    axis('square')

    currentFolder = pwd;
    F.fname=strcat(strcat(currentFolder,'\FigImprove1'));
    F.wh=[6 2]*2;
    F.PaperPositionMode='auto';
    print_fig(gcf,F)
end

if opt==2
    fs=15;
    load('polblogs.mat') %k=2
    [Z2,Y2]=GraphEncoder(Adj,2,opts2);
%     [Z2,Y2]=GraphEncoder(Adj,Y2,opts0);
    subplot(1,2,1)
    plot3(Z2(Y2==1,1),Z2(Y2==1,2),zeros(sum(Y2==1),1)','ro');
    hold on
    plot3(Z2(Y2==2,1),Z2(Y2==2,2),zeros(sum(Y2==2),1)','bx');
    % plot(Z(Y==3,1),Z(Y==3,2),'bs');
    hold off
    title('GEE at K=2');
    axis('square')
    %title('DC-SBM Graph','FontSize',fs)
    set(gca,'FontSize',fs);
    [Z3,Y3]=GraphEncoder(Adj,3,opts2);
%     [Z3,Y3]=GraphEncoder(Adj,Y3,opts0);
    subplot(1,2,2)
    plot3(Z3(Y3==1,1),Z3(Y3==1,2),Z3(Y3==1,3),'ro');
    hold on
    plot3(Z3(Y3==2,1),Z3(Y3==2,2),Z3(Y3==2,3),'bx');
    plot3(Z3(Y3==3,1),Z3(Y3==3,2),Z3(Y3==3,3),'g*');
    % plot(Z(Y==3,1),Z(Y==3,2),'bs');
    title('GEE at K=3');
    axis('square')
    set(gca,'FontSize',fs);
    hold off

    currentFolder = pwd;
    F.fname=strcat(strcat(currentFolder,'\FigImprove2'));
    F.wh=[4 2]*2;
    F.PaperPositionMode='auto';
    print_fig(gcf,F)
end

if opt==3    
    load('AEETimeEdge.mat');
    load('AEETimeEdge2.mat');
    figure('units','normalized','Position',[0 0 1 1]);
    x=[10^3,5*10^3,10^4,5*10^4,10^5,5*10^5,10^6,5*10^6,10^7,5*10^7,10^8,5*10^8,10^9];fs=25;lw=3;
%     x=1000*10.^(x-1);
    loglog(x,t1,'-','LineWidth',lw);
    hold on
    x=x(1:8);
%     semilogy(x,t2,'--','LineWidth',lw);
    loglog(x,t3,'--','LineWidth',lw);
    loglog(x,t4,'--','LineWidth',lw);
    loglog(x,t5,'--','LineWidth',lw);
    hold off
    xlim([10^3,10^9])
    xticks([10^3,10^6,10^9])
    xticklabels({'10^3','10^6','10^9'})
    legend('AEE','ASE (sparse SVD)','GCN (30 epoch)','Node2Vec','Location','NorthWest');
    xlabel('Number of Edges (log scale)')
    ylabel('Running Time (log scale)')
    axis('square')
    set(gca,'FontSize',fs);
    currentFolder = pwd;
    F.fname=strcat(strcat(currentFolder,'FigAEE6'));
    F.wh=[5 5]*2;
    F.PaperPositionMode='auto';
    print_fig(gcf,F)
    %xlim([1,ln])
    %ylim([0.5e-3,3e4]);
    %xticks([1000,l000000,1000000000]);
end

% Normalize Experiment
if opt==10
    %     result1=simRunClassify(20,opts0,opts2,100,1)
    %     result2=simRunClassify(21,opts0,opts2,100,1)
    %     result3=simRunClassify(22,opts0,opts2,100,1)
    %     result4=simRunClassify(25,opts0,opts2,200,1)
    rep=100;
% %     simRunCluster(10,opts0,opts1,2000,10);
% %     simRunCluster(11,opts0,opts1,2000,10);
% %     simRunCluster(13,opts0,opts1,1000,10);
% %     simRunCluster(15,opts0,opts3,500,10);
    simRunCluster(20,opts0,opts1,5000,rep);
    simRunCluster(21,opts0,opts1,3000,rep);
    simRunCluster(22,opts0,opts1,3000,rep);
    simRunCluster(25,opts0,opts1,3000,rep);
%     %     result3=simRunCluster(26,opts0,opts3,1000,10)
%     simRunCluster(27,opts0,opts1,1000,rep)
%     simRunCluster(29,opts0,opts1,1000,rep)
end

% Replicates Experiment
if opt==11
    rep=100;
    load('lastfm.mat') %AEK K=18
    simRunClusterReal2('lastFM',K,Adj,Y,opts1,opts2,rep);
    load('polblogs.mat') %k=2
    simRunClusterReal2('polblogs',K,Adj,Y,opts1,opts2,rep);
    load('CoraAdj.mat') %AEL / GFN K=7
    simRunClusterReal2('Cora',K,Adj,Y,opts1,opts2,rep);
    load('email.mat') %k=42
    simRunClusterReal2('email',K,Adj,Y,opts1,opts2,rep);
    [Adj,Y]=simGenerate(20,5000,3);
    simRunClusterReal2('20',3,Adj,Y,opts1,opts2,rep);
    [Adj,Y]=simGenerate(21,3000,5);
    simRunClusterReal2('21',5,Adj,Y,opts1,opts2,rep);
    [Adj,Y]=simGenerate(22,3000,4);
    simRunClusterReal2('22',4,Adj,Y,opts1,opts2,rep);
    [Adj,Y]=simGenerate(25,3000,2);
    simRunClusterReal2('25',2,Adj,Y,opts1,opts2,rep);

%     [Adj,Y]=simGenerate(10,1000,3);
%     simRunClusterReal2('10',3,Adj,Y,opts1,opts2,rep);
%     [Adj,Y]=simGenerate(11,1000,5);
%     simRunClusterReal2('11',5,Adj,Y,opts1,opts2,rep);
%     [Adj,Y]=simGenerate(13,500,3);
%     simRunClusterReal2('13',3     ,Adj,Y,opts1,opts2,rep);
%     [Adj,Y]=simGenerate(16,500,2);
%     simRunClusterReal2('16',2,Adj,Y,opts1,opts2,rep);
end

% Cluster Size Simulation
if opt==12
     rep=100;
% % >
% %       simRunClusterChoice(17,n,rep) %2
% %       simRunClusterChoice(18,n,rep) %4
load('lastfm.mat') %AEK K=18
simRunClusterReal('lastFM',2,30,Adj,Y,opts2)
load('polblogs.mat') %k=2
simRunClusterReal('polblogs',2,10,Adj,Y,opts2)
load('CoraAdj.mat') %AEL / GFN K=7
simRunClusterReal('Cora',2,20,Adj,Y,opts2)
load('email.mat') %k=42
simRunClusterReal('email',10,50,Adj,Y,opts2)
load('Gene.mat') %AEK K=18
simRunClusterReal('gene',2,20,Adj,Y,opts2)
load('IIP.mat') %AEL / GFN K=3
simRunClusterReal('iip',2,20,Adj,Y,opts2)

% simRunClusterChoice(20,5000,rep,opts2); %3 
% simRunClusterChoice(21,3000,rep,opts2); %5 
% simRunClusterChoice(22,3000,rep,opts2); %4 
% simRunClusterChoice(25,3000,rep,opts2); %2 
% simRunClusterChoice(13,450,rep,opts2); %4
%       simRunClusterChoice(26,n,rep) %2
%       simRunClusterChoice(27,n,rep) %2
%       simRunClusterChoice(28,n,rep) %4
%        simRunClusterChoice(29,n,rep) %4?
%%%     result4=simRunClusterChoice(21,2000,10)
%real data: lastFM
end

% running time benchmark
if opt==13
    %%% running time
rep=1;nn=1000;ln=11;
opts1 = struct('Replicates',1);
opts2 = struct('Replicates',5);
t6=zeros(ln,rep);
t7=zeros(ln,rep);
for i=1:ln
    s=10^(floor((i-1)/2))*nn*5^(mod(i+1,2));
    n=s/100;
    for r=1:rep
        
        Edge=zeros(s,3);
        Edge(:,1)=randi(n,1,s);
        Edge(:,2)=randi(n,1,s);
        Edge(:,3)=1;
        [~,~,Y]=unique(randi(10,1,n));
        
        tic
        GraphEncoder(Edge,3,opts1);
        t6(i,r)=toc;

        tic
        GraphEncoder(Edge,3,opts2);
        t7(i,r)=toc;
    end
    save(strcat('AEETimeEdge2.mat'),'t6','t7');
end
end

% % Cluster Size Experiment
% if opt==5;
% load('lastfm.mat') %AEK K=18
% simRunClusterReal('lastFM',30,Adj,Y);
% load('polblogs.mat') %k=2
% simRunClusterReal('polblogs',10,Adj,Y);
% load('CoraAdj.mat') %AEL / GFN K=7
% simRunClusterReal('Cora',20,Adj,Y);
% load('email.mat') %k=42
% simRunClusterReal('email',60,Adj,Y);
% end

function simRunClusterReal2(type,K,Adj,Y,opts1,opts2,rep)
ari1=zeros(rep,1);ari2=zeros(rep,1);ari3=zeros(rep,1);
score1=zeros(rep,1);score2=zeros(rep,1);score3=zeros(rep,1);
for r=1:rep
    sd=randi(10000);
    rng(sd);
    [~,Y1,~,~,score1(r)]=GraphEncoder(Adj,K,opts1);
    rng(sd);
    [~,Y2,~,~,score2(r)]=GraphEncoder(Adj,K,opts2);
    rng(sd);
    [~,Y3,~,~,score3(r)]=GraphEncoderSil(Adj,K,opts2);
    ari1(r)=RandIndex(Y1,Y+1);
    ari2(r)=RandIndex(Y2,Y+1);
    ari3(r)=RandIndex(Y3,Y+1);
end
ariStd=[std(ari1),std(ari2),std(ari3)];
scoreStd=[std(score1),std(score2),std(score3)];
ari=[mean(ari1),mean(ari2),mean(ari3)];
score=[mean(score1),mean(score2),mean(score3)];
save(strcat('Improve\Replicates',type,'.mat'),'K','ari','score','ariStd','scoreStd');

function simRunClusterReal(type,kmin,kmax,Adj,Y,opts2)
K=max(Y);
scoreMDRI=zeros(kmax,1);ariMDRI=zeros(kmax,1);tmpMax1=1;KEstMDRI=kmin;KEstSI=kmin;tmpMax2=0;
scoreSI=zeros(kmax,1);ariSI=zeros(kmax,1);
for r=kmin:kmax
    sd=randi(10000);
    rng(sd);
    [~,Y1,~,~,scoreMDRI(r)]=GraphEncoder(Adj,r,opts2);
    rng(sd);
    [~,Y2,~,~,scoreSI(r)]=GraphEncoderSil(Adj,r,opts2);
    ariMDRI(r)=RandIndex(Y1,Y+1);
    ariSI(r)=RandIndex(Y2,Y+1);
    if scoreMDRI(r)<=tmpMax1
        KEstMDRI=r;tmpMax1=scoreMDRI(r);
    end
    if scoreSI(r)>=tmpMax2
        KEstSI=r;tmpMax2=scoreSI(r);
    end
end
save(strcat('Improve\Clustering',type,'.mat'),'K','ariMDRI','scoreMDRI','KEstMDRI','ariSI','scoreSI','KEstSI');

function result=simRunCluster(type,opts1,opts2,n,rep)
lim=10;k=5;
acc1=zeros(rep,lim);acc2=zeros(rep,lim);acc4=zeros(rep,lim);acc5=zeros(rep,lim);
time1=zeros(rep,lim);time2=zeros(rep,lim);time4=zeros(rep,lim);time5=zeros(rep,lim);
MDRI1=zeros(rep,lim);MDRI2=zeros(rep,lim);
opts1.Adjacency=1;opts1.Laplacian=1;opts1.Spectral=1;
opts2.Adjacency=1;opts2.Laplacian=0;opts2.Spectral=0;
for q=1:lim
    nn=n/lim*q;
    for r=1:rep
        [Adj,Y]=simGenerate(type,nn,k);
        SBM1=GraphClusteringEvaluate(Adj,Y,opts1);
        SBM2=GraphClusteringEvaluate(Adj,Y,opts2);
        acc1(r,q)=SBM1{1,1};acc2(r,q)=SBM2{1,1};acc4(r,q)=SBM1{1,2};acc5(r,q)=SBM1{1,5};
        time1(r,q)=SBM1{2,1};time2(r,q)=SBM2{2,1};time4(r,q)=SBM1{2,2};time5(r,q)=SBM1{2,5};
        MDRI1(r,q)=SBM1{3,1};MDRI2(r,q)=SBM2{3,1};MDRI3(r,q)=SBM1{3,4};
    end
end
acc3=acc2-acc1;time3=time2-time1;
accN=[mean(acc1,1)',mean(acc2,1)',mean(acc3,1)',mean(acc4,1)',mean(acc5,1)'];
stdN=[std(acc1,0,1)',std(acc2,0,1)',std(acc3,0,1)',std(acc4,0,1)',std(acc5,0,1)'];
time=[mean(time1,1)',mean(time2,1)',mean(time3,1)',mean(time4,1)',mean(time5,1)'];
stdT=[std(time1,0,1)',std(time2,0,1)',std(time3,0,1)',std(time4,0,1)',std(time5,0,1)'];
MDRI=[mean(MDRI1,1)',mean(MDRI2,1)',mean(MDRI3,1)'];
result = accN; %array2table([accN; stdN; time;stdT], 'RowNames', {'acc', 'stdAcc', 'time','stdTime'},'VariableNames', {'AEE', 'AEE_New','AEE_Diff','ASE','LSE'});
save(strcat('Improve\Clustering',num2str(type),'.mat'),'n','lim','accN','MDRI','stdN','time','stdT','opts1','opts2');

function simRunClusterChoice(type,n,rep,opts2)
lim=5;
K=5;kmax=10;
scoreMDRI=zeros(kmax,lim);accMDRI=zeros(kmax,lim);ariMDRI=zeros(kmax,lim);tmpInd1=2;
scoreSI=zeros(kmax,lim);accSI=zeros(kmax,lim);ariSI=zeros(kmax,lim);tmpInd2=2;
for i=1:lim
    nn=n/lim*i;
    for r=1:rep
        [Adj,Y]=simGenerate(type,nn,K);
        tmp1=zeros(kmax,1);tmpMax1=1;tmp2=zeros(kmax,1);tmpMax2=0;
        for k=2:kmax
            sd=randi(10000);
            rng(sd);
            [~,Y1,~,~,tmp1(k)]=GraphEncoder(Adj,k,opts2);
            rng(sd);
            [~,Y2,~,~,tmp2(k)]=GraphEncoderSil(Adj,k,opts2);
            ariMDRI(k,i)=ariMDRI(k,i)+RandIndex(Y1,Y+1)/rep;
            ariSI(k,i)=ariSI(k,i)+RandIndex(Y2,Y+1)/rep;
            if tmp1(k)<=tmpMax1
                tmpInd1=k;tmpMax1=tmp1(k);
            end
            if tmp2(k)>=tmpMax2
                tmpInd2=k;tmpMax2=tmp2(k);
            end
        end
        scoreMDRI(:,i)=scoreMDRI(:,i)+tmp1/rep;accMDRI(tmpInd1,i)=accMDRI(tmpInd1,i)+1/rep;
        scoreSI(:,i)=scoreSI(:,i)+tmp2/rep;accSI(tmpInd2,i)=accSI(tmpInd2,i)+1/rep;
    end
end
save(strcat('Improve\ClusteringChoice',num2str(type),'.mat'),'n','lim','scoreMDRI','accMDRI','ariMDRI','scoreSI','accSI','ariSI','kmax','lim');


function result=simRunClassify(type,opts1,opts2,n,rep)
k=10;lim=10;
acc1=zeros(rep,lim);acc2=zeros(rep,lim);acc4=zeros(rep,lim);acc5=zeros(rep,lim);
time1=zeros(rep,lim);time2=zeros(rep,lim);time4=zeros(rep,lim);time5=zeros(rep,lim);
opts1.Adjacency=1;opts1.Laplacian=1;opts1.Spectral=1;
opts2.Adjacency=1;opts2.Laplacian=0;opts2.Spectral=0;
for q=1:lim
    nn=n/lim*q;
    for r=1:rep
        [Adj,Y]=simGenerate(type,n,k);
        indices = crossvalind('Kfold',Y,10);
        opts1.indices=indices;
        opts2.indices=indices;
        SBM1=GraphEncoderEvaluate(Adj,Y,opts1);
        SBM2=GraphEncoderEvaluate(Adj,Y,opts2);
        acc1(r)=SBM1{1,2};acc2(r)=SBM2{1,2};acc4(r)=SBM1{1,4};acc5(r)=SBM1{1,8};
        time1(r)=SBM1{4,2};time2(r)=SBM2{4,2};time4(r)=SBM1{4,4};time5(r)=SBM1{4,8};
    end
end
acc3=acc2-acc1;time3=time2-time1;
acc6=acc5-acc4;time6=time5-time4;
accN=[mean(acc1),mean(acc2),mean(acc3),mean(acc4),mean(acc5),mean(acc6)];
stdN=[std(acc1),std(acc2),std(acc3),std(acc4),std(acc5),std(acc6)];
time=[mean(time1),mean(time2),mean(time3),mean(time4),mean(time5),mean(time6)];
stdT=[std(time1),std(time2),std(time3),std(time4),std(time5),std(time6)];
result = array2table([accN; stdN; time;stdT], 'RowNames', {'acc', 'stdAcc', 'time','stdTime'},'VariableNames', {'AEE', 'AEE_New','AEE_Diff','ASE', 'LSE','ASE_LSE_Diff'});