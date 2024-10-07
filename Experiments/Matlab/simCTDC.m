function out=simCTDC(option)

if option==0 || option==1
    %%% Graph Processing into Annual
    load('CTDC.mat')
    Time=table2array(X(:,1));
    Data=X(:,7:end); % remove categorical variables
    Y=Y(7:end);Y=Y-2;K=max(Y);
    idx=~isnan(Time); Time=Time(idx,:); Data=Data(idx,:); X=X(idx,2:end);% index by year
    headers = Data.Properties.VariableNames; headers = categorical(headers);
    Data=table2array(Data); Data(isnan(Data))=0;
    n=length(Time);
    [a,~,~]=unique(Time); % find all years
    Time=Time-min(a)+1; TimeMax=max(Time);% re-arrange starting year to 1
    % add gender
    [a,b,c]=unique(X.gender);lim=3;gender=a(1:lim);
    Data2=zeros(n,lim);
    for i=1:lim
        Data2(X.gender == a(i), i) = 1;
    end
    Data=[Data,Data2];headers=[headers,gender'];
    K=K+1;Y=[Y;K*ones(lim,1)];
    % add agebroad
    [a,b,c]=unique(X.ageBroad);lim=9;age=a(1:lim);
    Data2=zeros(n,lim);
    for i=1:lim
        Data2(X.ageBroad == a(i), i) = 1;
    end
    Data=[Data,Data2]; headers=[headers,age'];
    K=K+1;Y=[Y;K*ones(lim,1)];
    % add country
    [a,b,c]=unique(X.citizenship);lim=54;country=a(1:lim);
    Data2=zeros(n,lim);
    for i=1:lim
        Data2(X.citizenship == a(i), i) = 1;
    end
    Data=[Data,Data2]; headers=[headers,country'];
    K=K+1;Y=[Y;K*ones(lim,1)];
    % add country of exploitation
    [a,b,c]=unique(X.CountryOfExploitation);lim=68;countryE=a(1:lim);
    Data2=zeros(n,lim);
    for i=1:lim
        Data2(X.CountryOfExploitation == a(i), i) = 1;
    end
    Data=[Data,Data2]; headers=[headers,countryE'];
    K=K+1;Y=[Y;K*ones(lim,1)];
    % add traffic month
    [a,b,c]=unique(X.traffickMonths);lim=3;trafficM=a(1:lim);
    Data2=zeros(n,lim);
    for i=1:lim
        Data2(X.traffickMonths == a(i), i) = 1;
    end
    Data=[Data,Data2]; headers=[headers,trafficM'];
    K=K+1;Y=[Y;K*ones(lim,1)];
    Y=1:size(Data,2);
    [Dist]=GraphProcess(Data,Y,Time);
    size(Dist)
    % size(Dist)
    % xplot=zeros(TimeMax,2);
    for i=1:TimeMax
        idx=[42,83,84];
        xplot1(i,:)=Dist(25,idx,i);
        xplot2(i,:)=Dist(24,idx,i);
        % % idx=[99,152,153];
        xplot3(i,:)=Dist(152,idx,i);
        xplot4(i,:)=Dist(5,idx,i);
    end
    % xplot
    fs=24;
    if option==1
        tcl = tiledlayout(2,2);
        myColor = brewermap(20,'Spectral');
        nexttile
        hold on
        plot(1:TimeMax,xplot1(:,1),'color',myColor(14,:),'LineWidth',2);
        plot(1:TimeMax,xplot1(:,2),'color',myColor(2,:),'LineWidth',2);
        plot(1:TimeMax,xplot1(:,3),'color',myColor(8,:),'LineWidth',2);
        legend('(0-8,CHN)','(0-8,UKR)','(0-8,USA)','Location','SouthEast','FontSize',20)
        xlabel('Year');
        xlim([1,21]);xticks([1 11 21]);xticklabels({'2002','2012','2022'});ylim([-0.05,0.05]);
        axis('square'); 
        title('AgeBroad vs Citizenship')
        set(gca,'FontSize',fs); 
        nexttile
        hold on
        plot(1:TimeMax,xplot2(:,1),'color',myColor(14,:),'LineWidth',2);
        plot(1:TimeMax,xplot2(:,2),'color',myColor(2,:),'LineWidth',2);
        plot(1:TimeMax,xplot2(:,3),'color',myColor(8,:),'LineWidth',2);
        legend('(Women,CHN)','(Women,UKR)','(Women,USA)','Location','SouthEast','FontSize',20)
        xlabel('Year');
        xlim([1,21]);xticks([1 11 21]);xticklabels({'2002','2012','2022'});ylim([-0.05,0.05]);
        axis('square'); 
        title('Gender vs Citizenship')
        set(gca,'FontSize',fs); 
        nexttile
        hold on
        plot(1:TimeMax,xplot3(:,1),'color',myColor(14,:),'LineWidth',2);
        plot(1:TimeMax,xplot3(:,2),'color',myColor(2,:),'LineWidth',2);
        plot(1:TimeMax,xplot3(:,3),'color',myColor(8,:),'LineWidth',2);
        legend('(UKR,CHN)','(UKR,UKR)','(UKR,USA)','Location','SouthEast','FontSize',20)
        xlabel('Year');
        xlim([1,21]);xticks([1 11 21]);xticklabels({'2002','2012','2022'});ylim([-0.05,0.15]);
        axis('square'); 
        title('Country vs Citizenship')
        set(gca,'FontSize',fs); 
        nexttile
        hold on
        plot(1:TimeMax,xplot4(:,1),'color',myColor(14,:),'LineWidth',2);
        plot(1:TimeMax,xplot4(:,2),'color',myColor(2,:),'LineWidth',2);
        plot(1:TimeMax,xplot4(:,3),'color',myColor(8,:),'LineWidth',2);
        legend('(False Promise,CHN)','(False Promise,UKR)','(False Promise,USA)','Location','SouthEast','FontSize',20)
        xlabel('Year');
        xlim([1,21]);xticks([1 11 21]);xticklabels({'2002','2012','2022'});ylim([-0.05,0.05]);
        axis('square'); 
        title('Means vs Citizenship')
        set(gca,'FontSize',fs); 
    end
    % 
    % plot(1:TimeMax,xplot(:,1),1:TimeMax,xplot(:,2),'LineWidth',2);
    % out=reshape(Dist(1,:,:),size(Dist,2),size(Dist,3));
    if option==0
    colormap(parula);
    tcl = tiledlayout(2,2);
    nexttile
    hold on
    imagesc(Dist(:,:,2));colorbar();clim([-0.05,0.05]);
    title('Year 2003','FontSize',20)
    set(gca,'FontSize',fs); 
    nexttile
    hold on
    imagesc(Dist(:,:,8));colorbar();clim([-0.05,0.05]);
    title('Year 2009','FontSize',20)
    set(gca,'FontSize',fs); 
    nexttile
    hold on
    imagesc(Dist(:,:,14));colorbar();clim([-0.05,0.05]);
    title('Year 2015','FontSize',20)
    set(gca,'FontSize',fs); 
    nexttile
    hold on
    imagesc(Dist(:,:,20));colorbar();clim([-0.05,0.05]);
    title('Year 2021','FontSize',20)
    set(gca,'FontSize',fs); 
    end
    F.fname=strcat('FigCTDCReal',num2str(option));
    F.wh=[8 8]*2;
        %     F.PaperPositionMode='auto';
    print_fig(gcf,F)
end

if option==2 || option==3 || option==4
    %(1,2): independent all times;  (1,3): highly related all times;
    %(1,4): increasing correlation; (1,5): decreasing correlations;
    %(1,6): decreasing and increasing; (1,7): sudden spike
    TimeMax=8;n=5000; choice=option-1;fs=24;
    switch choice
        case 1
            ymin=-0.05;ymax=0.2; titlestr='Binary Graph';
        case 2
            ymin=-0.1;ymax=1;titlestr='Normalized Correlation';
        case 3
            ymin=0;ymax=25;titlestr='Normalized Euclidean';
    end
    Data=zeros(n*TimeMax,7);
    Data(:,1)=binornd(1,0.2,n*TimeMax,1);
    Time=ones(n*TimeMax,1);
    Data(:,2)=binornd(1,0.1,n*TimeMax,1);
    Data(:,3)=binornd(1,0.6,n*TimeMax,1).*Data(:,1)+binornd(1,0.1,n*TimeMax,1).*(1-Data(:,1));
    Data(:,7)=binornd(1,0.1,n*TimeMax,1);

    for i=1:TimeMax
        Time((i-1)*n+1:i*n)=i;
        for j=1:n
            Data((i-1)*n+1:i*n,4)=binornd(1,0.1+0.1*i,n,1).*Data((i-1)*n+1:i*n,1);
            Data((i-1)*n+1:i*n,5)=binornd(1,0.9-0.1*i,n,1).*Data((i-1)*n+1:i*n,1);
            Data((i-1)*n+1:i*n,6)=binornd(1,0.1+0.2*abs(i-4),n,1).*Data((i-1)*n+1:i*n,1);
        end
    end
    Data((TimeMax-1)*n+1:TimeMax*n,7)=binornd(1,0.9,n,1).*Data((TimeMax-1)*n+1:TimeMax*n,1);
    Y=[1,2,3,4,5,6,7];

    [Dist]=GraphProcess(Data,Y,Time,choice);
    % Score(1,:,:)
    out=reshape(Dist(1,:,:),7,TimeMax);
    myColor = brewermap(18,'Spectral');
    tcl = tiledlayout(1,2);
    nexttile
    hold on
    plot(1:TimeMax,out(2,:),'color',myColor(1,:),'LineWidth',2);
    plot(1:TimeMax,out(3,:),'color',myColor(4,:),'LineWidth',2);
    plot(1:TimeMax,out(4,:),'color',myColor(7,:),'LineWidth',2);hold off
    if choice==1
    legend('(1,2) Independent','(1,3) Dependent','(1,4) Increasing','Location','NorthWest','FontSize',20)
    end
    ylim([ymin,ymax])
    xlabel('Timestep');
    xlim([1 TimeMax]);
    axis('square'); 
    set(gca,'FontSize',fs); 
    nexttile
    hold on
    plot(1:TimeMax,out(5,:),'color',myColor(12,:),'LineWidth',2);
    plot(1:TimeMax,out(6,:),'color',myColor(15,:),'LineWidth',2);
    plot(1:TimeMax,out(7,:),'color',myColor(18,:),'LineWidth',2);hold off
    if choice==1
    legend('(1,5) Decreasing','(1,6) Shifting','(1,7) Spike','Location','NorthWest','FontSize',20)
    end
    ylim([ymin,ymax])
    xlabel('Timestep');
    xlim([1 TimeMax]);
    axis('square'); 
    set(gca,'FontSize',fs);
    title(tcl,titlestr,'FontSize',30)
    F.fname=strcat('FigCTDCSim',num2str(choice));
    F.wh=[8 4]*2;
        %     F.PaperPositionMode='auto';
    print_fig(gcf,F)
end


function [Z]=GraphProcess(Data,Y,Time,choice)

if nargin<4
    choice=1;
end
TimeMax=max(Time);
K=max(Y);
% initialize graph
G=cell(TimeMax,1);
p=size(Data,2);
for i=1:TimeMax
    G{i}=zeros(p,p);
end

tic
for i=1:TimeMax
    ind=(Time==i);
    ni=sum(ind);
    if choice==1
        G{i}=Data(ind,:)'*Data(ind,:)/ni;
        tmp=diag(G{i});
        G{i}=G{i}-tmp*tmp';
    end
    if choice==2
        tmp=sum(Data(ind,:))/ni;
        for j=1:p
            for k=1:p
                G{i}(p,p)=xcorr(X, Y, 'coeff');
            end
        end
        % G{i}=1-squareform(pdist(Data(ind,:)','correlation'));
    end
    if choice==3
        G{i}=squareform(pdist(Data(ind,:)','euclidean'));
    end
    if choice==4
        G{i}=squareform(pdist(Data(ind,:)','cosine'));
    end
    for j=1:size(G{i},1)
        G{i}(j,j)=0;
    end
end
% G{1,1}
toc

opts = struct('Normalize',true,'Refine',0,'Principal',0,'Laplacian',false,'Discriminant',false,'Softmax',false);
% if choice==1
    opts.Normalize=false;
% end
[Z,out]=GraphEncoder(G,Y,opts);
Z=reshape(Z,p,K,TimeMax);
% size(Z)
% ZMean=mean(Z,3);
% ZStd=std(Z,[],3);
% size(ZMean)
% Score=zeros(p,p,TimeMax);
% for i=1:TimeMax
%     Score(:,:,i)=squareform(pdist(Z(:,:,i)','euclidean'));
% end
% ZMean=mean(Score,3)
% ZStd=std(Score,[],3);
% for i=1:TimeMax
%     Score(:,:,i)=abs(Score(:,:,i)-ZMean)/ZStd;
% end
% Score(isnan(Score))=0;
% toc
% metric='euclidean';
% ZMean=mean(Z,3);
% Dist1=squareform(pdist(ZMean,metric));
% Dist=zeros(p,p,TimeMax);
% for t=1:TimeMax
%     Dist(:,:,t)=squareform(pdist(Z(:,:,t),metric))-Dist1;
% end
% Dist(isnan(Dist))=0;
% Dist(Dist<0)=0;