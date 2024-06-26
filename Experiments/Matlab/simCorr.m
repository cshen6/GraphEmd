function simCorr(choice)

if nargin<1
    choice=1;
end
opts = struct('Normalize',0,'Unbiased',0,'DiagAugment',0,'Principal',0,'Laplacian',0,'Discriminant',0);
if choice<20
    reps=1000;
    spec=true;
    K=10;
    type=600+(choice);
    if choice>10
        type=type-10;
    end
    nn=100;lim=10;dd=20;
    if choice ==5 || choice == 6 || choice ==7 || choice ==8
        nn=200;
    end
    % if choice>3
    %     nn=200;
    % end
    power0=zeros(lim,1);
    power1=zeros(K,K,lim);
    power2=zeros(lim,5);
    time1=zeros(lim,reps);
    time2=zeros(lim,reps,4);
    alpha=0.05;
    stat=zeros(lim,reps);
    corrCom=zeros(K,K,reps,lim);
    for l=1:lim
        n=nn*l
        for r=1:reps
            % r
            [Dis,Y]=simGenerate(type,n,K,0);
            A=Dis{1}; B=Dis{2};
            % A=rand(n,1);B=rand(n,1);
            % A=squareform(pdist(A));B=squareform(pdist(B));Y=randi(K,n,1);
            % [B,~]=simGenerate(21,n,K,0);
            tic
            try
                [stat(l,r),pval,corrCom(:,:,r,l),pvalCom]=CorrGEE(A,B,Y);
            catch
                r=r-1;
                continue;
            end
            time1(l,r)=toc;
            if spec
                for jj=1:2
                    tic
                    if jj==1
                        Z1=ASE(A,dd);Z2=ASE(B,dd);
                    else
                        if size(Y,2)==2
                            Y1=Y(:,1);Y2=Y(:,2);
                        else
                            Y1=Y;Y2=Y;
                        end
                        Z1=GraphEncoder(A,Y1,opts);Z2=GraphEncoder(B,Y2,opts);
                    end
                    tmp=toc;
                    tic
                    [~,pval2]=DCorFastTest(Z1,Z2);
                    % [~,pval2]=CorrGEE(A,B,K,1);
                    time2(l,r,1+(jj-1)*2)=toc+tmp;
                    power2(l,1+(jj-1)*2)=power2(l,1+(jj-1)*2)+(pval2<alpha)/reps;
                    tic
                    [~,pval2]=DCorPartialTest(Z1,Z2,Y);
                    % [~,pval3]=CorrGEE(A,B,K,10);
                    time2(l,r,2+(jj-1)*2)=toc+tmp;
                    power2(l,2+(jj-1)*2)=power2(l,2+(jj-1)*2)+(pval2<alpha)/reps;
                end
            end
            power0(l)=power0(l)+(pval<alpha)/reps;
            power1(:,:,l)=power1(:,:,l)+(pvalCom<alpha)/reps;
            [~,pval]=CorrGEE(A,B);
            power2(l,5)=power2(l,5)+(pval<alpha)/reps;
        end
    end

    save(strcat('CorrGEESim',num2str(choice),'.mat'),'nn','lim','type','reps','power0','power1','power2','n','stat','corrCom','time1','time2');
end


if choice>20 && choice<30
    switch choice
        case 21
            load('CElegans.mat');G1=Ac;G2=Ag;Y=vcols;
        % case 12
        %     load('anonymized_msft.mat');G1=edge2adj(G{1});G2=edge2adj(G{3});Y=label;
        case 23
            load('Wiki_Data.mat'); G1=GEAdj;G2=GFAdj;Y=Label;
        case 24
            load('Letter.mat'); G1=edge2adj(Edge1); G2=edge2adj(Edge3);Y=Label1;n=10507;G2=G2(1:n,1:n);
        case 25
            load('Letter.mat'); G1=edge2adj(Edge1); G2=edge2adj(Edge3);Y=Label1;n=10507;G2=G2(1:n,1:n);
        case 26
            load('Wiki_Data.mat'); G1=max(max(TE))-TE;G2=GEAdj;Y=Label;
        case 27
            load('Wiki_Data.mat'); G1=max(max(TF))-TF;G2=GFAdj;Y=Label;
        case 28
            load('Wiki_Data.mat'); G1=max(max(TE))-TE;G2=max(max(TF))-TF;Y=Label;
    end
    [stat,pval,corrCom,pvalCom]=CorrGEE(G1,G2,Y);
    % d=10;
    % Z1=ASE(G1,d);Z2=ASE(G2,d);
    % [stat2,pval2]=DCorFastTest(Z1,Z2);
    %  [stat3,pval3]=DCorPartialTest(Z1,Z2,Y);
     save(strcat('CorrGEESim',num2str(choice),'.mat'),'pval','pvalCom','corrCom','pvalCom','stat');

    fs=15;lw=3;
    if choice==23
    txt={'category', 'people', 'locations', 'date', 'math'};
     h=heatmap(txt,txt,corrCom);
     h.YDisplayData = txt;
    else
        h=heatmap(corrCom);
    end
        clim([0, 0.1]);
        set(gca,'FontSize',fs);
        %ylabel('Running Time')
        %title('Running Time'
        %legend('Permutation','Fast Test','Zhang Subsampling','T-Test','Location','SouthEast')
    title('Community Correlations');
    F.fname='GraphCorFig7'; %strcat(pre2, num2str(i));
    F.wh=[6 6];
    print_fig(gcf,F)
end

if choice ==101 || choice==102%null distribution 
  %  corrChi=zeros(reps,1);
  K=10;reps=100;lw=3;fs=20;
    corrNor=normrnd(0,1,reps*100,1);
    corrNor2=normrnd(0,2,reps*100,1);
%    (chi2rnd(ones(reps,1))-1)/n*100;
 %   corrMax=(chi2rnd(ones(reps,1))-1)/n;

    t = tiledlayout(2,3);
    for i=1:6
        switch i
            case 1
                type=601;n=100;tstr='s=1K';
            case 2
                type=601;n=300;tstr='s=10K';
            case 3
                type=601;n=1000;tstr='s=100K';
            case 4
                type=602;n=100;tstr='n=100';
            case 5
                type=602;n=300;tstr='n=500';
            case 6
                type=602;n=1000;tstr='n=1000';
        end
        % A=rand(n,1);B=rand(n,1);
        % A=squareform(pdist(A));B=squareform(pdist(B));Y=randi(K,n,1);
        % [B,~]=simGenerate(21,n,K,0);
        corrCom=zeros(K,K,reps);
        for r=1:reps
            [Dis,Y]=simGenerate(type+(choice-101)*6,n,K,0);
            A=Dis{1}; B=Dis{2};
            [stat(r),~,corrCom(:,:,r),~,nk]=CorrGEE(A,B,Y);
            corrCom(:,:,r)=sqrt(nk).*corrCom(:,:,r);
        end
        corrCom=reshape(corrCom,1,size(corrCom,1)*size(corrCom,2)*size(corrCom,3));

        nexttile();
        hold on
        [a,b]=ecdf(corrCom);plot(b,a,'linewidth',lw)
        [a,b]=ecdf(corrNor);plot(b,a,'--','linewidth',lw)
        [a,b]=ecdf(corrNor2);plot(b,a,'--','linewidth',lw)
        ylim([0.8,1]);
        %legend('Null CDF by Permutation','Null CDF by Fast Approximation','Location','SouthEast')
        %xlim([0,0.1]);
        %ylim([0.8,1.01]);
        yticks([0.8,0.9,1])
        if i==1
        legend('Actual Null','Normal(0,1)','Normal(0,2)','Location','SouthEast');
        ylabel('Matched Label')
        end
        if i==4;
            ylabel('Un-matched Label')
        end
        if i>3
            xlabel(tstr);
        end
        hold off
        axis('square');
        set(gca,'FontSize',fs);
    end
    title(t,'Cumulative Distribution Function','FontSize',28);

    if choice==101
        F.fname='GraphCorFig1'; %strcat(pre2, num2str(i));
    else
        F.fname='GraphCorFig2'; %strcat(pre2, num2str(i));
    end
F.wh=[12 8];
print_fig(gcf,F)
end

if choice==103 % running time
    t = tiledlayout(1,3);fs=20;lw=3;
    for i=1:3
        switch i
            case 1
                type=1;n=100;tstr='SBM 1';
            case 2
                type=2;n=500;tstr='SBM 2';
            case 3
                type=3;n=1000;tstr='SBM 3';
        end
        nexttile();
        load(strcat('CorrGEESim',num2str(i),'.mat'));
        semilogy(1:lim,mean(time1,2),1:lim,mean(time2(:,:,1),2),':',1:lim,mean(time2(:,:,2),2),'--','linewidth',lw)
        xlim([1,lim])
        xticks([1,3,5,10])
        xticklabels({'1K','10K','30K','100K'})
        ylim([0.001,0.5])
        yticks([0.01,0.1])
        % ylim([0,1])
        xlabel('Number of Edges')
        title(tstr);
        set(gca,'FontSize',fs);
        %ylabel('Running Time')
        %title('Running Time')
        if i==1
            legend('CorrGEE','DCor','DCorPartial','Location','NorthWest')
        end
        %legend('Permutation','Fast Test','Zhang Subsampling','T-Test','Location','SouthEast')
        axis('square');
    end
    title(t,'Running Time','FontSize',28);
    F.fname='GraphCorFig3'; %strcat(pre2, num2str(i));
    F.wh=[12 6];
    print_fig(gcf,F)
end

if choice==104 % testing power and community correlation in SBM
    t = tiledlayout(2,3);fs=13;lw=3;
    myColor = brewermap(10,'RdYlBu');
    myColor2 = brewermap(10,'Spectral');
    myColor3 = brewermap(4,'PuOr');
    for i=1:6
        switch i
            case 1
                type=601;n=100;tstr='Conditional Independence';
            case 2
                type=601;n=500;tstr='Unconditional Independence';
            case 3
                type=601;n=1000;tstr='All Communities Dependence';
            case 4
                type=601;n=1000;tstr='Community (1,.) Dependence';
            case 5
                type=601;n=1000;tstr='Community (1,2) Dependence';
            case 6
                type=601;n=1000;tstr='(1,2) & (5,10) Dependence';
        end
        nexttile();
        load(strcat('CorrGEESim',num2str(i),'.mat'));
        hold on
        plot(1:lim,power0,'Color', myColor(9,:),'LineStyle', '-','linewidth',lw)
        plot(1:lim,power2(:,1),'Color', myColor(1,:),'LineStyle', ':','linewidth',lw)
        plot(1:lim,power2(:,2),'Color', myColor(3,:),'LineStyle', ':','linewidth',lw)
        plot(1:lim,power2(:,3),'Color', myColor3(2,:),'LineStyle', '--','linewidth',lw)
        plot(1:lim,power2(:,4),'Color', myColor3(3,:),'LineStyle', '--','linewidth',lw)
        ylim([0,1]);
        xlim([1,lim])
        xticks([1,lim])
        xticklabels([100,lim*100])
        if i>4
            xticklabels([200,lim*200])
        end
        % ylim([0.001,0.5])
        % yticks([0.01,0.1])
        % ylim([0,1])
        xlabel('Sample Size')
        title(tstr);
        set(gca,'FontSize',fs);
        %ylabel('Running Time')
        %title('Running Time')
        if i==2
            legend('CorrGEE','DCor*Spec','DCorP*Spec','DCor*GEE','DCorP*GEE','Location','NorthWest')
        end
        %legend('Permutation','Fast Test','Zhang Subsampling','T-Test','Location','SouthEast')
        axis('square');
    end
    title(t,'Testing Power','FontSize',28);
    F.fname='GraphCorFig4'; %strcat(pre2, num2str(i));
    F.wh=[12 8];
    print_fig(gcf,F)
end

if choice==105 % testing power and community correlation in SBM
    t = tiledlayout(2,2);fs=13;lw=3;
    myColor = brewermap(10,'RdYlBu');
    myColor2 = brewermap(10,'Spectral');
    myColor3 = brewermap(4,'PuOr');
    for i=1:4
        switch i
            case 1
                type=601;n=100;tstr='Conditional Independence';
            case 2
                type=601;n=500;tstr='Unconditional Independence';
            case 3
                type=601;n=1000;tstr='All Communities Dependence';
            case 4
                type=601;n=1000;tstr='Degree Dependence';
        end
        nexttile();
        load(strcat('CorrGEESim',num2str(i+6),'.mat'));
        hold on
        plot(1:lim,power0,'Color', myColor(9,:),'LineStyle', '-','linewidth',lw)
        plot(1:lim,power2(:,1),'Color', myColor(1,:),'LineStyle', ':','linewidth',lw)
        plot(1:lim,power2(:,2),'Color', myColor(3,:),'LineStyle', ':','linewidth',lw)
        plot(1:lim,power2(:,3),'Color', myColor3(2,:),'LineStyle', '--','linewidth',lw)
        plot(1:lim,power2(:,4),'Color', myColor3(3,:),'LineStyle', '--','linewidth',lw)
        ylim([0,1]);
        xlim([1,lim])
        xticks([1,lim])
        xticklabels([100,lim*100])
        if i<3
            xticklabels([200,lim*200])
        end
        % ylim([0.001,0.5])
        % yticks([0.01,0.1])
        % ylim([0,1])
        xlabel('Sample Size')
        title(tstr);
        set(gca,'FontSize',fs);
        %ylabel('Running Time')
        %title('Running Time')
        if i==2
            legend('CorrGEE','DCor*Spec','DCorP*Spec','DCor*GEE','DCorP*GEE','Location','NorthWest')
        end
        %legend('Permutation','Fast Test','Zhang Subsampling','T-Test','Location','SouthEast')
        axis('square');
    end
    title(t,'Testing Power','FontSize',23);
    F.fname='GraphCorFig5'; %strcat(pre2, num2str(i));
    F.wh=[8 8];
    print_fig(gcf,F)
end

if choice==106 % testing power and community correlation in SBM
    t = tiledlayout(2,2);fs=13;lw=3;
    myColor = brewermap(10,'RdYlBu');
    myColor2 = brewermap(10,'Spectral');
    myColor3 = brewermap(4,'PuOr');
    for i=1:4
        switch i
            case 1
                type=3;n=100;tstr='All Communities Dependence';
            case 2
                type=4;n=500;tstr='Community (1,.) Dependence';
            case 3
                type=6;n=1000;tstr='(1,2)+(5,10) Dependence';
            case 4
                type=9;n=1000;tstr='Degree Dependence';
        end
        nexttile();
        load(strcat('CorrGEESim',num2str(type),'.mat'));
        local=mean(corrCom(:,:,:,end),3);
        heatmap(local);
        clim([0, 0.1]);
        title(tstr);
        set(gca,'FontSize',fs);
        %ylabel('Running Time')
        %title('Running Time'
        %legend('Permutation','Fast Test','Zhang Subsampling','T-Test','Location','SouthEast')
    end
    title(t,'Community Correlations','FontSize',23);
    F.fname='GraphCorFig6'; %strcat(pre2, num2str(i));
    F.wh=[8 8];
    print_fig(gcf,F)
end