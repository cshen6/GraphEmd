function simTimePlot(opt)

if opts==1 || opts==2
    %%% Figure 1: 72 simulated SBM into 3 encoder visualization panel,
    [Z,Z_d,Y,time,ind]=simTimeSBM(101,100,20,96);
    res1=max(Z_d,[],2);hist(res1);
end

%%% Figure 3: running time figure compare with unfolded ASE for increasing n and t.
if opts==3
    subplot(1,2,1)
    n=1000;
    time=zeros(10,3);
    for i=1:10;
        t=10*i
        [Z,Z_d,Y,time(i,:),ind]=simTimeSBM(101,n,20,t);
    end
end

if opts>3
    load('anonymized_msft.mat')
    [Z,Z_d,ind,Y,time]=GraphDynamics(G,label);
    % Dist=Dist+Dist';
    %%% Figure 5: two vertex dynamics one for inlier one for outlier
    if opts==5;
        fs=24;
        subplot(1,2,1)
        hold on
        plot(1:24,Z_d(2,:),'b-.','LineWidth',2);
        plot(1:24,Z_d(3,:),'g-.','LineWidth',2);
        plot(1:24,Z_d(4,:),'r-.','LineWidth',2); % no action since month 2!
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
        subplot(1,2,2)
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
        F.fname='MSFT1';
        F.wh=[8 4]*2;
        %     F.PaperPositionMode='auto';
        print_fig(gcf,F)
        % xticks([1,13,24])
        % Gy=mdscale(Dist,2);
        % plot(Gy(:,1),Gy(:,2));
        %outlier
    end
    %%%Figure 6: vertex shifts as time goes.

    if opts==6;
        subplot(1,3,1)
        i=12;j=13;
        Z1=Z(:,:,i);
        Z2=Z(:,:,j);
        % res=vecnorm(Z1-Z2,2,2);
        res1=max(Z_d,[],2);hist(res2);
        res2=max(Z_d(:,1:6),[],2); hist(res2);
        res3=max(Z_d(:,1:12),[],2); hist(res3);
        res4=max(Z_d(:,1:18),[],2); hist(res4);




        [~,indOut]=sort(res,'descend');
        hist(res);
        mean(res>thres)
        xlim([0,1.414]);
        title("Vertex Shifts from 2019.12 to 2020.1");
        set(gca,'FontSize',fs);
        set(gca,'YTickLabel',[])
        axis('square');
        subplot(1,3,2)
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
        subplot(1,3,3)
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
        F.fname='MSFT2';
        F.wh=[12 4]*2;
        %     F.PaperPositionMode='auto';
        print_fig(gcf,F)
    end
end