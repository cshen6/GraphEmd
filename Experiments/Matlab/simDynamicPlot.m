function simDynamicPlot(optPlot)

if optPlot==1 || optPlot==2
    [Z,Dynamic,Y,time]=simDynamicSBM(101,30000,20,96);
    VD=Dynamic{1}; GD=Dynamic{3};
    %%% Figure 1: 72 simulated SBM into 3 encoder visualization panel, for
    %%% 3 community only
    if optPlot==1
        ind1=find(Y==1);ind2=find(Y==2);ind3=find(Y==3);
        myColor = brewermap(4,'RdYlGn'); myColor2 = brewermap(4,'PuOr');
        myColor=[myColor(2,:);myColor(3,:);myColor2(3,:)];
        fs=28;
        tl = tiledlayout(1,3);t1=12;t2=54;t3=96;
        nexttile(tl)
        scatter3(Z(ind1,1,t1), Z(ind1,2,t1),Z(ind1,3,t1),20,myColor(1,:),'filled');hold on
        scatter3(Z(ind2,1,t1), Z(ind2,2,t1),Z(ind2,3,t1),20,myColor(2,:),'filled');
        scatter3(Z(ind3,1,t1), Z(ind3,2,t1),Z(ind3,3,t1),20,myColor(3,:),'filled');
        hold off
        axis('square'); title('Time 12'); set(gca,'FontSize',fs); 
        nexttile(tl)
        scatter3(Z(ind1,1,t2), Z(ind1,2,t2),Z(ind1,3,t2),20,myColor(1,:),'filled');hold on
        scatter3(Z(ind2,1,t2), Z(ind2,2,t2),Z(ind2,3,t2),20,myColor(2,:),'filled');
        scatter3(Z(ind3,1,t2), Z(ind3,2,t2),Z(ind3,3,t2),20,myColor(3,:),'filled');
        axis('square'); title('Time 54'); set(gca,'FontSize',fs); 
        nexttile(tl)
        scatter3(Z(ind1,1,t3), Z(ind1,2,t3),Z(ind1,3,t3),20,myColor(1,:),'filled');hold on
        scatter3(Z(ind2,1,t3), Z(ind2,2,t3),Z(ind2,3,t3),20,myColor(2,:),'filled');
        scatter3(Z(ind3,1,t3), Z(ind3,2,t3),Z(ind3,3,t3),20,myColor(3,:),'filled');
        axis('square'); title('Time 96'); set(gca,'FontSize',fs); 
        xlabel(tl,'Embedding Visualization for the First Three Communities','FontSize',fs)
        %         set(gca,'FontSize',fs);

        F.fname='FigDynamic1';
        F.wh=[12 4]*2;
        %     F.PaperPositionMode='auto';
        print_fig(gcf,F)
    end

    %%% Figure 2: Vertex shift figure
    if optPlot==2
        fs=28;
        numBins=20;n=size(Z,1);edges=0:0.25/numBins:0.25; thres=0.25;
        myColor = brewermap(numBins,'RdPu'); % brewmap % 10 bins/colors with random r,g,b for each
        res=max(VD(:,1:12),[],2);d1 = histcounts(res,edges); mean(res>thres)
        res=max(VD(:,1:54),[],2);d2 = histcounts(res,edges); mean(res>thres)
        res=max(VD(:,1:60),[],2);d3 = histcounts(res,edges); mean(res>thres)
        res=max(VD(:,1:96),[],2);d4 = histcounts(res,edges); mean(res>thres)
        maxY=max([max(d1),max(d2),max(d3),max(d4)])*1.1;

        tl = tiledlayout(1,4);
        nexttile(tl)
        b = bar(d1, 'facecolor', 'flat'); xlim([0,20]); xticks([0 10 20]); xticklabels({'0','0.12','0.25'});ylim([0,maxY]);
        b.CData = myColor; axis('square'); set(gca,'FontSize',fs);xlabel('Vertex Dynamic','FontSize',fs)
        title('Time 12')
        nexttile(tl)
        b = bar(d2, 'facecolor', 'flat'); xlim([0,20]);xticks([0 10 20]); xticklabels({'0','0.12','0.25'});ylim([0,maxY]);
        b.CData = myColor; axis('square'); set(gca,'FontSize',fs);xlabel('Vertex Dynamic','FontSize',fs)
        title('Time 54')
%         nexttile(tl)
%         b = bar(d3, 'facecolor', 'flat'); xlim([0,20]);xticks([0 10 20]); xticklabels({'0','0.12','0.25'});ylim([0,maxY]);
%         b.CData = myColor; axis('square'); set(gca,'FontSize',fs);xlabel('Vertex Dynamic','FontSize',fs)
%         title('Time 60')
        nexttile(tl)
        b = bar(d4, 'facecolor', 'flat'); xlim([0,20]);xticks([0 10 20]); xticklabels({'0','0.12','0.25'});ylim([0,maxY]);
        b.CData = myColor; axis('square'); set(gca,'FontSize',fs); xlabel('Vertex Dynamic','FontSize',fs)
        title('Time 96')
%         title(tl,'Subplot Grid Title')

        nexttile(tl)
        myColor = brewermap(8,'Spectral'); fs=28;i=96;
        pval=zeros(24,4);
        thres=[0.25,0.10,0.05,0.02];
        for s=1:96
            for v=1:4
                pval(s,v)=mean(VD(:,s)>thres(v));
            end
        end
        hold on
        plot(1:i,pval(:,1),'Color', myColor(1,:), 'LineStyle', ':','LineWidth',4);
        plot(1:i,pval(:,2),'Color', myColor(3,:), 'LineStyle', '-.','LineWidth',4);
        plot(1:i,pval(:,3),'Color', myColor(6,:), 'LineStyle', '--','LineWidth',4);
        plot(1:i,pval(:,4),'Color', myColor(8,:), 'LineStyle', '-','LineWidth',4); hold off
        xlim([1,96]);ylim([0,0.4]);yticks([0 0.2 0.4]); yticklabels({'0','20%','40%'});
        axis('square'); 
        legend('0.25','0.10', '0.05','0.02','Location','NorthWest');
        set(gca,'FontSize',fs);
        title('Vertices Excedding Threshold'); xlabel('Time Step','FontSize',fs)
        %xlabel(tl,'Vertex Dynamic vs Time 1','FontSize',fs);
        %ylabel(tl,'Number of Vertices','FontSize',fs);
%         set(gca,'FontSize',fs);

        F.fname='FigDynamic2';
        F.wh=[16 4]*2;
        %     F.PaperPositionMode='auto';
        print_fig(gcf,F)
    end
end

%%% Figure 3: running time figure compare with unfolded ASE for increasing n and t.
if optPlot==3
    rep=10;time1=zeros(10,rep);t=3;spectral=false;time2=zeros(10,rep);
    for r=1:rep
        r
        for i=1:10
            n=5000*i;
            [Z,VD,~,tmp]=simTimeSBM(101,n,20,t);
            time1(i,r)=sum(tmp);
            if spectral==true
                [A,~]=simGenerate(101,n,20);
                %         if i<4
                A=repmat(A,1,t);
                tic
                svds(A,3);
                time2(i,r)=toc;
            end
            %         end
        end
    end
    n=5000;
    time3=zeros(10,rep);time4=zeros(10,rep);
    for r=1:rep
        r
        for i=1:10
            t=10*i;
            [Z,VD,~,tmp]=simTimeSBM(101,n,20,t);
            time3(i,r)=sum(tmp);
            if spectral==true
                [A,~]=simGenerate(101,n,20);
                A=repmat(A,1,t);
                tic
                svds(A,3);
                time4(i,r)=toc;
            end
        end
    end

    myColor = brewermap(2,'RdYlBu'); fs=28;
    tl = tiledlayout(1,2);
    nexttile(tl)
    semilogy(1:10,mean(time1,2),'Color', myColor(1,:), 'LineStyle', '-','LineWidth',5);hold on
    semilogy(1:10,mean(time2,2),'Color', myColor(2,:), 'LineStyle', ':','LineWidth',5);
    hold off
    legend('Temporal Encoder Embedding','Unfolded Spectral Embedding','Location','SouthEast'); 
    xlim([1,10]);xticks([1 5 10]); xticklabels({'5000','25000','50000'});
    xlabel('Number of Vertices','FontSize',fs); axis('square'); set(gca,'FontSize',fs);
    nexttile(tl)
    semilogy(1:10,mean(time3,2),'Color', myColor(1,:), 'LineStyle', '-','LineWidth',5); hold on
    semilogy(1:10,mean(time4,2),'Color', myColor(2,:), 'LineStyle', ':','LineWidth',5);
    hold off
    xlabel('Number of Time Steps','FontSize',fs); axis('square'); set(gca,'FontSize',fs);
    ylabel(tl,'Running Time (log scale)','FontSize',fs); xlim([1,10]);xticks([1 5 10]); xticklabels({'10','50','100'});
    %         set(gca,'FontSize',fs);

    F.fname='FigDynamic3';
    F.wh=[8 4]*2;
    %     F.PaperPositionMode='auto';
    print_fig(gcf,F)
end

if optPlot>3
    load('anonymized_msft.mat')
    [Z,Dynamic,Y,time]=GraphDynamics(G,label);
    VD=Dynamic{1}; GD=Dynamic{3}; CD=Dynamic{2};
    % Dist=Dist+Dist';
    %%% Figure 5: two vertex dynamics one for inlier one for outlier
    if optPlot==7
        %%% Figure 1: vertex dynamic figure for in + out.
        % create the plot
%         figure;
        % Create the figure
        myColor = brewermap(20,'PiYG'); fs=20;
        fig = figure;t=24;pind=[3,7];
        set(gcf, 'PaperSize', [8 4]);
%         scatter3(reshape(Z(1,1,1:t),1,t), reshape(Z(1,2,1:t),1,t),reshape(Z(1,3,1:t),1,t),20,myColor(1,:),'filled');
        for i = 1:t
            tl = tiledlayout(1,2);
            nexttile(tl)
            scatter3(reshape(Z(pind(1),1,1:i),1,i), reshape(Z(pind(1),2,1:i),1,i),reshape(Z(pind(1),3,1:i),1,i),20,myColor(1,:),'filled');
            hold on
            scatter3(reshape(Z(pind(2),1,1:i),1,i), reshape(Z(pind(2),2,1:i),1,i),reshape(Z(pind(2),3,1:i),1,i),20,myColor(20,:),'filled');
            hold off
            xlim([-1.5,1.5]);ylim([-1.5,1.5]);zlim([-1.5,1.5]);
            axis('square'); set(gca,'FontSize',fs-5); 
            nexttile(tl)
            hold on
            plot(1:i,VD(pind(1),1:i),'Color', myColor(1,:), 'LineStyle', '-.','LineWidth',2); 
            plot(1:i,VD(pind(2),1:i),'Color', myColor(20,:), 'LineStyle', '-.','LineWidth',2); hold off
            xlim([1,t]);ylim([0,1.4]);yticks([0 0.7 1.4]); xticklabels({'0','0.5','1'});
            axis('square'); set(gca,'FontSize',fs-5); 
            title(tl,strcat('Month ',{' '},num2str(i)),'FontSize',fs)
        end
    end

    if optPlot==5
        %%% Figure 1: vertex dynamic figure for in + out.
        % create the plot
%         figure;
        % Create the figure
        myColor = brewermap(8,'Spectral'); fs=28;i=24;
        fig = figure;t=24;pind=[8,7,66,29,6,41,2,25];
        set(gcf, 'PaperSize', [8 4]);
%         scatter3(reshape(Z(1,1,1:t),1,t), reshape(Z(1,2,1:t),1,t),reshape(Z(1,3,1:t),1,t),20,myColor(1,:),'filled');
            tl = tiledlayout(1,2);
            nexttile(tl)
            hold on
            plot(1:i,VD(pind(3),1:i),'Color', myColor(1,:), 'LineStyle', '-','LineWidth',4); 
            plot(1:i,VD(pind(4),1:i),'Color', myColor(2,:), 'LineStyle', '-','LineWidth',4); 
            plot(1:i,VD(pind(1),1:i),'Color', myColor(3,:), 'LineStyle', ':','LineWidth',4); 
            plot(1:i,VD(pind(2),1:i),'Color', myColor(4,:), 'LineStyle', ':','LineWidth',4); 
            plot(1:i,VD(pind(6),1:i),'Color', myColor(5,:), 'LineStyle', '--','LineWidth',4); 
            plot(1:i,VD(pind(5),1:i),'Color', myColor(6,:), 'LineStyle', '--','LineWidth',4); hold off
            xlim([1,t]);ylim([0,1]);yticks([0 0.5 1]); yticklabels({'0','0.5','1'});
            axis('square'); 
            legend('Vertex 29','Vertex 66', 'Vertex 7','Vertex 8','Vertex 6','Vertex 41','Location','NorthWest');set(gca,'FontSize',fs); 
            nexttile(tl)
            hold on
            plot(1:i,GD(1:i),'Color', myColor(1,:), 'LineStyle', '-','LineWidth',4); 
            plot(1:i,CD(8,1:i),'Color', myColor(3,:), 'LineStyle', ':','LineWidth',4); 
            plot(1:i,CD(32,1:i),'Color', myColor(3,:), 'LineStyle', ':','LineWidth',4); 
            plot(1:i,CD(1,1:i),'Color', myColor(5,:), 'LineStyle', '-.','LineWidth',4); 
            plot(1:i,CD(4,1:i),'Color', myColor(5,:), 'LineStyle', '-.','LineWidth',4);
            plot(1:i,CD(3,1:i),'Color', myColor(7,:), 'LineStyle', '--','LineWidth',4); 
            plot(1:i,CD(39,1:i),'Color', myColor(7,:), 'LineStyle', '--','LineWidth',4); 
            hold off
            xlim([1,t]);ylim([0,0.5]);yticks([0 0.25 0.5]); yticklabels({'0','0.25','0.5'});
            axis('square'); 
            legend('Graph Dynamic','Community 8','Community 32', 'Community 1','Community 4','Community 3', 'Community 39','Location','NorthWest');set(gca,'FontSize',fs); 
%             nexttile(tl)
%             hold on
%             pval=zeros(24,4);
%             thres=[0.25,0.5,0.75,0.99];
%             for s=1:24
%                 for v=1:4
%                    pval(s,v)=mean(Z_d(:,s)>thres(v));
%                 end
%             end
%             plot(1:i,pval(:,1),'Color', myColor(1,:), 'LineStyle', '-','LineWidth',4); 
%             plot(1:i,pval(:,2),'Color', myColor(3,:), 'LineStyle', ':','LineWidth',4); 
%             plot(1:i,pval(:,3),'Color', myColor(6,:), 'LineStyle', '-.','LineWidth',4); 
%             plot(1:i,pval(:,4),'Color', myColor(8,:), 'LineStyle', '--','LineWidth',4); hold off
%             xlim([1,t]);ylim([0,1]);yticks([0 0.5 1]); yticklabels({'0','50%','100%'});
%             axis('square'); set(gca,'FontSize',fs); 
%             legend('0.25','0.5', '0.75','0.99','Location','NorthWest');
            %ylabel('Vertex Dynamic','FontSize',fs);
            %xlabel(tl,'Month','FontSize',fs);
            title(tl,'Temporal Dynamic from Month 1 to Month 24','FontSize',fs)
            F.fname='FigDynamic5';
            F.wh=[8 4]*2;
            %     F.PaperPositionMode='auto';
            print_fig(gcf,F)

%         fs=24;
%         subplot(1,2,1)
%         hold on
%         plot(1:24,Z_d(2,:),'b-.','LineWidth',2);
%         plot(1:24,Z_d(3,:),'g-.','LineWidth',2);
%         plot(1:24,Z_d(4,:),'r-.','LineWidth',2); % no action since month 2!
%         plot(1:24,Z_d(5,:),'c-.','LineWidth',2);
%         plot(1:24,Z_d(6,:),'m-.','LineWidth',2);
%         hold off
%         ylim([-0.5,1]);
%         xlim([1,24]);
%         xlabel('Month')
%         legend('Vertex 1','Vertex 2','Vertex 3','Vertex 4','Vertex 5','Location','NorthWest')
%         title('Vertex Dynamics')
%         set(gca,'FontSize',fs);
%         axis('square');
%         subplot(1,2,2)
%         hold on
%         plot(1:24,Z_d(2,:),'b-.','LineWidth',2);
%         plot(1:24,Z_d(3,:),'g-.','LineWidth',2);
%         plot(1:24,Z_d(4,:),'r-.','LineWidth',2);
%         plot(1:24,Z_d(5,:),'c-.','LineWidth',2);
%         plot(1:24,Z_d(6,:),'m-.','LineWidth',2);
%         hold off
%         ylim([-0.5,1]);
%         xlim([1,24]);
%         xlabel('Month')
%         legend('Vertex 1','Vertex 2','Vertex 3','Vertex 4','Vertex 5','Location','NorthWest')
%         title('Vertex Dynamics')
%         set(gca,'FontSize',fs);
%         axis('square');
%         F.fname='MSFT1';
%         F.wh=[8 4]*2;
%         %     F.PaperPositionMode='auto';
%         print_fig(gcf,F)
        % xticks([1,13,24])
        % Gy=mdscale(Dist,2);
        % plot(Gy(:,1),Gy(:,2));
        %outlier
    end
    %%%Figure 6: vertex shifts as time goes.

    if optPlot==6;
        fs=28;
        numBins=20;n=size(Z,1);edges=0:1/numBins:1;thres=0.5;
        myColor = brewermap(numBins,'RdPu'); % brewmap % 10 bins/colors with random r,g,b for each
%         res=max(Z_d(:,1:2),[],2);d1 = histcounts(res,edges); 
%         res=max(Z_d(:,1:4),[],2);d2 = histcounts(res,edges); 
%         res=max(Z_d(:,1:6),[],2);d3 = histcounts(res,edges); %mean(res>thres)
%         res=max(Z_d(:,1:12),[],2);d4 = histcounts(res,edges); %mean(res>thres)
%         res=max(Z_d(:,1:18),[],2);d5 = histcounts(res,edges); %mean(res>thres)
%         res=max(Z_d(:,1:24),[],2);d6 = histcounts(res,edges); %mean(res>thres)
        res=VD(:,6);d3 = histcounts(res,edges); %mean(res>thres)
        res=VD(:,12);d4 = histcounts(res,edges); %mean(res>thres)
        res=VD(:,18);d5 = histcounts(res,edges); %mean(res>thres)
        res=VD(:,24);d6 = histcounts(res,edges); %mean(res>thres)
        per=[mean(VD(:,6)>thres),mean(VD(:,12)>thres),mean(VD(:,18)>thres),mean(VD(:,24)>thres)];
        maxY=max([max(d4),max(d3),max(d5),max(d6)])*1.1;

        tl = tiledlayout(1,4);
%         nexttile(tl)
%         b = bar(d3, 'facecolor', 'flat'); xlim([0,20]); xticks([0 10 20]); xticklabels({'0','50%','100%'});ylim([0,maxY]);
%         b.CData = myColor; axis('square'); set(gca,'FontSize',fs);
%         title('Month 6')
        nexttile(tl)
        b = bar(d3, 'facecolor', 'flat'); xlim([0,20]); xticks([0 10 20]); xticklabels({'0','0.5','1'});ylim([0,maxY]);
        b.CData = myColor; axis('square'); 
        title('Month 6'); set(gca,'FontSize',fs);
        nexttile(tl)
        b = bar(d4, 'facecolor', 'flat'); xlim([0,20]); xticks([0 10 20]); xticklabels({'0','0.5','1'});ylim([0,maxY]);
        b.CData = myColor; axis('square');
        title('Month 12'); set(gca,'FontSize',fs);
        nexttile(tl)
        b = bar(d5, 'facecolor', 'flat'); xlim([0,20]); xticks([0 10 20]); xticklabels({'0','0.5','1'});ylim([0,maxY]);
        b.CData = myColor; axis('square'); 
        title('Month 18'); set(gca,'FontSize',fs);
        nexttile(tl)
        b = bar(d6, 'facecolor', 'flat'); xlim([0,20]); xticks([0 10 20]); xticklabels({'0','0.5','1'});ylim([0,maxY]);
        b.CData = myColor; axis('square'); 
        title('Month 24'); set(gca,'FontSize',fs);
%         title(tl,'Subplot Grid Title')

        xlabel(tl,'Vertex Dynamic vs Month 1','FontSize',fs);
        ylabel(tl,'Number of Vertices','FontSize',fs);
%         set(gca,'FontSize',fs);

        F.fname='FigDynamic6';
        F.wh=[16 4]*2;
        %     F.PaperPositionMode='auto';
        print_fig(gcf,F)
% 
% 
%         [~,indOut]=sort(res,'descend');
%         hist(res);
%         mean(res>thres)
%         xlim([0,1.414]);
%         title("Vertex Shifts from 2019.12 to 2020.1");
%         set(gca,'FontSize',fs);
%         set(gca,'YTickLabel',[])
%         axis('square');
%         subplot(1,3,2)
%         i=12;j=18;
%         Z1=Z(:,:,i);
%         Z2=Z(:,:,j);
%         res=vecnorm(Z1-Z2,2,2);
%         [~,indOut]=sort(res,'descend');
%         hist(res);
%         mean(res>thres)
%         xlim([0,1.414]);
%         title("Vertex Shifts from 2019.12 to 2020.6");
%         set(gca,'FontSize',fs);
%         set(gca,'YTickLabel',[])
%         axis('square');
%         subplot(1,3,3)
%         i=12;j=24;
%         Z1=Z(:,:,i);
%         Z2=Z(:,:,j);
%         res=vecnorm(Z1-Z2,2,2);
%         [~,indOut]=sort(res,'descend');
%         hist(res);
%         mean(res>thres)
%         xlim([0,1.414]);
%         title("Vertex Shifts from 2019.12 to 2020.12");
%         set(gca,'FontSize',fs);
%         set(gca,'YTickLabel',[])
%         axis('square');
%         F.fname='MSFT2';
%         F.wh=[12 4]*2;
%         %     F.PaperPositionMode='auto';
%         print_fig(gcf,F)
    end
end