function [Z,VD,Y,time]=simDynamicSBM(type,n,K,t)

Lapl=false;
USE=false;
normalize=false;
extreme=false;
opts=1;
%type=101; n=1000; K=20;t=10;
if opts==1
    [A,Y]=simGenerate(type,n,K,0);
    E=adj2edge(A);
%     E=A;
    E(:,3)=randi(100,size(E,1),1);
    G={E};s=size(E,1);
%     if USE == true;
%         A=edge2adj(E);
        if Lapl==true
            %         A=laplacian(graph(A));
            A=Lap(A);
        end
%     end
    for i=1:t-1
        inlier=binornd(1,0.5,s,1);
        outlier=(~inlier);
        E(outlier,3)=E(outlier,3)+randi([-20,20],sum(outlier),1);
        if extreme==true && i==t-1
            extreme=1:9;
            E(extreme,3)=randi([500,1000],length(extreme),1);
            extreme=[E(extreme,1);E(extreme,2)];
            extreme=unique(extreme);
        end
        E(E<1)=1;
        G=[G,E];
        if USE == true;
            if Lapl==true
%                 A=[A,laplacian(graph(edge2adj(E)))];
                A=[A,Lap(edge2adj(E))];
            else
                A=[A,edge2adj(E)];
            end
        end
    end
end
% if opts==2
%     t=2;
%     [A1,Y]=simGenerate(11,1000,20);
%     [A2,Y]=simGenerate(11,1000,20);
%     E1=adj2edge(A1); E2=adj2edge(A2);
%     E(:,3)=randi(100,size(E,1),1);
%     A=[A1,A2];
%     G={E1,E2};
% end
opt=struct('Normalize',normalize);
[Z,VD,Y,time]=GraphDynamics(G,Y,opt);
[~,ind1]=sort(VD{1},'descend');
if USE == true;
    d=10;fs=28;
    [~,S,V]=svds(A,d);
    Z2=S^0.5*V';
    Z2=reshape(Z2,d,n,t);
    %     for i=1:10;
    %         ZNorm=vecnorm(Z2(:,:,i),1,1);
    %         Z2(:,:,i)=Z2(:,:,i)./repmat(ZNorm,d,1);
    %     end
    t_b=1;
    VDZ=zeros(n,t-t_b+1);
    %     VDZ2=zeros(n,t-t_b+1);
    for i=2:t-t_b+1
        VDZ(:,t_b+i-1)=vecnorm(Z2(:,:,t_b+i-1)-Z2(:,:,t_b),2,1);
    end
    %     for i=1:10
    %         ZNorm=vecnorm(Z2(:,:,i),1,1);
    %         Z2(:,:,i)=Z2(:,:,i)./repmat(ZNorm,d,1);
    %     end
    %     for i=2:t-t_b+1
    %         VDZ2(:,t_b+i-1)=1-dot(Z2(:,:,t_b+i-1),Z2(:,:,t_b),1);
    %     end
    [~,ind2]=sort(VDZ,'descend');
    norm(ind1-ind2);
    a=zeros(length(extreme),1);
    b=zeros(length(extreme),1);
    for i=1:length(extreme);
        a(i)=find(ind1(:,10)==extreme(i));
        if USE == true;
            b(i)=find(ind2(:,10)==extreme(i));
        end
    end
    [~,ind1]=sort(a,'descend');
    [~,ind2]=sort(b,'descend');

    myColor = brewermap(100,'RdPu');
    tl = tiledlayout(2,2);
    nexttile(tl)
    hist(VD{1}(:,t),20)
    xlim([0,0.5]);xticks([0 0.5]);
    title('Encoder * Vertex Dynamic');
    axis('square'); set(gca,'FontSize',fs); 
    nexttile(tl)
    hist(VDZ(:,t),20)
    xlim([0,0.5]);xticks([0 0.5]);
    title('USE * Euclidean Distance');
    axis('square'); set(gca,'FontSize',fs); 
    nexttile(tl)
    hold on
    %rk=VD{1}(extreme(ind1),10);
    rk=floor((1-(a-1)/n)*100)/100;
    h=barh(VD{1}(extreme(ind1),10), 'FaceColor', 'flat');
    for i=1:length(extreme)
        h.CData(i,:) = myColor(round(rk(ind1(i))*100),:);
        text(VD{1}(extreme(ind1(i)),10)+0.01, i, sprintf('%0.5g', rk(ind1(i))), 'Color', myColor(60,:), 'FontSize', fs, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
    end
    hold off
    xlim([0,0.5]); xticks([0 0.5]);%xticklabels({'95%','100%'});
    yticks([1 length(extreme)]); yticklabels({num2str(length(extreme)),'1'});
    title('Outlier Percentile Ranking');
    axis('square'); set(gca,'FontSize',fs); 
    nexttile(tl)
    %rk=VDZ(extreme(ind2),10);
    hold on
    rk=floor((1-(b-1)/n)*100)/100;
    h=barh(VDZ(extreme(ind2),10), 'FaceColor', 'flat');
    for i=1:length(extreme)
        h.CData(i,:) = myColor(round(rk(ind2(i))*100),:);
        text(VDZ(extreme(ind2(i)),10)+0.01, i, sprintf('%0.5g', rk(ind2(i))), 'Color', myColor(60,:), 'FontSize', fs, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
    end
    hold off
    xlim([0,0.5]); xticks([0 0.5]);%xticklabels({'95%','100%'});
    yticks([1 length(extreme)]); yticklabels({num2str(length(extreme)),'1'});
    title('Outlier Percentile Ranking');
    axis('square'); set(gca,'FontSize',fs); 
    % nexttile(tl)
    % hist(VDZ2(:,10))
    % % xlim([0,0.6]);
    % title('USE * Correlation')

    F.fname='FigDynamic7';
    F.wh=[8 8]*2;
    %     F.PaperPositionMode='auto';
    print_fig(gcf,F)
end
