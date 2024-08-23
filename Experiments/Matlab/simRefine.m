function simRefine(choice,rep,cvf,spectral,n2v)

if nargin<2
rep=3;
end
if nargin<3
cvf=10;
end
if nargin<4
    spectral=0;
end
if nargin<5
    n2v=0;
end
% thres=0.98;
% if choice==1
%     load('Wiki_Data.mat');
%     opts = struct('Normalize',true,'Refine',10,'Principal',0,'Laplacian',false,'Discriminant',true,'Softmax',true);
%     [Z,out]=RefinedGEE(GEAdj,Y,opts);
%     [~,YVal]=max(Z,[],2);
%     YVal=out.dimClass(YVal);
%     mean(YVal~=Y)
% 
%     [X,Y]=simGenerate(400,300,3,0); Y2=Y;Y2(Y2==2)=1;Y2(Y2==3)=2;
%     opts = struct('Normalize',true,'Refine',10,'Principal',0,'Laplacian',false,'Discriminant',true,'Softmax',true);
%     [Z,out]=RefinedGEE(X,Y2,opts);
%     idx=out.idx;
%     sum(idx)
%     sum(idx2)
% 
%     Y3=kmeans(Z(Y2==1,:),2);
%     idx=(Y2==1);
%     Y2(idx)=Y3;
%     [Z,out]=RefinedGEE(X,Y2,opts);
%     [ZMax,YVal]=max(Z,[],2);
%     YVal=out.dimClass(YVal);
%     idx=(YVal~=Y);
%     idx2=(ZMax<thres) & idx;
%     sum(idx)
%     sum(idx2)
% end

if choice>=10 && choice <30
    rng("default")
    switch choice
        case 10
            load('adjnoun.mat'); X=Adj;n2vstr='AdjNoun';
        case 11
            load('citeseer.mat');X=edge2adj(Edge);Y=Label;n2vstr='Citeseer';
        case 12
            load('Cora.mat');X=edge2adj(Edge);Y=Label;n2vstr='Cora'; %unmatched
        case 13
            load('Coil-Rag.mat');X=edge2adj(Edge);Y=Label;n2vstr='Coil';%unmatched
        case 14 
            load('karate.mat'); X=G;n2vstr='karate';
        case 15
            load('IIP.mat'); X=double(Adj+Adj'>0);n2vstr='IIP';
        case 16
            load('letter.mat'); X=edge2adj(Edge1);Y=Label1;LeidenY=LeidenY1;n2vstr='letter1';%unmatched
        case 17
            load('polblogs.mat');X=Adj;n2vstr='polblogs';
        case 18
            load('pubmed.mat');X=edge2adj(Edge);Y=Label;n2vstr='pubmed';%unmatched
        case 19
            load('soc-political-retweet.mat'); X=edge2adj(Edge);Y=Label;n2vstr='poltweet'; 
        case 20
            load('CElegans.mat');X=double(Ac+Ac'>0);Y=vcols;LeidenY=AcLeidenY;n2vstr='CElegansAc';
        case 21
            load('CElegans.mat');X=double(Ag+Ag'>0);Y=vcols;LeidenY=AgLeidenY;n2vstr='CElegansAg';
        case 22 %RGEE better
            load('email.mat'); X=Adj;n2vstr='email';
        case 23
            load('Gene.mat'); X=double(AdjOri+AdjOri'>0);n2vstr='Gene';
        case 24 %RGEE better
            load('lastfm.mat'); X=Adj;n2vstr='lastfm';
        case 25 % improve
            load('Wiki_Data.mat'); X=TE;Y=Label;LeidenY=TELeidenY;n2v=0;
        case 26 % improve
            load('Wiki_Data.mat'); X=TF;Y=Label;LeidenY=TFLeidenY;n2v=0;
        case 27
            load('Wiki_Data.mat'); X=GEAdj;Y=Label;LeidenY=GELeidenY;n2vstr='WikiGE';
        case 28
            load('Wiki_Data.mat'); X=GFAdj;Y=Label;LeidenY=GFLeidenY;n2vstr='WikiGF';%unmatched
    end
    %RefineEvaluate(X,Y);
    K=max(Y);
    ind=(sum(X)>0);%length(Y)-sum(ind)
    X=X(ind,ind);Y=Y(ind);
    if spectral==1
        tic
        %[Z]=UnsupGraph(X,max(Y)*5,length(Y));
        dim=20;
        [ZASE]=ASE(X,dim);
        tt1=toc;
        tic
        % ZLSE = node2vec(X+X', 128, 10, 80, 1, 1);
        [ZLSE]=ASE(Lap(X),dim);
        % ZLSE=ZNV;
        % ZLeiden=GraphEncoder(X,LeidenY);
        tt2=toc;
    end
    if n2v==1
        ZNV=load('n2v.mat',n2vstr);
        ZNV=ZNV.(n2vstr);
    end
    % tic
    % [Z2]=ASE(X,10,true);
    % tt2=toc;
    error1=zeros(rep,7);error2=zeros(rep,4);time1=zeros(rep,7);time2=zeros(rep,4);
    for r=1:rep
        indices=crossvalind('Kfold',Y,cvf); 
        if spectral==1
        tmp=AttributeEvaluate({ZASE,X},Y,indices); %K=6
        error1(r,1:3)=tmp(1,:);time1(r,1:3)=tmp(2,:)+tt1;
        tmp=AttributeEvaluate({ZLSE,X},Y,indices); %K=6
        error1(r,4:6)=tmp(1,:);time1(r,4:6)=tmp(2,:)+tt2;
        end
        if n2v==1
            tmp=AttributeEvaluate(ZNV,Y,indices); %K=6
            error1(r,7)=tmp(1,1);
        end
        [tmp,tmp1]=RefineEvaluate(X,Y,indices);
        error2(r,1:4)=mean(tmp);time2(r,1:4)=mean(tmp1);
    end
    [mean(error1);std(error1);]
    [mean(error2);std(error2);]
    save(strcat('GraphRefine',num2str(choice),'CV',num2str(cvf),'.mat'),'choice','error1','time1','error2','time2','cvf');
    % ARI=zeros(4,1);
    % YASE=kmeans(ZASE,K);ARI(1)=RandIndex(Y,YASE);
    % ARI(3)=RandIndex(Y,LeidenY+1);
    % Y2=kmeans(ZLeiden,K);
    % ARI(4)=RandIndex(Y,Y2);
    % [~,Y2]=UnsupGEE(X,K,size(X,1));
    % ARI(2)=RandIndex(Y,Y2);
    % ARI

X=Edge;Y=Label;cvf=10;
indices=crossvalind('Kfold',Y,cvf);
[tmp,tmp1]=RefineEvaluate(X,Y,indices);
mean(tmp)
end

if choice>=30 && choice<=40
    rng("default")
    load('Wiki_Data.mat'); Y=Label;
    error1=zeros(rep,7);error2=zeros(rep,4);time1=zeros(rep,7);time2=zeros(rep,4);dim=20;
    if choice==30 % improve
        X={TE,TF};n2v=0;
        % [Z1]=UnsupGraph(TE,max(Y)*5,length(Y));[Z2]=UnsupGraph(TF,max(Y)*5,length(Y));
        % Z=[Z1,Z2];
        if spectral==1
        tic;ZASE=ASE([X{1},X{2}],dim);n=size(X{1},1);Z1=[ZASE(1:n,:),ZASE(n+1:2*n,:)];t1=toc;
        tic;ZLSE=ASE([Lap(X{1}),Lap(X{2})],dim);Z2=[ZLSE(1:n,:),ZLSE(n+1:2*n,:)];t2=toc;
        end
        % ZLeidenTE=GraphEncoder(TE,TELeidenY);ZLeidenTF=GraphEncoder(TF,TFLeidenY);ZLeiden=[ZLeidenTE,ZLeidenTF];
    end
    if choice==31 % improve
        X={TE,GEAdj};n2v=0;
        % [Z1]=UnsupGraph(TE,max(Y)*5,length(Y));[Z2]=UnsupGraph(TF,max(Y)*5,length(Y));
        % Z=[Z1,Z2];
        if spectral==1
        tic;ZASE=ASE([X{1},X{2}],dim);n=size(X{1},1);Z1=[ZASE(1:n,:),ZASE(n+1:2*n,:)];t1=toc;
        tic;ZLSE=ASE([Lap(X{1}),Lap(X{2})],dim);Z2=[ZLSE(1:n,:),ZLSE(n+1:2*n,:)];t2=toc;
        end
        % ZLeidenTE=GraphEncoder(TE,TELeidenY);ZLeidenTF=GraphEncoder(GEAdj,GELeidenY);ZLeiden=[ZLeidenTE,ZLeidenTF];
    end
    if choice==32 % improve
        X={TF,GFAdj};n2v=0;
        % [Z1]=UnsupGraph(TE,max(Y)*5,length(Y));[Z2]=UnsupGraph(TF,max(Y)*5,length(Y));
        % Z=[Z1,Z2];
        if spectral==1
        tic;ZASE=ASE([X{1},X{2}],dim);n=size(X{1},1);Z1=[ZASE(1:n,:),ZASE(n+1:2*n,:)];t1=toc;
        tic;ZLSE=ASE([Lap(X{1}),Lap(X{2})],dim);Z2=[ZLSE(1:n,:),ZLSE(n+1:2*n,:)];t2=toc;
        end
    end
    if choice==33 % improve
        X={GEAdj,GFAdj};
        % [Z1]=UnsupGraph(TE,max(Y)*5,length(Y));[Z2]=UnsupGraph(TF,max(Y)*5,length(Y));
        % Z=[Z1,Z2];
        if spectral==1
        tic;ZASE=ASE([X{1},X{2}],dim);n=size(X{1},1);Z1=[ZASE(1:n,:),ZASE(n+1:2*n,:)];t1=toc;
        tic;ZLSE=ASE([Lap(X{1}),Lap(X{2})],dim);Z2=[ZLSE(1:n,:),ZLSE(n+1:2*n,:)];t2=toc;
        end
        if n2v==1
            load('n2v.mat')
            ind=(sum(GFAdj)>0);
            GF2=zeros(1382,128);
            GF2(ind,:)=WikiGF;
            ZNV=[WikiGE,GF2];
        end
    end
    % if choice==33 % improve
    %     X={TE,TF,GEAdj};
    %     % [Z1]=UnsupGraph(TE,max(Y)*5,length(Y));[Z2]=UnsupGraph(TF,max(Y)*5,length(Y));
    %     % Z=[Z1,Z2];
    %     tic;ZASE=ASE([X{1},X{2},X{3}],dim);n=size(X{1},1);Z1=[ZASE(1:n,:),ZASE(n+1:2*n,:),ZASE(2*n+1:3*n,:)];t1=toc;
    %     tic;ZLSE=ASE([Lap(X{1}),Lap(X{2}),Lap(X{3})],dim);Z2=[ZLSE(1:n,:),ZLSE(n+1:2*n,:),ZLSE(2*n+1:3*n,:)];t2=toc;
    % end
    % if choice==34 % improve
    %     X={TE,TF,GFAdj};
    %     % [Z1]=UnsupGraph(TE,max(Y)*5,length(Y));[Z2]=UnsupGraph(TF,max(Y)*5,length(Y));
    %     % Z=[Z1,Z2];
    %     tic;ZASE=ASE([X{1},X{2},X{3}],dim);n=size(X{1},1);Z1=[ZASE(1:n,:),ZASE(n+1:2*n,:),ZASE(2*n+1:3*n,:)];t1=toc;
    %     tic;ZLSE=ASE([Lap(X{1}),Lap(X{2}),Lap(X{3})],dim);Z2=[ZLSE(1:n,:),ZLSE(n+1:2*n,:),ZLSE(2*n+1:3*n,:)];t2=toc;
    % end
    % if choice==35 % improve
    %     X={TE,GEAdj,GFAdj};
    %     % [Z1]=UnsupGraph(TE,max(Y)*5,length(Y));[Z2]=UnsupGraph(TF,max(Y)*5,length(Y));
    %     % Z=[Z1,Z2];
    %     tic;ZASE=ASE([X{1},X{2},X{3}],dim);n=size(X{1},1);Z1=[ZASE(1:n,:),ZASE(n+1:2*n,:),ZASE(2*n+1:3*n,:)];t1=toc;
    %     tic;ZLSE=ASE([Lap(X{1}),Lap(X{2}),Lap(X{3})],dim);Z2=[ZLSE(1:n,:),ZLSE(n+1:2*n,:),ZLSE(2*n+1:3*n,:)];t2=toc;
    % end
    % if choice==36 % improve
    %     X={TF,GEAdj,GFAdj};
    %     % [Z1]=UnsupGraph(TE,max(Y)*5,length(Y));[Z2]=UnsupGraph(TF,max(Y)*5,length(Y));
    %     % Z=[Z1,Z2];
    %     tic;ZASE=ASE([X{1},X{2},X{3}],dim);n=size(X{1},1);Z1=[ZASE(1:n,:),ZASE(n+1:2*n,:),ZASE(2*n+1:3*n,:)];t1=toc;
    %     tic;ZLSE=ASE([Lap(X{1}),Lap(X{2}),Lap(X{3})],dim);Z2=[ZLSE(1:n,:),ZLSE(n+1:2*n,:),ZLSE(2*n+1:3*n,:)];t2=toc;
    % end
    if choice==34 % improve
        X={TE,TF,GEAdj,GFAdj};n2v=0;
        % [Z1]=UnsupGraph(TE,max(Y)*5,length(Y));[Z2]=UnsupGraph(TF,max(Y)*5,length(Y));[Z3]=UnsupGraph(GEAdj,max(Y)*5,length(Y));[Z4]=UnsupGraph(GFAdj,max(Y)*5,length(Y));
        % Z=[Z1,Z2,Z3,Z4];
        if spectral==1
        tic;ZASE=ASE([X{1},X{2},X{3},X{4}],dim);n=size(X{1},1);Z1=[ZASE(1:n,:),ZASE(n+1:2*n,:),ZASE(2*n+1:3*n,:),ZASE(3*n+1:4*n,:)];t1=toc;
        tic;ZLSE=ASE([Lap(X{1}),Lap(X{2}),Lap(X{3}),Lap(X{4})],dim);Z2=[ZLSE(1:n,:),ZLSE(n+1:2*n,:),ZLSE(2*n+1:3*n,:),ZLSE(3*n+1:4*n,:)];t2=toc;
        end
    end
    if choice==35 % improve
        load('CElegans.mat');X={double(Ac+Ac'>0),double(Ag+Ag'>0)};Y=vcols;
        if spectral==1
        tic;ZASE=ASE([X{1},X{2}],dim);n=size(X{1},1);Z1=[ZASE(1:n,:),ZASE(n+1:2*n,:)];t1=toc;
        tic;ZLSE=ASE([Lap(X{1}),Lap(X{2})],dim);Z2=[ZLSE(1:n,:),ZLSE(n+1:2*n,:)];t2=toc;
        end
        if n2v==1
            load('n2v.mat')
            ZNV=[CElegansAc,CElegansAg];
        end
    end
    % if choice==40 % improve
    %     load('IMDB.mat');X={Edge1,Edge2};Y=Label2;Z1=0;Z2=0;
    % end
    % if choice==39 % improve
    %     load('Letter.mat');X={Edge1,Edge2,Edge3};Y=Label2;Z1=0;Z2=0;
    % end
    % if choice==39 % improve
    %    load('Cora.mat');Dist1='cosine';D = 1-squareform(pdist(X, Dist1));X={Edge,D};Y=Label;Z1=0;Z2=0;
    % end
    % if choice==35 % improve
    %    load('citeseer.mat');Dist1='cosine';D = 1-squareform(pdist(X, Dist1));X={Edge,D};Y=Label;
    % end
    for r=1:rep
        indices=crossvalind('Kfold',Y,cvf); 
        if spectral==1
        tmp=AttributeEvaluate({Z1,X},Y,indices); %K=6
        error1(r,1:3)=tmp(1,:);time1(r,1:3)=tmp(2,:)+t1;
        tmp=AttributeEvaluate({Z2,X},Y,indices); %K=6
        error1(r,4:6)=tmp(1,:);time1(r,4:6)=tmp(2,:)+t2;
        end
        if n2v==1
            tmp=AttributeEvaluate(ZNV,Y,indices); %K=6
            error1(r,7)=tmp(1,1);
        end
        [tmp,tmp1]=RefineEvaluate(X,Y,indices);
        error2(r,1:4)=mean(tmp);time2(r,1:4)=mean(tmp1);
    end
    [mean(error1);std(error1);]
    [mean(error2);std(error2);]
    save(strcat('GraphRefine',num2str(choice),'CV',num2str(cvf),'.mat'),'choice','error1','time1','error2','time2','cvf');
end


if choice==50
    load('Wiki_Data.mat'); Y=Label;RefineEvaluate(TE,Y);RefineEvaluate(TF,Y);RefineEvaluate(GEAdj,Y);RefineEvaluate(GFAdj,Y);RefineEvaluate({TE,TF},Y);RefineEvaluate({TE,GEAdj},Y);RefineEvaluate({TE,GEAdj,TF,GFAdj},Y);
end

if choice==51
    [X,Y]=simGenerate(400,3000,3,0); Y2=Y;Y2(Y2==2)=1;Y2(Y2==3)=2;[error,tmp]=RefineEvaluate(X,Y);[error,tmp]=RefineEvaluate(X,Y2);
    [Z]=UnsupGraph(X,3,length(Y));AttributeEvaluate({Z,X},Y2)

    [X,Y]=simGenerate(401,3000,4,0); Y2=Y;Y2(Y2==3)=1;Y2(Y2==4)=2;[error,tmp]=RefineEvaluate(X,Y);[error,tmp]=RefineEvaluate(X,Y2);
    [Z]=UnsupGraph(X,4,length(Y));AttributeEvaluate({Z,X},Y2)

    [X,Y]=simGenerate(402,3000,6,0); Y2=Y;Y2(Y2<=3)=1;Y2(Y2>3)=2;[error,tmp]=RefineEvaluate(X,Y);[error,tmp]=RefineEvaluate(X,Y2);
    [Z]=UnsupGraph(X,6,length(Y));AttributeEvaluate({Z,X},Y2)
end

if choice==1 || choice==2 || choice==3 || choice==4
    n=200;lss=12;
    switch choice
        case 1
            [X,Y]=simGenerate(500,n,4,0);G=X;figName='FigRefine1';str1='Simulated Graph 1';
            Y0=Y;Y(Y<=2)=1;Y(Y>=3)=2;%Y(Y==4)=2;
        case 2
            [X,Y]=simGenerate(501,n,4,0);G=X;figName='FigRefine2';str1='Simulated Graph2';
            Y0=Y;Y(Y<=2)=1;Y(Y>=3)=2;%Y(Y==4)=2;
        case 3
            load('karate.mat'); X=G;G=X;figName='FigRefine3';str1='Karate Club Graph';Y0=Y;lss=8;
        case 4
            load('polblogs.mat'); X=Adj;G=X;figName='FigRefine4';str1='Political Blog Graph';Y0=Y;Y=Y0;lss=8;
        % case 4
        %     [X,Y]=simGenerate(502,n,4,0);G=X;figName='FigRefine4';str1='Stochastic Block Model 3';
        %     Y0=Y;Y(Y<=3)=1;Y(Y==4)=2;Y(Y==5)=3;
    end
    % else
    %     load('polblogs.mat');ind=[1:200,1001:1200];X=Adj(ind,ind);G=X;Y=Y(ind);figName='FigRefine2';
    % else
    %     load('polblogs.mat');ind=[1:200,1001:1200];X=Adj(ind,ind);G=X;Y=Y(ind);figName='FigRefine2';

    fs=28;
    opts = struct('Normalize',true,'RefinedK',1,'RefinedY',1,'Principal',0,'Laplacian',false,'Discriminant',true,'Softmax',true);
    [Z1,out1]=GraphEncoder(X,Y,opts);
    idx1=out1.idx;sum(idx1)

    [Z2,out2]=GraphEncoder(G,out1.YVal+idx1*size(Z1,2),opts);
    idx2=out2.idx;
    idxOri=((Y0==2) | (Y0==3));
    acc1=sum(idxOri & idx2)/sum(idx2)
    acc2=sum(idxOri & idx2)/sum(idxOri)
    Y2=out2.Y; 

    A = graph(G,'omitselfloops','upper');
    [C] = conncomp(A);
    % Find the sizes of each connected component
    component_sizes = histcounts(C, 1:max(C)+1);
    % Find the index of the largest connected component
    [~, largest_component_index] = max(component_sizes);
    % Find the indices of nodes belonging to the largest connected component
    indTrn = find(C == largest_component_index);
    A = graph(G(indTrn,indTrn),'omitselfloops','upper');

    myColor = brewermap(18,'RdYlGn');
    colorY=Y0;colorY(Y0==4)=18; colorY(Y0==2)=5;colorY(Y0==3)=14;
    % if choice ==4
    %     colorY(Y0==5)=18; colorY(Y0==4)=10;colorY(Y0==3)=14;colorY(Y0==2)=5;
    % end
    colorY=myColor(colorY,:);
    if choice>2
        t1 = tiledlayout(1,2);
    else
    t1 = tiledlayout(1,3);
    nexttile();
    h=plot(A,'-.r','NodeLabel', {},'NodeColor',colorY(indTrn,:),'MarkerSize',12);%'NodeLabel',Y0(indTrn),
    h.EdgeColor = [0.7, 0.7, 0.7];
    h.LineWidth = 0.5; 
    axis('square'); 
    xlabel('Latent Community');
    set(gca,'fontSize',fs);
    end

    colorY=Y;colorY(Y==2)=18; 
    % if choice ==4
    %     colorY(Y==3)=18; colorY(Y==2)=10; 
    % end
    colorY=myColor(colorY,:);
    nexttile();
    h=plot(A,'-.r','NodeLabel', {},'NodeColor',colorY(indTrn,:),'MarkerSize',12);
    h.EdgeColor = [0.7, 0.7, 0.7];
    h.LineWidth = 0.5; 
    axis('square'); 
    xlabel('Observed Community');
    set(gca,'fontSize',fs);

    nexttile();
    colorY=Y2;colorY(Y2==2)=18;colorY(Y2==4)=5;colorY(Y2==3)=14; 
    % if choice ==4
    %     colorY(Y2==5)=18; colorY(Y2==4)=10;colorY(Y2==3)=14;colorY(Y2==2)=5;
    % end
    colorY=myColor(colorY,:);
    h=plot(A,'-.r','NodeLabel', {},'NodeColor',colorY(indTrn,:),'MarkerSize',12);
    h.EdgeColor = [0.7, 0.7, 0.7];
    h.LineWidth = 0.5;
    xlabel('GEE-Refined Community');
    axis('square'); 
    % if choice>2
    % else
    %     title(strcat('Precision / Recall = ', '[',num2str(floor(acc1*100)/100),',', num2str(floor(acc2*100)/100),']'));
    % end
    set(gca,'fontSize',fs);

    title(t1,str1,'fontSize',fs+12);

    F.fname=figName;
        F.wh=[lss 4]*2;
        %     F.PaperPositionMode='auto';
        print_fig(gcf,F)
end


if choice==5
    % myColor = brewermap(18,'Spectral');
    figName='FigRefine5A';
    myColor2 = brewermap(10,'RdYlBu');lw=4;fs=28;n=10;
    t1 = tiledlayout(1,3);
    for i=1:3
        nexttile();
        switch i
            case 1
                load('GraphRefine101CV10.mat');str='Simulated Graph 1';
            case 2
                load('GraphRefine102CV10.mat');str='Simulated Graph 2';
            case 3
                load('GraphRefine103CV10.mat');str='Simulated Graph 3';
        end
        errorbar(1:n,mean(error1,1),1*std(error1,[],1),'Color', myColor2(4,:),'LineStyle', ':','LineWidth',lw-2);hold on
        errorbar(1:n,mean(error2,1),1*std(error2,[],1),'Color', myColor2(7,:),'LineStyle', '-','LineWidth',lw);
        errorbar(1:n,mean(error4,1),1*std(error4,[],1),'Color', myColor2(10,:),'LineStyle', '-','LineWidth',lw);
        errorbar(1:n,mean(error0,1),1*std(error0,[],1),'Color', myColor2(2,:),'LineStyle', '-','LineWidth',lw);
        if i==3
            legend('GEE0','GEE','R-GEE','ASE','Location','NorthEast');
        end
        xlim([1,10]);xticks([1 5 10]);xticklabels({'200','1000','2000'});%ylim([0,0.5]);
        title(str);

        set(gca,'FontSize',fs);
        axis('square');
    end
    ylabel(t1,'Classification Error','FontSize',fs)
    xlabel(t1,'Number of Vertices','FontSize',fs)

    F.fname=figName;
    F.wh=[12 4]*2;
    %     F.PaperPositionMode='auto';
    print_fig(gcf,F)

    % myColor = brewermap(18,'Spectral');
    figName='FigRefine5B';
    myColor2 = brewermap(10,'Spectral');lw=4;fs=28;n=10;
    t1 = tiledlayout(1,3);
    for i=1:3
        nexttile();
        switch i
            case 1
                load('GraphRefine101CV10.mat');str='Simulated Graph 1';
            case 2
                load('GraphRefine102CV10.mat');str='Simulated Graph 2';
            case 3
                load('GraphRefine103CV10.mat');str='Simulated Graph 3';
        end
        errorbar(1:n,mean(acc1,1),1*std(acc1,[],1),'Color', myColor2(8,:),'LineStyle', '-','LineWidth',lw);hold on
        errorbar(1:n,mean(acc2,1),1*std(acc2,[],1),'Color', myColor2(3,:),'LineStyle', '-','LineWidth',lw);
        if i==3
            legend('R-GEE Precision','R-GEE Recall','Location','East');
        end
        xlim([1,10]);xticks([1 5 10]);xticklabels({'200','1000','2000'});ylim([0,1]);
        title(str);

        set(gca,'FontSize',fs);
        axis('square');
    end
    ylabel(t1,'Community Discovery','FontSize',fs)
    xlabel(t1,'Number of Vertices','FontSize',fs)

    F.fname=figName;
    F.wh=[12 4]*2;
    %     F.PaperPositionMode='auto';
    print_fig(gcf,F)
end

if choice==6;%time figure
%         Spec=2;
        i=10;lw=4;F.fname='FigRefine6';str1='Running Time';loc='East';
        myColor2 = brewermap(10,'RdYlBu');%myColor2 = brewermap(4,'RdYlBu');myColor(10,:)=myColor2(4,:);
%         myColor=[myColor(2,:);myColor(3,:);myColor2(3,:)];
        fs=28;
        load('GraphRefineTime.mat')
        time1=log10(time1);time2=log10(time2);time3=log10(time3);
        errorbar(1:i,mean(time1,2),1*std(time1,[],2),'Color', myColor2(7,:),'LineStyle', '-','LineWidth',lw);hold on
        errorbar(1:i,mean(time2,2),1*std(time2,[],2),'Color', myColor2(10,:),'LineStyle', '-','LineWidth',lw);
        errorbar(1:i,mean(time3,2),1*std(time3,[],2),'Color', myColor2(2,:),'LineStyle', '-','LineWidth',lw);
        legend('GEE','R-GEE','SVD (d=20)','Location','SouthEast');
        xlim([1,10]);xticks([1 5 10]);xticklabels({'0.5M','10M','50M'});
        ylim([-3,3]);yticks([-2 -1 0 1 2]);yticklabels({'0.01','0.1','1','10','100'});
        set(gca,'FontSize',fs); 
        ylabel('Running Time (Log Scale)','FontSize',fs)
        xlabel('Approximate Number of Edges','FontSize',fs)
        axis('square'); 
        %         set(gca,'FontSize',fs);
        F.wh=[4 4]*2;
        %     F.PaperPositionMode='auto';
        print_fig(gcf,F)
end

if choice==101 || choice==102 || choice==103
    % switch choice
    %     case 60
    %         load('karate.mat'); X=G;figName='FigRefine1';str1='Karate Club';Y0=Y;
    %     case 63
    %         [X,Y]=simGenerate(500,200,4,0);G=X;figName='FigRefine2';str1='Stochastic Block Model 1';
    %         Y0=Y;Y0(Y==2)=3;Y0(Y==4)=2;Y0(Y==3)=4;
    %         Y(Y==2)=1;Y(Y==3)=4;Y(Y==4)=2;
    %     case 64
    %         [X,Y]=simGenerate(501,1000,4,0);G=X;figName='FigRefine2';str1='Stochastic Block Model 2';
    %         Y0=Y;Y(Y<=3)=1;Y(Y==4)=2;Y(Y==5)=3;
    % end
    ll=10;fs=12;dim=20;
    opts = struct('Normalize',true,'Refine',0,'Principal',0,'Laplacian',false,'Discriminant',true,'Softmax',false);
    % else
    %     load('polblogs.mat');ind=[1:200,1001:1200];X=Adj(ind,ind);G=X;Y=Y(ind);figName='FigRefine2';
    % else
    %     load('polblogs.mat');ind=[1:200,1001:1200];X=Adj(ind,ind);G=X;Y=Y(ind);figName='FigRefine2';
    acc1=zeros(rep,ll);acc2=zeros(rep,ll);error0=zeros(rep,ll);error1=zeros(rep,ll);error2=zeros(rep,ll);error3=zeros(rep,ll);error4=zeros(rep,ll);error5=zeros(rep,ll);
    for r=1:rep
        for i=1:ll;
            n=200*i;
            [X,Y0]=simGenerate(500+choice-101,n,4,0);G=X;
            [ZASE]=ASE(X,dim);

            %Y0=Y;Y0(Y==2)=3;Y0(Y==4)=2;Y0(Y==3)=4;%Y0 original; Y reduced; Y2 refined from Y.
            Y=Y0;
            Y(Y==2)=1;Y(Y==3)=4;Y(Y==4)=2;
            if choice==8
                Y(Y0<4)=1;
            end
            % 
            [Z2,out2]=RefinedGEE(X,Y,opts);
            idx2=out2.idx;
            % 
            % % [Z2,out2]=GraphEncoder(X,Y2,opts);
            % % idx2=out2.idx;
            % 
            idxOri=((Y0==2) | (Y0==3));
            acc1(r,i)=sum(idxOri & idx2)/sum(idx2);
            acc2(r,i)=sum(idxOri & idx2)/sum(idxOri);

            indices=crossvalind('Kfold',Y,5);
            tmp=AttributeEvaluate(ZASE,Y,indices); error0(r,i)=tmp(1,1);
            [tmp,~]=RefineEvaluate(X,Y0,indices); error1(r,i)=mean(tmp(:,1));
            [tmp,~]=RefineEvaluate(X,Y,indices); error2(r,i)=mean(tmp(:,1));error3(r,i)=mean(tmp(:,2));error4(r,i)=mean(tmp(:,3));error5(r,i)=mean(tmp(:,4));
            % [tmp,~]=RefineEvaluate(X,Y2,indices); error4(r,i)=mean(tmp(:,1));
        end
    end
    [mean(acc1);mean(acc2)]
    [mean(error0);mean(error1);mean(error2);mean(error3);mean(error4);mean(error5)]
    save(strcat('GraphRefine',num2str(choice),'CV',num2str(cvf),'.mat'),'choice','error0','error1','acc1','error2','error3','error4','error5','acc2','cvf');
end

if choice ==104 %time
    opts = struct('Normalize',true,'DiagAugment',false,'Principal',0,'Laplacian',false,'Discriminant',false);
    lim=10;time1=zeros(lim,rep);time2=zeros(lim,rep);time3=zeros(lim,rep);numEdges=zeros(lim,rep);
    for i=1:lim
        n=3000*i
        % GraphEncoder(Dis{1},Label);
        for r=1:rep
            [X,Label]=simGenerate(502,n,4,0);
            X=sparse(X);
            numEdges(i,r)=sum(sum(X));
            tic
            GraphEncoder(X,Label,opts);
            time1(i,r)=toc;
            tic
            RefinedGEE(X,Label);
            time2(i,r)=toc;
            tic
            if spectral==1
            tic
            svds(X,20);
            time3(i,r)=toc;
            end
        end
        [mean(time1,2),mean(time2,2),mean(time3,2)]
        [std(time1,[],2),std(time2,[],2),std(time3,[],2)]
    end
    save('GraphRefineTime.mat','lim','n','time1','time2','time3','numEdges');
end

% 
% if choice==8
%     % switch choice
%     %     case 60
%     %         load('karate.mat'); X=G;figName='FigRefine1';str1='Karate Club';Y0=Y;
%     %     case 63
%     %         [X,Y]=simGenerate(500,200,4,0);G=X;figName='FigRefine2';str1='Stochastic Block Model 1';
%     %         Y0=Y;Y0(Y==2)=3;Y0(Y==4)=2;Y0(Y==3)=4;
%     %         Y(Y==2)=1;Y(Y==3)=4;Y(Y==4)=2;
%     %     case 64
%     %         [X,Y]=simGenerate(501,1000,4,0);G=X;figName='FigRefine2';str1='Stochastic Block Model 2';
%     %         Y0=Y;Y(Y<=3)=1;Y(Y==4)=2;Y(Y==5)=3;
%     % end
%     ll=10;fs=12;
%     opts = struct('Normalize',true,'Refine',0,'Principal',0,'Laplacian',false,'Discriminant',true,'Softmax',false);
%     % else
%     %     load('polblogs.mat');ind=[1:200,1001:1200];X=Adj(ind,ind);G=X;Y=Y(ind);figName='FigRefine2';
%     % else
%     %     load('polblogs.mat');ind=[1:200,1001:1200];X=Adj(ind,ind);G=X;Y=Y(ind);figName='FigRefine2';
%     acc1=zeros(rep,ll);acc2=zeros(rep,ll);error1=zeros(rep,ll);error2=zeros(rep,ll);error3=zeros(rep,ll);error4=zeros(rep,ll);error5=zeros(rep,ll);
%     for r=1:rep
%         for i=1:ll;
%             n=500*i;
%             [X,Y0]=simGenerate(503,n,4,0);G=X;figName='FigRefine4';str1='Stochastic Block Model 1';
%             %Y0=Y;Y0(Y==2)=3;Y0(Y==4)=2;Y0(Y==3)=4;%Y0 original; Y reduced; Y2 refined from Y.
%             Y=Y0; Y(Y0==2)=1;Y(Y0==3)=1;Y(Y0==4)=2;Y(Y0==5)=3;
%             % 
%             % [Z2,out2]=RefinedGEE(X,Y,opts);
%             % idx2=out2.idx;
%             % 
%             % % [Z2,out2]=GraphEncoder(X,Y2,opts);
%             % % idx2=out2.idx;
%             % 
%             % idxOri=((Y0==2) | (Y0==3));
%             % acc1(r,i)=sum(idxOri & idx2)/sum(idx2);
%             % acc2(r,i)=sum(idxOri & idx2)/sum(idxOri);
% 
%             indices=crossvalind('Kfold',Y,5);
%             [tmp,~]=RefineEvaluate(X,Y0,indices); error1(r,i)=mean(tmp(:,1));
%             [tmp,~]=RefineEvaluate(X,Y,indices,0); error2(r,i)=mean(tmp(:,1));error3(r,i)=mean(tmp(:,2));error4(r,i)=mean(tmp(:,3));error5(r,i)=mean(tmp(:,4));
%             % [tmp,~]=RefineEvaluate(X,Y2,indices); error4(r,i)=mean(tmp(:,1));
%         end
%     end
%     [mean(acc1);mean(acc2)]
%     [mean(error1);mean(error2);mean(error3);mean(error4);mean(error5)]
%     % ground-truth classifier, given classifier, refined classifier
% end