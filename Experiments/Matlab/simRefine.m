function simRefine(choice,rep,cvf)

rng("default")
if nargin<2
rep=3;
end
if nargin<3
cvf=10;
end
thres=0.98;
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
    switch choice
        case 10
            load('Cora.mat');X=edge2adj(Edge);Y=Label;
        case 11
            load('citeseer.mat');X=edge2adj(Edge);Y=Label;
        case 12 %remove
            load('email.mat'); X=AdjOri;
        case 13 
            load('karate.mat'); X=G;
        case 14 %remove
            load('IIP.mat'); X=Adj;
        case 15
            load('lastfm.mat'); X=AdjOri;
        case 16
            load('polblogs.mat');X=Adj;
        case 17
            load('pubmed.mat');X=edge2adj(Edge);Y=Label;
        case 18
            load('CElegans.mat');X=Ac;Y=vcols;LeidenY=AcLeidenY;
        case 19
            load('CElegans.mat');X=Ag;Y=vcols;LeidenY=AgLeidenY;
        case 20
            load('Gene.mat'); X=AdjOri;
        case 21 % improve
            load('Wiki_Data.mat'); X=TE;Y=Label;LeidenY=TELeidenY;
        case 22 % improve
            load('Wiki_Data.mat'); X=TF;Y=Label;LeidenY=TFLeidenY;
        case 23
            load('Wiki_Data.mat'); X=GEAdj;Y=Label;LeidenY=GELeidenY;
        case 24
            load('Wiki_Data.mat'); X=GFAdj;Y=Label;LeidenY=GFLeidenY;
        case 25
            load('adjnoun.mat'); X=Adj;
        % case 27
        %     load('letter.mat'); X=edge2adj(Edge1);Y=Label1;LeidenY=LeidenY1;
        % case 28
        %     load('protein.mat'); X=edge2adj(Edge);Y=Label;
        % case 29
        %     load('soc-political-retweet.mat'); X=edge2adj(Edge);Y=Label;
        case 26
            load('SBM.mat'); X=Adj1;Y=Y1;LeidenY=LeidenY1;%Y(Y==2)=1;Y(Y==3)=4;Y(Y==4)=2;
        case 27
            load('SBM.mat'); X=Adj1;Y=Y1;LeidenY=LeidenY1;Y(Y==2)=1;Y(Y==3)=4;Y(Y==4)=2;
        case 28
            load('SBM.mat'); X=Adj2;Y=Y2;LeidenY=LeidenY2;%Y(Y==2)=1;Y(Y==3)=4;Y(Y==4)=2;
        case 29
            load('SBM.mat'); X=Adj2;Y=Y2;LeidenY=LeidenY2;Y(Y==2)=1;Y(Y==3)=4;Y(Y==4)=2;
    end
    %RefineEvaluate(X,Y);
    K=max(Y);
    tic
    %[Z]=UnsupGraph(X,max(Y)*5,length(Y));
    [ZASE]=ASE(X,20);
    ZLeiden=GraphEncoder(X,LeidenY);
    tt=toc;
    % tic
    % [Z2]=ASE(X,10,true);
    % tt2=toc;
    error=zeros(rep,9);time=zeros(rep,9);
    for r=1:rep
        indices=crossvalind('Kfold',Y,cvf); 
        tmp=AttributeEvaluate({ZASE,X},Y,indices); %K=6
        error(r,1:3)=tmp(1,:);time(r,1:3)=tmp(2,:)+tt;
        tmp=AttributeEvaluate({ZLeiden,X},Y,indices); %K=6
        error(r,4:6)=tmp(1,:);time(r,4:6)=tmp(2,:);
        [tmp,tmp1]=RefineEvaluate(X,Y,indices);
        error(r,7:9)=mean(tmp);time(r,7:9)=mean(tmp1);
    end
    % ARI=zeros(4,1);
    % YASE=kmeans(ZASE,K);ARI(1)=RandIndex(Y,YASE);
    % ARI(3)=RandIndex(Y,LeidenY+1);
    % Y2=kmeans(ZLeiden,K);
    % ARI(4)=RandIndex(Y,Y2);
    % [~,Y2]=UnsupGEE(X,K,size(X,1));
    % ARI(2)=RandIndex(Y,Y2);
    [mean(error);std(error);mean(time);std(time)]
    % ARI
    save(strcat('GraphRefine',num2str(choice),'CV',num2str(cvf),'.mat'),'choice','error','time','cvf');
end

if choice>=30 && choice<=40
    load('Wiki_Data.mat'); Y=Label;
    error=zeros(rep,9);
    tic
    if choice==30 % improve
        X={TE,TF};
        % [Z1]=UnsupGraph(TE,max(Y)*5,length(Y));[Z2]=UnsupGraph(TF,max(Y)*5,length(Y));
        % Z=[Z1,Z2];
        ZASE=ASE([TE,TF],20);
        ZLeidenTE=GraphEncoder(TE,TELeidenY);ZLeidenTF=GraphEncoder(TF,TFLeidenY);ZLeiden=[ZLeidenTE,ZLeidenTF];
    end
    if choice==31 % improve
        X={TE,GEAdj};
        % [Z1]=UnsupGraph(TE,max(Y)*5,length(Y));[Z2]=UnsupGraph(TF,max(Y)*5,length(Y));
        % Z=[Z1,Z2];
        ZASE=ASE([TE,GEAdj],20);
        ZLeidenTE=GraphEncoder(TE,TELeidenY);ZLeidenTF=GraphEncoder(GEAdj,GELeidenY);ZLeiden=[ZLeidenTE,ZLeidenTF];
    end
    if choice==32 % improve
        X={TE,GFAdj};
        % [Z1]=UnsupGraph(TE,max(Y)*5,length(Y));[Z2]=UnsupGraph(TF,max(Y)*5,length(Y));
        % Z=[Z1,Z2];
        ZASE=ASE([TE,GFAdj],20);
        ZLeidenTE=GraphEncoder(TF,TFLeidenY);ZLeidenTF=GraphEncoder(GFAdj,GFLeidenY);ZLeiden=[ZLeidenTE,ZLeidenTF];
    end
    if choice==33 % improve
        X={TE,TF,GEAdj};
        % [Z1]=UnsupGraph(TE,max(Y)*5,length(Y));[Z2]=UnsupGraph(TF,max(Y)*5,length(Y));
        % Z=[Z1,Z2];
        ZASE=ASE([TE,TF,GEAdj],20);
        ZLeidenTE=GraphEncoder(TF,TFLeidenY);ZLeidenTF=GraphEncoder(TF,TFLeidenY);ZLeidenGE=GraphEncoder(GEAdj,GELeidenY);ZLeiden=[ZLeidenTE,ZLeidenTF,ZLeidenGE];
    end
    if choice==34 % improve
        X={TE,TF,GFAdj};
        % [Z1]=UnsupGraph(TE,max(Y)*5,length(Y));[Z2]=UnsupGraph(TF,max(Y)*5,length(Y));
        % Z=[Z1,Z2];
        ZASE=ASE([TE,TF,GFAdj],20);
        ZLeidenTE=GraphEncoder(TF,TFLeidenY);ZLeidenTF=GraphEncoder(TF,TFLeidenY);ZLeidenGE=GraphEncoder(GFAdj,GFLeidenY);ZLeiden=[ZLeidenTE,ZLeidenTF,ZLeidenGE];
    end
    if choice==35 % improve
        X={TE,TF,GEAdj,GFAdj};
        % [Z1]=UnsupGraph(TE,max(Y)*5,length(Y));[Z2]=UnsupGraph(TF,max(Y)*5,length(Y));[Z3]=UnsupGraph(GEAdj,max(Y)*5,length(Y));[Z4]=UnsupGraph(GFAdj,max(Y)*5,length(Y));
        % Z=[Z1,Z2,Z3,Z4];
        ZASE=ASE([TE,TF,GEAdj,GFAdj],20);
        ZLeidenTE=GraphEncoder(TF,TFLeidenY);ZLeidenTF=GraphEncoder(TF,TFLeidenY);ZLeidenGE=GraphEncoder(GEAdj,GELeidenY);ZLeidenGF=GraphEncoder(GFAdj,GFLeidenY);ZLeiden=[ZLeidenTE,ZLeidenTF,ZLeidenGE,ZLeidenGF];
    end
    if choice==36 % improve
        load('CElegans.mat');X={Ac,Ag};Y=vcols;
        ZASE=ASE([Ac,Ag],20);
        ZLeidenTE=GraphEncoder(Ac,AcLeidenY);ZLeidenTF=GraphEncoder(Ag,AgLeidenY);ZLeiden=[ZLeidenTE,ZLeidenTF];
    end
    % if choice==32 % improve
    %     load('IMDB.mat');X={Edge1,Edge2};Y=Label2;
    % end
    % if choice==33 % improve
    %     load('Letter.mat');X={Edge1,Edge2,Edge3};Y=Label2;
    % end
    % if choice==34 % improve
    %    load('Cora.mat');Dist1='cosine';D = 1-squareform(pdist(X, Dist1));X={Edge,D};Y=Label;
    % end
    % if choice==35 % improve
    %    load('citeseer.mat');Dist1='cosine';D = 1-squareform(pdist(X, Dist1));X={Edge,D};Y=Label;
    % end
    tt=toc;
    for r=1:rep
        indices=crossvalind('Kfold',Y,cvf); 
        tmp=AttributeEvaluate({ZASE,X},Y,indices); %K=6
        error(r,1:3)=tmp(1,:);time(r,1:3)=tmp(2,:)+tt;
        tmp=AttributeEvaluate({ZLeiden,X},Y,indices); %K=6
        error(r,4:6)=tmp(1,:);time(r,4:6)=tmp(2,:)+tt;
        [tmp,tmp1]=RefineEvaluate(X,Y,indices);
        error(r,7:9)=mean(tmp);time(r,7:9)=mean(tmp1);
    end
    [mean(error);std(error);mean(time);std(time)]
    save(strcat('GraphRefine',num2str(choice),'CV',num2str(cvf),'.mat'),'choice','error','time','cvf');
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

if choice==1 || choice==2 || choice==3
    switch choice
        case 1
            load('karate.mat'); X=G;figName='FigRefine1';str1='Karate Club';Y0=Y;
        case 2
            [X,Y]=simGenerate(500,100,4,0);G=X;figName='FigRefine2';str1='Stochastic Block Model 1';
            Y0=Y;Y0(Y==2)=3;Y0(Y==4)=2;Y0(Y==3)=4;
            Y(Y==2)=1;Y(Y==3)=4;Y(Y==4)=2;
        case 3
            [X,Y]=simGenerate(501,100,4,0);G=X;figName='FigRefine2';str1='Stochastic Block Model 2';
            Y0=Y;Y(Y<=3)=1;Y(Y==4)=2;Y(Y==5)=3;
    end
    % else
    %     load('polblogs.mat');ind=[1:200,1001:1200];X=Adj(ind,ind);G=X;Y=Y(ind);figName='FigRefine2';
    % else
    %     load('polblogs.mat');ind=[1:200,1001:1200];X=Adj(ind,ind);G=X;Y=Y(ind);figName='FigRefine2';
    fs=12;
    opts = struct('Normalize',true,'Refine',0,'Principal',0,'Laplacian',false,'Discriminant',true,'Softmax',true);
    [Z1,out1]=RefinedGEE(X,Y,opts);
    idx1=out1.idx;

    opts = struct('Normalize',true,'Refine',1,'Principal',0,'Laplacian',false,'Discriminant',true,'Softmax',true);
    [Z2,out2]=RefinedGEE(X,Y,opts);
    idx2=out2.idx;Y2=Y;
    Y2(idx2)=Y2(idx2)+2;

    A = graph(G,'omitselfloops','upper');
    myColor = brewermap(8,'PiYg');
    colorY=Y;colorY(Y==2)=8; colorY=myColor(colorY,:);
    t1 = tiledlayout(1,2);
    nexttile();
    plot(A,'-.r','NodeLabel',Y0,'NodeColor',colorY,'MarkerSize',10);
    axis('square'); 
    xlabel('Ground-Truth Class');
    set(gca,'fontSize',fs);

    nexttile();
    colorY=Y2;colorY(Y2==2)=8;colorY(Y2==3)=3;colorY(Y2==4)=6; colorY=myColor(colorY,:);
    plot(A,'-.r','NodeLabel',Y0,'NodeColor',colorY,'MarkerSize',10);
    axis('square'); 
    xlabel('GEE-Refined Class')
    set(gca,'fontSize',fs);

    title(t1,str1,'fontSize',fs+12);

    F.fname=figName;
        F.wh=[8 4]*2;
        %     F.PaperPositionMode='auto';
        print_fig(gcf,F)
end

if choice==6 || choice==7 || choice==8
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
    rep=10;ll=10;fs=12;
    opts = struct('Normalize',true,'Refine',0,'Principal',0,'Laplacian',false,'Discriminant',true,'Softmax',false);
    % else
    %     load('polblogs.mat');ind=[1:200,1001:1200];X=Adj(ind,ind);G=X;Y=Y(ind);figName='FigRefine2';
    % else
    %     load('polblogs.mat');ind=[1:200,1001:1200];X=Adj(ind,ind);G=X;Y=Y(ind);figName='FigRefine2';
    acc1=zeros(rep,ll);acc2=zeros(rep,ll);error1=zeros(rep,ll);error2=zeros(rep,ll);error3=zeros(rep,ll);error4=zeros(rep,ll);
    for r=1:rep
        for i=1:ll;
            n=200*i;
            [X,Y0]=simGenerate(500+choice-6,n,4,0);G=X;figName='FigRefine4';str1='Stochastic Block Model 1';
            %Y0=Y;Y0(Y==2)=3;Y0(Y==4)=2;Y0(Y==3)=4;%Y0 original; Y reduced; Y2 refined from Y.
            Y=Y0;
            Y(Y==2)=1;Y(Y==3)=4;Y(Y==4)=2;
            % 
            [Z2,out2]=RefinedGEE(X,Y,opts);
            idx2=out2.idx;Y2=Y;
            Y2(idx2)=Y2(idx2)+max(Y);
            % 
            % % [Z2,out2]=RefinedGEE(X,Y2,opts);
            % % idx2=out2.idx;
            % 
            % idxOri=(Y0>2);
            % acc1(r,i)=sum(idxOri & idx2)/sum(idx2);
            % acc2(r,i)=sum(idxOri & idx2)/sum(idxOri);

            indices=crossvalind('Kfold',Y,5);
            [tmp,~]=RefineEvaluate(X,Y0,indices); error1(r,i)=mean(tmp(:,1));
            [tmp,~]=RefineEvaluate(X,Y,indices); error2(r,i)=mean(tmp(:,1));error3(r,i)=mean(tmp(:,3));
            [tmp,~]=RefineEvaluate(X,Y2,indices); error4(r,i)=mean(tmp(:,1));
        end
    end
    mean(acc1)
    mean(acc2)
    mean(error1)
    mean(error2)
    mean(error3)
    mean(error4)
    % ground-truth classifier, given classifier, refined classifier
end