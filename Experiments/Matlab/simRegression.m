function simRegression(choice,dim,spec, rep)
% simRegression(1,4,0,3)
% use choice =1 to 12 to replicate the simulation and experiments. 
% use choice =100/101 to plot the simulation figure
% spec =1 for Omnibus benchmark, 2 for USE, 3 for MASE

if nargin<3
    spec=0;
end
if nargin<4
    rep=3;
end

if choice==1
load('protein.mat');
id1=zeros(1,size(X,2));
id1(dim)=1;
Y=X(:,logical(id1));
X=X(:,~id1);
XLabel=onehotencode(categorical(Label),2);
opts = struct('eval',2,'Adjacency',1,'DiagAugment',1,'Laplacian',0,'Spectral',spec,'LDA',0,'GNN',0,'knn',5,'dim',30);

    Acc1=zeros(rep,8);Acc2=zeros(rep,8);Time1=zeros(rep,8);Time2=zeros(rep,8);
%     if spec>0
%         Edge=edge2adj(Edge);
%     end
    %     Y2=kmeans(X,K2,'Distance',Dist2);
%     tmpZ=GraphEncoder(Edge,0,X); 
    GNN=0;
    for i=1:rep
        i
        indices = crossvalind('Kfold',Label,5);
        opts.indices=indices;
        tmp=AttributeEvaluate(X,Y,indices,opts.eval); Acc1(i,1)=tmp(1);Time1(i,1)=tmp(2);
        tmp=GraphEncoderEvaluate(Edge,Y,opts,X); Acc1(i,2)=tmp{1,2-GNN};Acc2(i,2)=tmp{1,4};Time1(i,2)=tmp{4,2-GNN};Time2(i,2)=tmp{4,4};
        tmp=GraphEncoderEvaluate(Edge,Y,opts,[X,XLabel]); Acc1(i,3)=tmp{1,2-GNN};Acc2(i,3)=tmp{1,4};Time1(i,3)=tmp{4,2-GNN};Time2(i,3)=tmp{4,4};
%         tmp=GraphEncoderEvaluate({Edge,D},Label,opts); Acc1(i,3)=tmp{1,2-GNN};Acc2(i,3)=tmp{1,4};Time1(i,3)=tmp{4,2-GNN};Time2(i,3)=tmp{4,4};
%         if spec==0
%             tmp=AttributeEvaluate(tmpZ,Label,indices,opts.eval); Acc1(i,6)=tmp(1);Time1(i,6)=tmp(2);
%             tmp=GraphEncoderEvaluate(Edge,Label,opts,X); Acc1(i,7)=tmp{1,2-GNN};Time1(i,7)=tmp{1,4};
%         end
        %        tmp=GraphEncoderEvaluate(Edge,{Label,Y2},opts); Acc1(i,4)=tmp{1,2-GNN};Acc2(i,4)=tmp{1,4};
        %        tmp=GraphEncoderEvaluate({Edge,D},{Label,Y2},opts); Acc1(i,5)=tmp{1,2-GNN};Acc2(i,5)=tmp{1,4};
        %        Z=GraphEncoder(Edge,Y2);tmp=AttributeEvaluate(Z,Label,indices); Acc1(i,6)=tmp;Acc2(i,6)=tmp;
        %        tmp=GraphEncoderEvaluate(Edge,Label,opts,X); Acc1(i,7)=tmp{1,2-GNN};Acc2(i,7)=tmp{1,4};
    end
    save(strcat('GEERegression',num2str(choice),'Dim',num2str(dim),'.mat'),'choice','Acc1','Acc2','Time1','Time2');
    [mean(Acc1);mean(Acc2);mean(Time1);mean(Time2)]
    [std(Acc1);std(Acc2);std(Time1);std(Time2)]
end

if choice==2
load('COIL-RAG.mat');
id1=zeros(1,size(X,2));
id1(dim)=1;
Y=X(:,logical(id1));
X=X(:,~id1);
XLabel=onehotencode(categorical(Label),2);
opts = struct('eval',2,'Adjacency',1,'DiagAugment',1,'Laplacian',0,'Spectral',spec,'LDA',0,'GNN',0,'knn',5,'dim',30);

    Acc1=zeros(rep,8);Acc2=zeros(rep,8);Time1=zeros(rep,8);Time2=zeros(rep,8);
%     if spec>0
%         Edge=edge2adj(Edge);
%     end
    %     Y2=kmeans(X,K2,'Distance',Dist2);
%     tmpZ=GraphEncoder(Edge,0,X); 
    GNN=0;
    for i=1:rep
        i
        indices = crossvalind('Kfold',Label,5);
        opts.indices=indices;
        tmp=AttributeEvaluate(X,Y,indices,opts.eval); Acc1(i,1)=tmp(1);Time1(i,1)=tmp(2);
        tmp=GraphEncoderEvaluate(Edge,Y,opts,X); Acc1(i,2)=tmp{1,2-GNN};Acc2(i,2)=tmp{1,4};Time1(i,2)=tmp{4,2-GNN};Time2(i,2)=tmp{4,4};
        tmp=GraphEncoderEvaluate(Edge,Y,opts,[X,XLabel]); Acc1(i,3)=tmp{1,2-GNN};Acc2(i,3)=tmp{1,4};Time1(i,3)=tmp{4,2-GNN};Time2(i,3)=tmp{4,4};
%         tmp=GraphEncoderEvaluate({Edge,D},Label,opts); Acc1(i,3)=tmp{1,2-GNN};Acc2(i,3)=tmp{1,4};Time1(i,3)=tmp{4,2-GNN};Time2(i,3)=tmp{4,4};
%         if spec==0
%             tmp=AttributeEvaluate(tmpZ,Label,indices,opts.eval); Acc1(i,6)=tmp(1);Time1(i,6)=tmp(2);
%             tmp=GraphEncoderEvaluate(Edge,Label,opts,X); Acc1(i,7)=tmp{1,2-GNN};Time1(i,7)=tmp{1,4};
%         end
        %        tmp=GraphEncoderEvaluate(Edge,{Label,Y2},opts); Acc1(i,4)=tmp{1,2-GNN};Acc2(i,4)=tmp{1,4};
        %        tmp=GraphEncoderEvaluate({Edge,D},{Label,Y2},opts); Acc1(i,5)=tmp{1,2-GNN};Acc2(i,5)=tmp{1,4};
        %        Z=GraphEncoder(Edge,Y2);tmp=AttributeEvaluate(Z,Label,indices); Acc1(i,6)=tmp;Acc2(i,6)=tmp;
        %        tmp=GraphEncoderEvaluate(Edge,Label,opts,X); Acc1(i,7)=tmp{1,2-GNN};Acc2(i,7)=tmp{1,4};
    end
    save(strcat('GEERegression',num2str(choice),'Dim',num2str(dim),'.mat'),'choice','Acc1','Acc2','Time1','Time2');
    [mean(Acc1);mean(Acc2);mean(Time1);mean(Time2)]
    [std(Acc1);std(Acc2);std(Time1);std(Time2)]
end