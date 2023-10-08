function simNN(choice,rep,spec)
% use choice =1 to 12 to replicate the simulation and experiments. 
% use choice =100/101 to plot the simulation figure
% spec =1 for Omnibus benchmark, 2 for USE, 3 for MASE

if nargin<3
    spec=0;
end
if nargin<2
    rep=3;
end
ind=1;ind2=3;ind3=2;
% Figure 1 SBM
if choice==1 || choice==2
    lim=10;G1=cell(lim,rep);G2=cell(lim,rep);G3=cell(lim,rep);G12=cell(lim,rep);G23=cell(lim,rep);G13=cell(lim,rep);G123=cell(lim,rep);
    opts = struct('Adjacency',1,'Laplacian',0,'Spectral',0,'LDA',0,'GNN',1,'knn',5,'dim',30);
    for i=1:lim
        for r=1:rep
            n=100*i
            [Dis,Label]=simGenerate(18+(choice-1)*10,n,1);
            indices = crossvalind('Kfold',Label,3);
            opts.indices=indices;
            G1{i,r}=GraphEncoderEvaluate(Dis{1},Label,opts);
            G2{i,r}=GraphEncoderEvaluate(Dis{2},Label,opts);
            G3{i,r}=GraphEncoderEvaluate(Dis{3},Label,opts);
            G13{i,r}=GraphEncoderEvaluate({Dis{1},Dis{3}},Label,opts);
            G23{i,r}=GraphEncoderEvaluate({Dis{2},Dis{3}},Label,opts);
            G12{i,r}=GraphEncoderEvaluate({Dis{1},Dis{2}},Label,opts);
            G123{i,r}=GraphEncoderEvaluate(Dis,Label,opts);
        end
    end
    Acc1=zeros(lim,6);Acc2=zeros(lim,6);Acc3=zeros(lim,6);Acc12=zeros(lim,6);Acc23=zeros(lim,6);Acc13=zeros(lim,6);Acc123=zeros(lim,6);
    for i=1:lim
        for r=1:rep
            Acc1(i,1)=Acc1(i,1)+G1{i,r}{1,1}/rep;Acc1(i,2)=Acc1(i,2)+G1{i,r}{1,2}/rep;Acc1(i,3)=Acc1(i,3)+G1{i,r}{1,4}/rep;
            Acc1(i,4)=Acc1(i,4)+G1{i,r}{4,1}/rep;Acc1(i,5)=Acc1(i,5)+G1{i,r}{4,2}/rep;Acc1(i,6)=Acc1(i,6)+G1{i,r}{4,4}/rep;
            Acc2(i,1)=Acc2(i,1)+G2{i,r}{1,1}/rep;Acc2(i,2)=Acc2(i,2)+G2{i,r}{1,2}/rep;Acc2(i,3)=Acc2(i,3)+G2{i,r}{1,4}/rep;
            Acc2(i,4)=Acc2(i,4)+G2{i,r}{4,1}/rep;Acc2(i,5)=Acc2(i,5)+G2{i,r}{4,2}/rep;Acc2(i,6)=Acc2(i,6)+G2{i,r}{4,4}/rep;
            Acc3(i,1)=Acc3(i,1)+G3{i,r}{1,1}/rep;Acc3(i,2)=Acc3(i,2)+G3{i,r}{1,2}/rep;Acc3(i,3)=Acc3(i,3)+G3{i,r}{1,4}/rep;
            Acc3(i,4)=Acc3(i,4)+G3{i,r}{4,1}/rep;Acc3(i,5)=Acc3(i,5)+G3{i,r}{4,2}/rep;Acc3(i,6)=Acc3(i,6)+G3{i,r}{4,4}/rep;
            Acc12(i,1)=Acc12(i,1)+G12{i,r}{1,1}/rep;Acc12(i,2)=Acc12(i,2)+G12{i,r}{1,2}/rep;Acc12(i,3)=Acc12(i,3)+G12{i,r}{1,4}/rep;
            Acc12(i,4)=Acc12(i,4)+G12{i,r}{4,1}/rep;Acc12(i,5)=Acc12(i,5)+G12{i,r}{4,2}/rep;Acc12(i,6)=Acc12(i,6)+G12{i,r}{4,4}/rep;
            Acc13(i,1)=Acc13(i,1)+G13{i,r}{1,1}/rep;Acc13(i,2)=Acc13(i,2)+G13{i,r}{1,2}/rep;Acc13(i,3)=Acc13(i,3)+G13{i,r}{1,4}/rep;
            Acc13(i,4)=Acc13(i,4)+G13{i,r}{4,1}/rep;Acc13(i,5)=Acc13(i,5)+G13{i,r}{4,2}/rep;Acc13(i,6)=Acc13(i,6)+G13{i,r}{4,4}/rep;
            Acc23(i,1)=Acc23(i,1)+G23{i,r}{1,1}/rep;Acc23(i,2)=Acc23(i,2)+G23{i,r}{1,2}/rep;Acc23(i,3)=Acc23(i,3)+G23{i,r}{1,4}/rep;
            Acc23(i,4)=Acc23(i,4)+G23{i,r}{4,1}/rep;Acc23(i,5)=Acc23(i,5)+G23{i,r}{4,2}/rep;Acc23(i,6)=Acc23(i,6)+G23{i,r}{4,4}/rep;
            Acc123(i,1)=Acc123(i,1)+G123{i,r}{1,1}/rep;Acc123(i,2)=Acc123(i,2)+G123{i,r}{1,2}/rep;Acc123(i,3)=Acc123(i,3)+G123{i,r}{1,4}/rep;
            Acc123(i,4)=Acc123(i,4)+G123{i,r}{4,1}/rep;Acc123(i,5)=Acc123(i,5)+G123{i,r}{4,2}/rep;Acc123(i,6)=Acc123(i,6)+G123{i,r}{4,4}/rep;
        end
    end
    save(strcat('GEEFusionSim',num2str(choice),'Spec',num2str(spec),'.mat'),'choice','Acc1','Acc2','Acc3','Acc12','Acc13','Acc23','Acc123');
%     [mean(Acc1);mean(Acc2);mean(Acc3);mean(Acc12);mean(Acc13);mean(Acc23);mean(Acc123)]
%     [std(Acc1);std(Acc2);std(Acc3);std(Acc12);std(Acc13);std(Acc23);std(Acc123)]
end

if choice==3
    lim=10;G1=cell(lim,rep);G2=cell(lim,rep);G3=cell(lim,rep);G4=cell(lim,rep);
    opts = struct('Adjacency',1,'Laplacian',0,'Spectral',0,'LDA',0,'GNN',0,'knn',5,'dim',30);
    for i=1:lim
        for r=1:rep
            n=100*i
            [Dis,Label]=simGenerate(11,n,5);
            Dis1=simGenerate(11,n,5);
            Dis2=simGenerate(11,n,5);
            Dis3=simGenerate(11,n,5);
            Dis4=simGenerate(11,n,5);
            indices = crossvalind('Kfold',Label,3);
            opts.indices=indices;
            G1{i,r}=GraphEncoderEvaluate(Dis,Label,opts);
            G2{i,r}=GraphEncoderEvaluate(Dis1,Label,opts);
            G3{i,r}=GraphEncoderEvaluate({Dis,Dis1},Label,opts);
            G4{i,r}=GraphEncoderEvaluate({Dis,Dis1,Dis2,Dis3,Dis4},Label,opts);
        end
    end
    Acc1=zeros(lim,6);Acc2=zeros(lim,6);Acc3=zeros(lim,6);Acc4=zeros(lim,6);
    for i=1:lim
        for r=1:rep
            Acc1(i,1)=Acc1(i,1)+G1{i,r}{1,1}/rep;Acc1(i,2)=Acc1(i,2)+G1{i,r}{1,2}/rep;Acc1(i,3)=Acc1(i,3)+G1{i,r}{1,4}/rep;
            Acc1(i,4)=Acc1(i,4)+G1{i,r}{4,1}/rep;Acc1(i,5)=Acc1(i,5)+G1{i,r}{4,2}/rep;Acc1(i,6)=Acc1(i,6)+G1{i,r}{4,4}/rep;
            Acc2(i,1)=Acc2(i,1)+G2{i,r}{1,1}/rep;Acc2(i,2)=Acc2(i,2)+G2{i,r}{1,2}/rep;Acc2(i,3)=Acc2(i,3)+G2{i,r}{1,4}/rep;
            Acc2(i,4)=Acc2(i,4)+G2{i,r}{4,1}/rep;Acc2(i,5)=Acc2(i,5)+G2{i,r}{4,2}/rep;Acc2(i,6)=Acc2(i,6)+G2{i,r}{4,4}/rep;
            Acc3(i,1)=Acc3(i,1)+G3{i,r}{1,1}/rep;Acc3(i,2)=Acc3(i,2)+G3{i,r}{1,2}/rep;Acc3(i,3)=Acc3(i,3)+G3{i,r}{1,4}/rep;
            Acc3(i,4)=Acc3(i,4)+G3{i,r}{4,1}/rep;Acc3(i,5)=Acc3(i,5)+G3{i,r}{4,2}/rep;Acc3(i,6)=Acc3(i,6)+G3{i,r}{4,4}/rep;
            Acc4(i,1)=Acc4(i,1)+G4{i,r}{1,1}/rep;Acc4(i,2)=Acc4(i,2)+G4{i,r}{1,2}/rep;Acc4(i,3)=Acc4(i,3)+G4{i,r}{1,4}/rep;
            Acc4(i,4)=Acc4(i,4)+G4{i,r}{4,1}/rep;Acc4(i,5)=Acc4(i,5)+G4{i,r}{4,2}/rep;Acc4(i,6)=Acc4(i,6)+G4{i,r}{4,4}/rep;
        end
    end
    save(strcat('GEEFusionSim',num2str(choice),'Spec',num2str(spec),'.mat'),'choice','Acc1','Acc2','Acc3','Acc4');
%     [mean(Acc1);mean(Acc2);mean(Acc3);mean(Acc4)]
%     [std(Acc1);std(Acc2);std(Acc3);std(Acc4)]
end

if choice==6
    [X,Label]=simGenerateDis(1,100,5);
%     Y=[1,2,3,4,5];
%     Z=-0.5+1:5;
%     A=X*Z;
    D=squareform(pdist(X));
%     D=squareform(pdist(X,'cosine'));
%     [Z]=GraphEncoder(1-A,Label);

     opts = struct('Adjacency',1,'DiagAugment',1,'Laplacian',0,'Spectral',0,'LDA',1,'GNN',1,'knn',3,'dim',30);
    Acc1=zeros(rep,3);Acc2=zeros(rep,3);Time1=zeros(rep,3);Time2=zeros(rep,3);Acc3=zeros(rep,3);Time3=zeros(rep,3);
    for i=1:rep
        i
       indices = crossvalind('Kfold',Label,5);
       opts.indices=indices;
       tmp=GraphEncoderEvaluate(1-D,Label,opts); 
       Acc1(i,2)=tmp{1,ind};Acc2(i,2)=tmp{1,ind2};Acc3(i,2)=tmp{1,ind3};Time1(i,2)=tmp{4,ind};Time2(i,2)=tmp{4,ind2};Time3(i,2)=tmp{4,ind3};
    end
    save(strcat('GEENN',num2str(choice),'Spec',num2str(spec),'.mat'),'choice','Acc1','Acc2','Acc3','Time1','Time2','Time3');
     [mean(Acc1);mean(Acc2);mean(Acc3);mean(Time1);mean(Time2);mean(Time3)]
end
