function result=simStructure(choice)

%% GCN paramers
% net2=net1;
% netGFN=feedforwardnet(30);
% d=100;n=1000;dd=1;
% [A,Y]=simGenerate(121,n,d,1);
% 
% opts = struct('DiagA',false,'Normalize',true);
% [Z,Y]=GraphEncoder(A,Y,opts);
% 
% pcorr = corr(Z, Y)
% [~,ind]=sort(pcorr,'descend');
% 
% [dcorInd,dcorNum,pval] = DCorScreening(Z,Y)

% % choice=1;
% ll=true;
% % ll=false
newK=10;
switch choice
    case 1
        load('anonymized_msft.mat')
        X=G{1}; newK=10;
    case 2
        load('smartphone.mat')
        X=G;Y=label; newK=10;
    case 11 % Works?
        load('polblogs.mat')
        X=adj2edge(Adj);Y=Y;newK=10; 
    case 12
        load('CoraAdj.mat') %AEL / GFN K=2
        X=Edge;newK=20;
    case 13
        load('email.mat') %AEL / GFN K=2. Works?
        X=Edge;newK=5;
    case 14
        load('lastFM.mat')
        X=Edge;newK=10;
    case 15
        load('pubmedEdge.mat')
        X=Edge;newK=20;
    case 16
        load('graphCElegans.mat')
        X=Ac;Y=vcols;newK=10;
%     case 17
%         load('adjnoun.mat')
%         X=Edge+1;newK=10;
%     case 17
%         load('KKI.mat')
%         X=Edge;newK=3;
%     case 3
%         load('OHSU.mat')
%         X=Edge;newK=3;
%     case 4
%         load('Peking.mat')
%         X=Edge;newK=3;
    case 18
        load('Gene.mat') %AEL / GFN K=2
        X=Edge;newK=5; 
    case 19
        load('IIP.mat') %AEL / GFN K=2
        X=Edge;newK=10;
    case 21
        load('Wiki_Data.mat')
        Y=Label+1;X=TE; 
    case 22
        load('Wiki_Data.mat')
        Y=Label+1;X=TF; 
    case 23
        load('Wiki_Data.mat')
        Y=Label+1;X=adj2edge(GE); 
    case 24
        load('Wiki_Data.mat')
        Y=Label+1;X=adj2edge(GF); 
    case 25
        load('Wiki_Data.mat')
        Y=Label+1;X={TE,TF};
    case 26
        load('Wiki_Data.mat')
        Y=Label+1;X={adj2edge(GF),adj2edge(GE)}; 
    case 27
        load('Wiki_Data.mat')
        Y=Label+1;X={TE,adj2edge(GE)}; 
    case 28
        load('Wiki_Data.mat')
        Y=Label+1;X={TF,adj2edge(GF)}; 
    case 29
        load('Wiki_Data.mat')
        Y=Label+1;X={TE,TF,adj2edge(GE),adj2edge(GF)};
    case 30 %improve about 60 observe per class, should achieve 0.2 
        [X,Y]=simGenerate(130,3000,3,1); newK=20; 
    case 31 %improve about 10 observe per class
        [X,Y]=simGenerate(131,500,3,1); newK=50;
    case 32 %improve about 10 observe per class
        [X,Y]=simGenerate(132,500,3,1); newK=20;
end
% Yexp=LabelProbagation(Y, 50);

net1 = patternnet(10,'trainscg','crossentropy'); % number of neurons, Scaled Conjugate Gradient, cross entropy
net1.layers{1}.transferFcn = 'poslin';
net1.trainParam.showWindow = false;
net1.trainParam.epochs=100;
net1.divideParam.trainRatio = 0.8;
net1.divideParam.valRatio   = 0.2;
net1.divideParam.testRatio  = 0/100;

n=length(Y); K=max(Y);
% Yexp=randi([1,kmax],n,1);
% Yext=cell(1,rep);
% for r=1:rep
%     Yext{r}=LabelExt(Y,newC);
% end
% if ll==true;
% YExt =LabelExt(Y,newC);
% end
kfold=5;
indices = crossvalind('Kfold',Y,kfold);
acc=zeros(kfold,6);time=zeros(kfold,6);
for i = 1:kfold
    i
    tsn = (indices == i); % tst indices
    trn = ~tsn; % trning indices

    YT=Y;
    YT(tsn)=-1;
    %     Yext2=Yext;
    %     for r=1:rep
    %         Yext2{r}(tsn)=-1;
    %     end
%     YExt2=YExt;
%     YExt2(tsn)=-1;
%     YExt{1}=YT;
%     YExt{1}(trn)=LabelExt(Y(trn),newC);
%     for r=1:length(A)
%         YLei{r}=YExt{1};
%         YLei{r}(trn)=louvain(A{r}(trn,trn));
% %         YLei2{r}=YExt{1};
% %         YLei2{r}(trn)=louvain(A{r}(trn,trn));
%     end
%     if ll==false
%         Yext=cell(1,rep);
%         for r=1:rep
%             Yext{r}=Y;
%             Yext{r}(trn)=LabelExt(Y(trn),floor(newC/rep*r));
%             Yext{r}(tsn)=-1;
%         end
%     end
    YTrn=Y(trn);
    YTsn=Y(tsn);
%     [Z]=GraphEncoder(X,Yexp);
%     [Z2]=GraphEncoder(X,YT);
%     Z=[Z,Z2];

    for j=1:2
        tic
        switch j
            case 1
                [Z]=GraphEncoder(X,YT);
            case 2
                [Z]=GraphEncoder(X,YT);
                YLei=kmeans(Z,newK*K);
                [Z]=GraphEncoder(X,YLei);
            case 3
                [Z]=GraphEncoder(X,randi([1,newK*K],n,1));
                YTmp=kmeans(Z,newK*K);
                [Z]=GraphEncoder(X,YTmp);
        end
        timeEmd=toc;
        tic
        mdl=fitcdiscr(Z(trn,:),YTrn,'DiscrimType','pseudoLinear');
%         mdl=fitcknn(Z(trn,:),YTrn,'Distance','correlation','NumNeighbors',5);
        [YSTesn,score]=predict(mdl,Z(tsn,:));
        acc(i,j)=mean(YSTesn~=YTsn);
        time(i,j)=toc+timeEmd;

        tic
        Y2=onehotencode(categorical(YTrn),2)';
        %Y2=zeros(length(YTrn),K);
        %for j=1:length(YTrn)
        %    Y2(j,YTrn(j))=1;
        %end
        %Y2Trn=Y2(trn,:);
        mdl3 = train(net1,Z(trn,:)',Y2);
        %     tt = sim(mdl3, Z(tsn,:)') > 0.5;
        classes = mdl3(Z(tsn,:)'); % class-wise probability for tsting data
        %acc_NN = perform(mdl3,Y2Tsn',classes);
        tt = vec2ind(classes)'; % this gives the actual class for each observation
        time(i,j+3)=toc+timeEmd;
        acc(i,j+3)=mean(YTsn~=tt);
    end
end
% mean(acc1)
% mean(acc2)
% mean(time1)
% mean(time2)

result = array2table([mean(acc); std(acc); mean(time); std(time)], 'RowNames', {'Err','Std_Err','Time','Std_Time'},'VariableNames', {'LDA', 'LDA Louvain','LDA Rand', 'NN', 'NN Louvain','NN Rand'});