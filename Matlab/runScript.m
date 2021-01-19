% RDPG increase training / reduce testing
error1=zeros(5,1);error20=zeros(5,1);std1=zeros(5,1);std2=zeros(5,1);
rn=1;n=50;
[error1(1),error20(1),time1,time2,std1(1),std2(1)]=GraphNNSim(2,rn,n,10)
[error1(2),error20(2),time1,time2,std1(2),std2(2)]=GraphNNSim(2,rn,n,5)
[error1(3),error20(3),time1,time2,std1(3),std2(3)]=GraphNNSim(2,rn,n,2)

% EN vs LDA, and speed advantage
lim=10;rn=10;n=50;
dimRange=10:10:100;
errorEn=zeros(lim,1);errorLDA=zeros(lim,1);errorSVM=zeros(lim,1);errorRF=zeros(lim,1);
stdEn=zeros(lim,1);stdLDA=zeros(lim,1);stdSVM=zeros(lim,1);stdRF=zeros(lim,1);
tEn=zeros(lim,1);tLDA=zeros(lim,1);tSVM=zeros(lim,1);tRF=zeros(lim,1);
for i=1:lim
    d=dimRange(i);
    [errorEn(i),errorLDA(i),errorSVM(i),errorRF(i),tEn(i),tLDA(i),tSVM(i),tRF(i),stdEn(i),stdLDA(i),stdSVM(i),stdRF(i)]=GraphNNSim(3,rn,n,d);
end

% %% semi-supervised
% [dis,Y,d,X]=generateSims(3,100,2);
% [error1,error2,error3]=GraphNNLearn(X,Y);


%%% WRT Dimension
opt=1;
n=200;
dr=200;
d=2;
rep=100;
[adj,Y,d,X]=generateSims(opt,n,d);
% per=[find(Label==0);find(Label==1)];
% n1=length(find(Label==0));
% adj=adj(per,per);
% Y=Label(per);
% tic
[U,S,~]=svd(adj);
X=U(:,1:d)*S(1:d,1:d);
% toc
% plot(X(1:n1,1),X(1:n1,2),'r.')
% hold on
% plot(X(n1+1:n,1),X(n1+1:n,2),'b.')
% hold off
error00 = GraphNNLearn(X,Y);
error01 = graphC(X,Y);

% tic
[Z,filter]=GraphFilter(adj,Y);
% toc
error10 = GraphNNLearn(Z,Y);
error11 = graphC(Z,Y);
% plot(Z(1:n1,1),Z(1:n1,2),'r.')
% hold on
% plot(Z(n1+1:n,1),Z(n1+1:n,2),'b.')
% hold off
error20=zeros(dr,rep);
error21=zeros(dr,rep);
for rn=1:rep
    V=unifrnd(0,1,n,dr);
    %[U,S,V]=svd(unifrnd(0,1,n,n));
    for i=1:dr
        W=adj*V(:,1:i);
        error20(i,rn) = GraphNNLearn(W,Y);
        error21(i,rn) = graphC(W,Y);
    end
end
error30=zeros(dr,rep);
error31=zeros(dr,rep);
for rn=1:rep
    %V=unifrnd(0,1,n,n);
    %[U,S,V]=svd(unifrnd(0,1,n,n));
    V=randi(2,n,dr);
    for i=1:dr
        W=adj*V(:,1:i);
        error30(i,rn) = GraphNNLearn(W,Y);
        error31(i,rn) = graphC(W,Y);
    end
end
error40=zeros(dr,rep);
error41=zeros(dr,rep);
for rn=1:rep
    %V=unifrnd(0,1,n,n);
    [U,S,V]=svd(unifrnd(0,1,n,n));
    %V=randi(2,n,dr);
    for i=1:dr
        W=adj*V(:,1:i);
        error40(i,rn) = GraphNNLearn(W,Y);
        error41(i,rn) = graphC(W,Y);
    end
end
hold on
plot(1:dr,error00*ones(dr,1),'b-',1:dr,error01*ones(dr,1),'b:','LineWidth',2);
plot(1:dr,error10*ones(dr,1),'g-',1:dr,error11*ones(dr,1),'g:','LineWidth',2);
errorbar(1:dr,mean(error20,2),std(error20,[],2),'r-','LineWidth',2);
errorbar(1:dr,mean(error21,2),std(error21,[],2),'r:','LineWidth',2);
errorbar(1:dr,mean(error30,2),std(error30,[],2),'c-','LineWidth',2);
errorbar(1:dr,mean(error31,2),std(error31,[],2),'c:','LineWidth',2);
errorbar(1:dr,mean(error40,2),std(error40,[],2),'m-','LineWidth',2);
errorbar(1:dr,mean(error41,2),std(error41,[],2),'m:','LineWidth',2);
hold off
legend('ASE + NN','ASE + LDA','Encoder + NN','Encoder + LDA','Random Dense + NN','Random Dense + LDA','Random Sparse + NN','Random Sparse + LDA','Random Ortho + NN','Random Ortho + LDA')
xlabel('Dimension')
ylabel('10-Fold Error')


%%% WRT Sample Size
opt=1;
n=500;
nn=50:50:500;
dr=50;
d=2;
ns=10;
rep=100;
error00=zeros(ns,rep);
error01=zeros(ns,rep);
error10=zeros(ns,rep);
error11=zeros(ns,rep);
error20=zeros(ns,rep);
error21=zeros(ns,rep);
error30=zeros(ns,rep);
error31=zeros(ns,rep);
error40=zeros(ns,rep);
error41=zeros(ns,rep);

for r=1:rep
    [adj,Y,~,X]=generateSims(opt,n);
    for rn=1:ns
        nt=nn(rn);
        tmpX=adj(1:nt,1:nt);
        tmpY=Y(1:nt);
        % per=[find(Label==0);find(Label==1)];
        % n1=length(find(Label==0));
        % adj=adj(per,per);
        % Y=Label(per);
        % tic
        [U,S,~]=svd(tmpX);
        X=U(:,1:d)*S(1:d,1:d);
        % toc
        % plot(X(1:n1,1),X(1:n1,2),'r.')
        % hold on
        % plot(X(n1+1:n,1),X(n1+1:n,2),'b.')
        % hold off
        error00(rn,r) = GraphNNLearn(X,tmpY);
        error01(rn,r) = graphC(X,tmpY);
        
        % tic
        [Z,filter]=GraphFilter(tmpX,tmpY);
        % toc
        error10(rn,r) = GraphNNLearn(Z,tmpY);
        error11(rn,r) = graphC(Z,tmpY);
        % plot(Z(1:n1,1),Z(1:n1,2),'r.')
        % hold on
        % plot(Z(n1+1:n,1),Z(n1+1:n,2),'b.')
        % hold off
    end
    
    V=unifrnd(0,1,n,dr);
    for rn=1:ns
        nt=nn(rn);
        tmpX=adj(1:nt,1:nt);
        tmpY=Y(1:nt);
        W=tmpX*V(1:nt,:);
        error20(rn,r) = GraphNNLearn(W,tmpY);
        error21(rn,r) = graphC(W,tmpY);
    end
    V=randi(2,n,dr);
    for rn=1:ns
        nt=nn(rn);
        tmpX=adj(1:nt,1:nt);
        tmpY=Y(1:nt);
        W=tmpX*V(1:nt,:);
        error30(rn,r) = GraphNNLearn(W,tmpY);
        error31(rn,r) = graphC(W,tmpY);
    end
    [U,S,V]=svd(unifrnd(0,1,n,n));
    for rn=1:ns
        nt=nn(rn);
        tmpX=adj(1:nt,1:nt);
        tmpY=Y(1:nt);
        W=tmpX*V(1:nt,:);
        error40(rn,r) = GraphNNLearn(W,tmpY);
        error41(rn,r) = graphC(W,tmpY);
    end
end
hold on
errorbar(nn,mean(error00,2),std(error00,[],2),'b-','LineWidth',2);
errorbar(nn,mean(error01,2),std(error01,[],2),'b:','LineWidth',2);
errorbar(nn,mean(error10,2),std(error10,[],2),'g-','LineWidth',2);
errorbar(nn,mean(error11,2),std(error11,[],2),'g:','LineWidth',2);
errorbar(nn,mean(error20,2),std(error20,[],2),'r-','LineWidth',2);
errorbar(nn,mean(error21,2),std(error21,[],2),'r:','LineWidth',2);
errorbar(nn,mean(error30,2),std(error30,[],2),'c-','LineWidth',2);
errorbar(nn,mean(error31,2),std(error31,[],2),'c:','LineWidth',2);
errorbar(nn,mean(error40,2),std(error40,[],2),'m-','LineWidth',2);
errorbar(nn,mean(error41,2),std(error41,[],2),'m:','LineWidth',2);
hold off
legend('ASE + NN','ASE + LDA','Encoder + NN','Encoder + LDA','Random Dense + NN','Random Dense + LDA','Random Sparse + NN','Random Sparse + LDA','Random Ortho + NN','Random Ortho + LDA')
xlabel('Sample Size')
ylabel('10-Fold Error')















%%% MNIST: Not used
metric='cosine';
XTr=squareform(pdist(XTrn',metric));
[mdl,filter]=GraphNN(XTr,Y1);
XTs=pdist2(XTsn',XTrn',metric);
[label]=GraphNNPredict(XTs,mdl,filter);
error=mean(Y2~=label);

%%% COIL20: Not used
load('COIL20.mat')
tic
error0=GraphNNEvaluate(fea,gnd);
toc
tic
error1=GraphSVM(fea,gnd,0);
toc
tic
error20=GraphSVM(fea,gnd,3);
toc

% C-elegans: TBA
load('graphCElegans.mat')
tic
error0=GraphNNEvaluate(Ac,vcols);
toc
tic
error1=GraphSVM(Ac,vcols,2,30);
toc

% Wikipedia: TBA
load('Wiki_Data.mat')
tic
errorTE0=GraphNNEvaluate(TE,Label);
toc
tic
errorTE1=GraphSVM(TE,Label,1,30);
toc

load('polblogs.mat')
tic
error0=GraphNNEvaluate(Adj,Label);
toc
tic
error1=GraphSVM(Adj,Label,2,5);
toc
