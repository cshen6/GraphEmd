%% generate graph
n=1000; opt=1;
switch opt
    case 1 
        X=unifrnd(0,0.5,n/2,1);
        Y=unifrnd(0.5,1,n/2,1);
        X=[X;Y]; Y=[ones(n/2,1);2*ones(n/2,1)];
        A=double(unifrnd(0,1,n,n)<(X*X'));
        for i=1:n
            A(i,i)=0;
            for j=i+1:n
                A(j,i)=A(i,j);
            end
        end
    case 2 
        X=betarnd(1,3,n/2,1);
        Y=betarnd(3,1,n/2,1);
        X=[X;Y]; Y=[ones(n/2,1);2*ones(n/2,1)];
        A=double(unifrnd(0,1,n,n)<(X*X'));
        for i=1:n
            A(i,i)=0;
            for j=i+1:n
                A(j,i)=A(i,j);
            end
        end
    case 3 % SBM; Bayes optimal being 0
        K=2;
        pp=[0.5,0.5];
        Bl=zeros(K,K);
        %             Bl=rand(clas,clas);
        Bl(:,1)=[0.13,0.1];
        Bl(:,2)=[0.1,0.13];
        A=zeros(n,n);
        tt=rand([n,1]);
        Y=ones(n,1);
        thres=0;
        for i=1:size(pp,2)
            thres=thres+pp(i);
            Y =Y+(tt>thres); %determine the block of each data
        end
        for i=1:n
            A(i,i)=0;%diagonals are zeros
            for j=i+1:n
                A(i,j)=rand(1)<Bl(Y(i),Y(j));
                A(j,i)=A(i,j);
            end
        end
end
%% create edge labels based on vertex labels
edge=adj2edge(A);
s=size(edge,1);
YE=ones(s,1); % edge label
tt=rand([s,1]);
%%% specify label probability p1 and p2 --- what happens if p1=p2?
p1=0.8; 
p2=0.2; 
for i=1:s
    if Y(edge(i,1))==Y(edge(i,2))
        YE(i)=1+(tt(i)>p1);
    else
        YE(i)=1+(tt(i)>p2);
    end
end
edge(:,3)=YE;

%% train a simple two-layer net
net = patternnet(10,'trainscg','crossentropy'); % number of neurons, Scaled Conjugate Gradient, cross entropy
net.trainParam.showWindow = false; % suppress UI
% activation function: purelin is identity activation; poslin is relu;
% tansig is sigmoid
net.layers{1}.transferFcn = 'poslin';
% net.layers{2}.transferFcn = 'softmax';
net.trainParam.epochs=100;
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio   = 20/100;
net.divideParam.testRatio  = 0/100;
Y2=onehotencode(categorical(YE),2);

%% Vertex Embedding
%%% ASE
[U,S,V]=svds(double(A));d=2;
XASE=U(:,1:d)*S(1:d,1:d)^0.5;
XE1=edgeEmd(edge,XASE);
%%% AEE
[XAEE]=GraphEncoder(A,Y);
XE2=edgeEmd(edge,XAEE);
% hold on
% plot(XE2(YE==1,1),XE2(YE==1,3),'ro');
% plot(XE2(YE==2,1),XE2(YE==2,3),'bo');
% hold off

%% k-fold evaluation for edge label classification
num=10;errorASE=0; errorAEE=0;tAEE=0;tASE=0;
indices = crossvalind('Kfold',YE,num);
for j=1:num
    tsn = (indices == j); % tst indices
    trn = ~tsn; % trning indices
    XTrn=XE1(trn,:);XTsn=XE1(tsn,:);
    YTsn=YE(tsn);
    Y2Trn=Y2(trn,:);   
    
    % ASE * NN
    tic
    mdl3 = train(net,XTrn',Y2Trn');
    classes = mdl3(XTsn'); % class-wise probability for testing data
    classes = vec2ind(classes); % this gives the actual class for each observation
    errorASE=errorASE+mean(classes~=YTsn')/num; % classification error 
    tASE=tASE+toc/num;
    
    % AEE * NN
    tic
    XTrn=XE2(trn,:);XTsn=XE2(tsn,:);
    mdl4 = train(net,XTrn',Y2Trn');
    classes = mdl4(XTsn'); % class-wise probability for testing data
    classes = vec2ind(classes); % this gives the actual class for each observation
    errorAEE=errorAEE+mean(classes~=YTsn')/num; % classification error
    tAEE=tAEE+toc/num;
end

%%% For edge label classification, AEE is either similar to or better than ASE in classification error.
%%% Note that we assumed all vertex labels are given, which gives AEE an
%%% advantage in running time and accuracy.
