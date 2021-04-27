%%% generate graph
n=2000; opt=1;
if opt==1 % Bayes optimal being 0
    X=unifrnd(0,0.5,n/2,1);
    Y=unifrnd(0.5,1,n/2,1);
else % Bayes optimal being ?
    X=betarnd(1,3,n/2,1);
    Y=betarnd(3,1,n/2,1);
end
X=[X;Y]; Y=[ones(n/2,1);2*ones(n/2,1)];
A=(unifrnd(0,1,n,n)<(X*X'));
for i=1:n
    A(i,i)=0;
    for j=i+1:n
        A(j,i)=A(i,j);
    end
end
[~,~,Y]=unique(Y); % in case Y is not ordered from 1 to K; unique will correct the ordering

%%% k-fold evaluation
num=10;error=0; t=0;
indices = crossvalind('Kfold',Y,num);
for j=1:num
    tsn = (indices == j); % tst indices
    trn = ~tsn; % trning indices
    ATrn=A(trn,trn);ATsn=A(tsn,trn);
    YTsn=Y(tsn);
    
    %%% one-hot encoding   
    Y2=zeros(n,max(Y));
    for i=1:n
        Y2(i,Y(i))=1;
    end
    Y2Trn=Y2(trn,:);   
    
    % train a simple two-layer net
    net = patternnet(10,'trainscg','crossentropy'); % number of neurons, Scaled Conjugate Gradient, cross entropy
    net.trainParam.showWindow = false; % suppress UI
    % activation function: purelin is identity activation; poslin is relu;
    % tansig is sigmoid
    net.layers{1}.transferFcn = 'purelin';     
    net.layers{2}.transferFcn = 'softmax';
    net.trainParam.epochs=100;
    net.divideParam.trainRatio = 80/100;
    net.divideParam.valRatio   = 20/100;
    net.divideParam.testRatio  = 0/100;
    
    tic
    mdl3 = train(net,ATrn',Y2Trn');
    classes = mdl3(ATsn'); % class-wise probability for testing data
    classes = vec2ind(classes); % this gives the actual class for each observation
    error=error+mean(classes~=YTsn')/num; % classification error 
    t=t+toc/num;
end

%%% Compare to ASE and others
%%% [error_AEE,error_AEE2,error_AEN,error_ASE,t_AEE,t_AEE2,t_AEN,t_ASE]=GraphEncoderEvaluate(double(A),Y)

% A direct NN application is not too lousy! 
% Compare to ASE / Graph NN, they have an error of about 0.02 for uniform, and about 0.12 for beta; 
% ASE has a similar running time of 0.15 sec; 

% Observe that as you increase n, running time is about O(n^2)! while ASE is O(n^2).

