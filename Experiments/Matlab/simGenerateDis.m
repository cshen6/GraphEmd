function [X,Label]=simGenerateDis(option,n,K,d)
if nargin<3
    K=3;
end
if nargin<4
    d=1;
end
switch option
    case 1 % SBM with 3 classes
        fileName='SBM';
        pp=1/K*ones(K,1);
        Dis=zeros(n,n);
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:K
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        X=zeros(n,d);
        for i=1:K
            X(Label==i,:)=unifrnd(0+i,1+i,[sum(Label==i),d]);
        end
end