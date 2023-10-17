function [X,Label]=simGenerateDis(option,n,K,p)
if nargin<3
    K=3;
end
if nargin<4
    p=100;
end
switch option
    case 0 % 1D Uniform [3k,3+3k] per class
%         fileName='SBM';
        pp=1/K*ones(K,1);
%         Dis=zeros(n,n);
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:K
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        X=zeros(n,p);
%         if p>=K
            for i=1:p
                X(Label==(mod(i,p)+1),i)=mvnrnd(10,1,sum(Label==(mod(i,p)+1)));
            end
%         else
%             for i=1:p
%                 X(Label==mod(i,K),i)=mvnrnd(10,1,sum(Label==i));
%             end
%         end
    case 1 % 1D Uniform [3k,3+3k] per class
%         fileName='SBM';
        pp=1/K*ones(K,1);
%         Dis=zeros(n,n);
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:K
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        X=zeros(n,1);
        for i=1:K
            X(Label==i,1)=unifrnd(0+3*i,3+3*i,[sum(Label==i),1]);
        end
    case 2 % K-D Uniform [0,3] on each axis per class
%         fileName='SBM';
        pp=1/K*ones(K,1);
%         Dis=zeros(n,n);
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:K
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        X=zeros(n,K);
        for i=1:K
            X(Label==i,i)=unifrnd(0,3,[sum(Label==i),1]);
        end
    case 3 % HD Uniform, with K-D + Noise
%         fileName='SBM';
        pp=1/K*ones(K,1);
%         Dis=zeros(n,n);
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:K
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        X=zeros(n,p);
        for i=1:K
            X(Label==i,i)=unifrnd(1,3,[sum(Label==i),1]);
            for j=1:p
                if j ~= i
                    X(Label==i,j)=unifrnd(0,1,[sum(Label==i),1]);
                end
            end
        end
    case 4 % K-D Gaussian on each axis per class
%         fileName='SBM';
        pp=1/K*ones(K,1);
%         Dis=zeros(n,n);
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:K
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        X=zeros(n,1);
        for i=1:K
            X(Label==i,1)=mvnrnd(0+3*i,1,sum(Label==i));
        end
    case 5 % K-D Gaussian on each axis per class
%         fileName='SBM';
        pp=1/K*ones(K,1);
%         Dis=zeros(n,n);
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:K
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        X=zeros(n,K);
        for i=1:K
            X(Label==i,i)=mvnrnd(2,1,sum(Label==i));
        end
    case 6 % HD Gaussian, with K-D + Noise
%         fileName='SBM';
        pp=1/K*ones(K,1);
%         Dis=zeros(n,n);
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:K
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        X=zeros(n,p);
        for i=1:K
            X(Label==i,i)=mvnrnd(5,1,sum(Label==i));
            for j=1:p
                if j ~= i
                     X(Label==i,j)=mvnrnd(1,1,sum(Label==i));
                end
            end
        end
    case 100 % SBM with 3 classes
        fileName='SBM';
        p=3;
        bd=0.2; %0.13 at n=2000;0.12 at n=5000
        pp=[0.33,0.33,0.34];
        Bl=zeros(p,p);
        %             Bl=rand(clas,clas);
        Bl(:,1)=[bd,0.1,0.1];
        Bl(:,2)=[0.1,bd,0.1];
        Bl(:,3)=[0.1,0.1,bd];
        Dis=zeros(n,n);
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        W=unifrnd(0,10,[n,n]);
        for i=1:size(pp,2)
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        for i=1:n
            Dis(i,i)=0;%diagonals are zeros
            for j=i+1:n
                Dis(i,j)=W(i,j)*(rand(1)<Bl(Label(i),Label(j)));
                Dis(j,i)=Dis(i,j);
            end
        end
        X=Dis;
end