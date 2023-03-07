function [Dis,Label,d,X]=simGenerate(option,n,d,edge)
if nargin<3
    d=10;
end
if nargin<4
    edge=0;
end
repeat=1;
switch option
    case 10 % SBM with 3 classes
        fileName='SBM';
        d=3;
        bd=0.13; %0.13 at n=2000;0.12 at n=5000
        pp=[0.2,0.3,0.5];
        Bl=zeros(d,d);
        %             Bl=rand(clas,clas);
        Bl(:,1)=[bd,0.1,0.1];
        Bl(:,2)=[0.1,bd,0.1];
        Bl(:,3)=[0.1,0.1,bd];
        Dis=zeros(n,n);
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:size(pp,2)
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        for i=1:n
            Dis(i,i)=0;%diagonals are zeros
            for j=i+1:n
                Dis(i,j)=rand(1)<Bl(Label(i),Label(j));
                Dis(j,i)=Dis(i,j);
            end
        end
        X=Dis;
    case 11 % SBM with 5 classes
        fileName='SBM';
        Dis=zeros(n,n);
        pp=1/d*ones(d,1);
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:d
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end       
        Bl=0.1*ones(d,d);
        %             Bl=rand(clas,clas);
        for i=1:d
            Bl(i,i)=0.2;%0.2 at n=2000;
        end
        for i=1:n
            Dis(i,i)=0;%diagonals are zeros
            for j=i+1:n
                Dis(i,j)=rand(1)<Bl(Label(i),Label(j));
                Dis(j,i)=Dis(i,j);
            end
        end
        X=Dis;
%         if (contam==0)
%             titleStr='No Contamination';
%         else
%             if contam<1
%                 ind=unifrnd(0,1,3,n);
%                 ind=(ind>1-contam);
%                 for i=1:n
%                     Dis(i,ind(Label(i),:))=0;
%                 end
%                 titleStr='With Fixed Contamination';
%             else
%                 ind=unifrnd(0,1,n,n);
%                 ind=(ind>(contam-1));
%                 Dis(ind)=0;
%                 titleStr='With Random Contamination';
%             end
%         end
     case 12 % SBM with 3 classes
        fileName='SBM';
        d=2;
        pp=[0.5,0.5];
        Bl=zeros(d,d);
        %             Bl=rand(clas,clas);
        Bl(:,1)=[0.13,0.1];
        Bl(:,2)=[0.1,0.13];
        Dis=zeros(n,n);
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:size(pp,2)
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        for i=1:n
            Dis(i,i)=0;%diagonals are zeros
            for j=i+1:n
                Dis(i,j)=rand(1)<Bl(Label(i),Label(j));
                Dis(j,i)=Dis(i,j);
            end
        end
        X=Dis;
    case 13 % SBM with 3 classes
        fileName='SBM';
        d=3;
        bd=0.13; %0.13 at n=2000;
        pp=[1/3,1/3,1/3];
        Bl=zeros(d,d);
        %             Bl=rand(clas,clas);
        Bl(:,1)=[0.1,0.01,0.05];
        Bl(:,2)=[0.01,0.1,0.025];
        Bl(:,3)=[0.05,0.025,0.15];
        Dis=zeros(n,n);
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:size(pp,2)
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        for i=1:n
            Dis(i,i)=0;%diagonals are zeros
            for j=i+1:n
                Dis(i,j)=rand(1)<Bl(Label(i),Label(j));
                Dis(j,i)=Dis(i,j);
            end
        end
        X=Dis;
    case 15 % SBM with 2 classes
        fileName='SBM';
        d=2;
        pp=[0.5,0.5];
        Bl=zeros(d,d);
        %             Bl=rand(clas,clas);
        Bl(:,1)=[0.1,0.05];
        Bl(:,2)=[0.05,0.1];
        Dis=zeros(n,n);
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:size(pp,2)
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        for i=1:n
            Dis(i,i)=0;%diagonals are zeros
            for j=i+1:n
                Dis(i,j)=rand(1)<Bl(Label(i),Label(j));
                Dis(j,i)=Dis(i,j);
            end
        end
        X=Dis;
     case 16 % SBM with 2 classes
        fileName='SBM';
        d=2;
        pp=[0.5,0.5];
        Bl=zeros(d,d);
        %             Bl=rand(clas,clas);
        Bl(:,1)=[0.1,0.2];
        Bl(:,2)=[0.2,0.1];
        Dis=zeros(n,n);
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:size(pp,2)
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        for i=1:n
            Dis(i,i)=0;%diagonals are zeros
            for j=i+1:n
                Dis(i,j)=rand(1)<Bl(Label(i),Label(j));
                Dis(j,i)=Dis(i,j);
            end
        end
        X=Dis;
      case 17 % SBM with 2 classes
        fileName='SBM';
        d=2;
        pp=[0.5,0.5];
        Bl=zeros(d,d);
        %             Bl=rand(clas,clas);
        Bl(:,1)=[0.05,0.05];
        Bl(:,2)=[0.05,0.2];
        Dis=zeros(n,n);
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:size(pp,2)
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        for i=1:n
            Dis(i,i)=0;%diagonals are zeros
            for j=i+1:n
                Dis(i,j)=rand(1)<Bl(Label(i),Label(j));
                Dis(j,i)=Dis(i,j);
            end
        end
        X=Dis;
     case 18 % SBM with 2 classes
        fileName='SBM';
        d=4;
        pp=[0.2,0.3,0.2,0.3];
        Dis=cell(1,3);
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:size(pp,2)
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        for k=1:3
            Bl=0.1*ones(d,d);
            %             Bl=rand(clas,clas);
            Bl(k,k)=0.2;
            Dis{k}=zeros(n,n);
            for i=1:n
                Dis{k}(i,i)=0;%diagonals are zeros
                for j=i+1:n
                    Dis{k}(i,j)=rand(1)<Bl(Label(i),Label(j));
                    Dis{k}(j,i)=Dis{k}(i,j);
                end
            end
        end
        X=Dis;
    case 19 % SBM with 2 classes
        fileName='SBM';
        d=4;
        pp=[0.2,0.3,0.2,0.3];
        Dis=cell(1,10);
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:size(pp,2)
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        Bl=0.1*ones(d,d);
        %             Bl=rand(clas,clas);
        for k=1:4
            Bl(k,k)=0.2;
        end
        for k=1:10
            Bl(1,1)=Bl(1,1)+0.01*(k-1);
            Bl(1,2)=Bl(1,2)-0.005*(k-1);
            Bl(2,1)=Bl(1,2);
            Bl(2,2)=Bl(2,2)-0.005*(k-1);
            Dis{k}=zeros(n,n);
            for i=1:n
                Dis{k}(i,i)=0;%diagonals are zeros
                for j=i+1:n
                    Dis{k}(i,j)=rand(1)<Bl(Label(i),Label(j));
                    Dis{k}(j,i)=Dis{k}(i,j);
                end
            end
        end
        X=Dis;
     case 20 % DC-SBM with 3 classes
        fileName='DCSBM';
        pp=[0.2,0.3,0.5];
        d=3;
        Bl=zeros(d,d);
        %             Bl=rand(clas,clas);
        Bl(:,1)=[0.9,0.1,0.1];
        Bl(:,2)=[0.1,0.5,0.1];
        Bl(:,3)=[0.1,0.1,0.2];
        if edge==0
           Dis=zeros(n,n);
        else
            Dis=[];
        end
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        theta=betarnd(1,4,n,1);
        % theta=unifrnd(0.5,1.5,n,1);
%         theta=theta;
        for i=1:d
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        if edge==0
            for i=1:n
                Dis(i,i)=0;%diagonals are zeros
                for j=i+1:n
                    weight=1;%randi(10);
                    Dis(i,j)=weight*(rand(1)<theta(i)*theta(j)*Bl(Label(i),Label(j)));
                    Dis(j,i)=Dis(i,j);
                end
            end
        else
            for i=1:n
                for j=i+1:n
                    tmp=(rand(1)<theta(i)*theta(j)*Bl(Label(i),Label(j)));
                    if tmp==1;
                        Dis=[Dis;i,j,1];
                    end
                end
            end
        end
        X=Dis;
     case 21 % DC-SBM with 10 classes
        fileName='DCSBM';
        pp=1/d*ones(d,1);
        Bl=0.1*ones(d,d);
        %             Bl=rand(clas,clas);
        for i=1:d
            Bl(i,i)=0.9;
        end
        if edge==0
           Dis=zeros(n,n);
        else
            Dis=[];
        end
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
%         theta=betarnd(1,10,n,1);
%         ind=(Label==2);
%         theta(ind)= betarnd(10,10,sum(ind),1);
%         ind=(Label==3);
%         theta(ind)= betarnd(10,1,sum(ind),1);
        theta=betarnd(1,4,n,1);
%         theta=theta;
        for i=1:d
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        if edge==0
            for i=1:n
                Dis(i,i)=0;%diagonals are zeros
                for j=i+1:n
                    weight=1;%randi(10);
                    Dis(i,j)=weight*(rand(1)<theta(i)*theta(j)*Bl(Label(i),Label(j)));
                    Dis(j,i)=Dis(i,j);
                end
            end
        else
            for i=1:n
                for j=i+1:n
                    tmp=weight*(rand(1)<theta(i)*theta(j)*Bl(Label(i),Label(j)));
                    if tmp==1;
                        Dis=[Dis;i,j,1];
                    end
                end
            end
        end
        X=Dis;
   case 22 % DC-SBM with 4 classes
        fileName='DCSBM';
        pp=[0.2,0.2,0.3,0.3];
        d=4;
        Bl=zeros(d,d);
        %             Bl=rand(clas,clas);
        Bl(:,1)=[0.3,0.1,0.1,0.1];
        Bl(:,2)=[0.1,0.5,0.1,0.1];
        Bl(:,3)=[0.1,0.1,0.7,0.1];
        Bl(:,4)=[0.1,0.1,0.1,0.9];
        Dis=zeros(n,n);
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        theta=betarnd(1,4,n,1);
        %theta=unifrnd(0.05,0.35,n,1);
%         theta=theta;
        for i=1:d
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        for i=1:n
            Dis(i,i)=0;%diagonals are zeros
            for j=i+1:n
                Dis(i,j)=rand(1)<theta(i)*theta(j)*Bl(Label(i),Label(j));
                Dis(j,i)=Dis(i,j);
            end
        end
        X=theta;
    case 25 % SBM with 2 classes, better for normalize
        fileName='SBM';
        d=2;
        pp=[0.5,0.5];
        Bl=zeros(d,d);
        %             Bl=rand(clas,clas);
        Bl(:,1)=[0.5,0.1];
        Bl(:,2)=[0.1,0.5];
        Dis=zeros(n,n);
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        theta=betarnd(1,4,n,1);
        %theta=unifrnd(0.1,0.5,n,1);%randi(10,n,1)/10;%betarnd(1,4,n,1); %0.2*ones(n,1);%
        for i=1:size(pp,2)
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        for i=1:n
            Dis(i,i)=0;%diagonals are zeros
            for j=i+1:n
                Dis(i,j)=rand(1)<theta(i)*theta(j)*Bl(Label(i),Label(j));
                Dis(j,i)=Dis(i,j);
            end
        end
        X=theta;
     case 26 % SBM with 2 classes
        fileName='SBM';
        d=2;
        pp=[0.2,0.8];
        Bl=zeros(d,d);
        %             Bl=rand(clas,clas);
        Bl(:,1)=[0.1,0.2];
        Bl(:,2)=[0.2,0.1];
        Dis=zeros(n,n);
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        theta=unifrnd(0.1,0.5,n,1);%randi(10,n,1)/10;%betarnd(1,4,n,1); %0.2*ones(n,1);%
        for i=1:size(pp,2)
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        for i=1:n
            Dis(i,i)=0;%diagonals are zeros
            for j=i+1:n
                Dis(i,j)=rand(1)<theta(i)*theta(j)*Bl(Label(i),Label(j));
                Dis(j,i)=Dis(i,j);
            end
        end
        X=theta;
      case 27 % SBM with 2 classes, better for no normalize
        fileName='SBM';
        d=2;
        pp=[0.5,0.5];
        Bl=zeros(d,d);
        %             Bl=rand(clas,clas);
        Bl(:,1)=[0.1,0.1];
        Bl(:,2)=[0.1,0.5];
        Dis=zeros(n,n);
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        theta=unifrnd(0.1,0.5,n,1);%randi(10,n,1)/10;%betarnd(1,4,n,1); %0.2*ones(n,1);%
        for i=1:size(pp,2)
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        for i=1:n
            Dis(i,i)=0;%diagonals are zeros
            for j=i+1:n
                Dis(i,j)=rand(1)<theta(i)*theta(j)*Bl(Label(i),Label(j));
                Dis(j,i)=Dis(i,j);
            end
        end
        X=theta;
     case 28 % SBM with 2 classes
        fileName='SBM';
        d=4;
        pp=[0.2,0.3,0.2,0.3];
        Dis=cell(1,3);
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        theta=unifrnd(0.1,0.5,n,1);%randi(10,n,1)/10;%betarnd(1,4,n,1); %0.2*ones(n,1);%
        for i=1:size(pp,2)
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        for k=1:3
            Bl=0.1*ones(d,d);
            %             Bl=rand(clas,clas);
            Bl(k,k)=0.5;
            Dis{k}=zeros(n,n);
            for i=1:n
                Dis{k}(i,i)=0;%diagonals are zeros
                for j=i+1:n
                    Dis{k}(i,j)=rand(1)<theta(i)*theta(j)*Bl(Label(i),Label(j));
                    Dis{k}(j,i)=Dis{k}(i,j);
                end
            end
        end
        X=Dis;
    case 29 % SBM with 2 classes
        fileName='SBM';
        d=4;
        pp=[0.2,0.3,0.2,0.3];
        Dis=cell(1,10);
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        theta=unifrnd(0.1,0.5,n,1);%randi(10,n,1)/10;%betarnd(1,4,n,1); %0.2*ones(n,1);%
        for i=1:size(pp,2)
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        Bl=0.1*ones(d,d);
        %             Bl=rand(clas,clas);
        for k=1:4
            Bl(k,k)=0.2;
        end
        for k=1:10
            Bl(1,1)=Bl(1,1)+0.01*(k-1);
            Bl(1,2)=Bl(1,2)-0.005*(k-1);
            Bl(2,1)=Bl(1,2);
            Bl(2,2)=Bl(2,2)-0.005*(k-1);
            Dis{k}=zeros(n,n);
            for i=1:n
                Dis{k}(i,i)=0;%diagonals are zeros
                for j=i+1:n
                    Dis{k}(i,j)=rand(1)<(theta(i)*theta(j)*Bl(Label(i),Label(j)));
                    Dis{k}(j,i)=Dis{k}(i,j);
                end
            end
        end
        X=Dis;
    case 30 % RDPG
        fileName='RDPG';
        p=1;
        d=3;
        pp=[0.2,0.3,0.5];
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:d
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        X = betarnd(1,5,n,p);
        ind=(Label==2);
        X(ind,:)= betarnd(5,5,sum(ind),p);
        ind=(Label==3);
        X(ind,:)= betarnd(5,1,sum(ind),p);
        A=X*X';
        Dis=zeros(n,n);
        for i=1:n
            Dis(i,i)=0;%diagonals are zeros
            for j=i+1:n
                Dis(i,j)=rand(1)<A(i,j);
                Dis(j,i)=Dis(i,j);
            end
        end
%         Dis=double(rand(n,n)<A);
        d=p;
    case 31 % RDPG
        fileName='RDPG';
        p=1;
        pp=1/d*ones(d,1);
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:d
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        %X = betarnd(1,1,n,p);
        X = 0.2+randn(n,1)*0.01;
        for i=2:d
            ind=(Label==i);
            %X(ind,:)= betarnd(1,i,sum(ind),p);
            X(ind,:)= X(ind,:)+0.5/d*(i-1);
        end
        A=X*X';
        for i=1:n
            Dis(i,i)=0;%diagonals are zeros
            for j=i+1:n
                Dis(i,j)=rand(1)<A(i,j);
                Dis(j,i)=Dis(i,j);
            end
        end
%         Dis=double(rand(n,n)<A);
        d=p;
%         X=Dis;
    case 32 % RDPG
        fileName='RDPG';
        p=1;
        d=2;
        pp=[0.5,0.5];
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:d
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        X = betarnd(1,5,n,p);
%         ind=(Label==2);
%         X(ind,:)= betarnd(5,5,sum(ind),p);
        ind=(Label==2);
        X(ind,:)= betarnd(5,1,sum(ind),p);
%         X = mvnrnd(zeros(d,1),eye(d),n);
%         X(ind,:) = mvnrnd(ones(d,1),eye(d),sum(ind));
        A=X*X';
        for i=1:n
            Dis(i,i)=0;%diagonals are zeros
            for j=i+1:n
                Dis(i,j)=rand(1)<A(i,j);
                Dis(j,i)=Dis(i,j);
            end
        end
%         Dis=double(rand(n,n)<A);
        d=p;
    case 35 % RDPG
        fileName='RDPG';
        p=1;
        d=2;
        pp=[0.5,0.5];
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:d
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        X = betarnd(2,3,n,p);
%         ind=(Label==2);
%         X(ind,:)= betarnd(5,5,sum(ind),p);
        ind=(Label==2);
        X(ind,:)= betarnd(3,2,sum(ind),p);
%         X = 0.2*ones(n,1);
%         ind=(Label==2);
%         X(ind,:)= 0.3*ones(sum(ind),1);
        A=X*X';
        for i=1:n
            Dis(i,i)=0;%diagonals are zeros
            for j=i+1:n
                Dis(i,j)=rand(1)<A(i,j);
                Dis(j,i)=Dis(i,j);
            end
        end
        d=p;
    case 36 % RDPG
        fileName='RDPG';
        p=1;
        d=2;
        pp=[0.5,0.5];
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:d
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        X = unifrnd(0.15,0.25,n,p);
%         ind=(Label==2);
%         X(ind,:)= betarnd(5,5,sum(ind),p);
        ind=(Label==2);
        X(ind,:)= unifrnd(0.1,0.2,sum(ind),p);
%         X = randi(2,n,1)/10;
%         ind=(Label==2);
%         X(ind,:)= 0.2+randi(2,sum(ind),1)/10;
        A=X*X';
        for i=1:n
            Dis(i,i)=0;%diagonals are zeros
            for j=i+1:n
                Dis(i,j)=rand(1)<A(i,j);
                Dis(j,i)=Dis(i,j);
            end
        end
        d=p;
    case 37 % RDPG
        fileName='RDPG';
        p=1;
        d=2;
        pp=[0.5,0.5];
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:d
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        X = mvnrnd(15,1,n,p)/100;
%         ind=(Label==2);
%         X(ind,:)= betarnd(5,5,sum(ind),p);
        ind=(Label==2);
        X(ind,:)= mvnrnd(20,3,sum(ind),p)/100;
%         X=zeros(n,2);
%         X(:,1) = 0+randi(2,n,1)/10;
%         X(:,2) = 0.1+randi(2,n,1)/10;
%         ind=(Label==2);
%         X(ind,1)= 0.3+randi(2,sum(ind),1)/10;
%         X(ind,2)= 0+randi(2,sum(ind),1)/10;
        A=X*X';
        for i=1:n
            Dis(i,i)=0;%diagonals are zeros
            for j=i+1:n
                Dis(i,j)=rand(1)<A(i,j);
                Dis(j,i)=Dis(i,j);
            end
        end
        d=p;
    case 40 % RDPG with dist 
        fileName='RDPG';
        p=1;
        d=3;
        pp=[0.2,0.3,0.5];
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:d
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        X = betarnd(1,5,n,p);
        ind=(Label==2);
        X(ind,:)= betarnd(5,5,sum(ind),p);
        ind=(Label==3);
        X(ind,:)= betarnd(5,1,sum(ind),p);
        Dis=squareform(pdist(X));
%         A=exp(-squareform(pdist(X)))/4;
%         Dis=double(rand(n,n)<A);
        d=p;
    case 41 % RDPG
        fileName='RDPG';
        p=1;
        pp=1/d*ones(d,1);
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:d
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        %X = betarnd(1,1,n,p);
        X = 0.2+randn(n,1)*0.02;
        for i=2:d
            ind=(Label==i);
            %X(ind,:)= betarnd(1,i,sum(ind),p);
            X(ind,:)= X(ind,:)+0.5/d*(i-1);
        end
        Dis=squareform(pdist(X));
        d=p;
    case 50 % RDPG with kernel
        fileName='RDPG';
        p=1;d=3;
        pp=[0.2,0.3,0.5];
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:d
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        X = betarnd(1,5,n,p);
        ind=(Label==2);
        X(ind,:)= betarnd(5,5,sum(ind),p);
        ind=(Label==3);
        X(ind,:)= betarnd(5,1,sum(ind),p);
        Dis=exp(-squareform(pdist(X)))/4;
%         Dis=double(rand(n,n)<A);
%         d=p;
    case 51 % RDPG
        fileName='RDPG';
        p=1;
        pp=1/d*ones(d,1);
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:d
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        %X = betarnd(1,1,n,p);
        X = 0.2+randn(n,1)*0.02;
        for i=2:d
            ind=(Label==i);
            %X(ind,:)= betarnd(1,i,sum(ind),p);
            X(ind,:)= X(ind,:)+0.5/d*(i-1);
        end
        Dis=exp(-squareform(pdist(X)))/4;
        d=p;
    case 52 % RDPG with 3 data
        fileName='SBMFusion';
        Dis=zeros(n,n,3);
        pp=[0.2,0.3,0.2,0.3];
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:d
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end       
        Bl=zeros(4,4);
        %             Bl=rand(clas,clas);
        Bl(:,1)=[0.2,0.1,0.1,0.1];
        Bl(:,2)=[0.1,0.1,0.1,0.1];
        Bl(:,3)=[0.1,0.1,0.1,0.1];
        Bl(:,4)=[0.1,0.1,0.1,0.1];
        for i=1:n
            Dis(i,i,1)=0;%diagonals are zeros
            for j=i+1:n
                Dis(i,j,1)=rand(1)<Bl(Label(i),Label(j));
                Dis(j,i,1)=Dis(i,j,1);
            end
        end
        Bl(:,1)=[0.1,0.1,0.1,0.1];
        Bl(:,2)=[0.1,0.2,0.1,0.1];
        Bl(:,3)=[0.1,0.1,0.1,0.1];
        Bl(:,4)=[0.1,0.1,0.1,0.1];
        for i=1:n
            Dis(i,i,2)=0;%diagonals are zeros
            for j=i+1:n
                Dis(i,j,2)=rand(1)<Bl(Label(i),Label(j));
                Dis(j,i,2)=Dis(i,j,2);
            end
        end
        Bl(:,1)=[0.1,0.1,0.1,0.1];
        Bl(:,2)=[0.1,0.1,0.1,0.1];
        Bl(:,3)=[0.1,0.1,0.2,0.1];
        Bl(:,4)=[0.1,0.1,0.1,0.1];
        for i=1:n
            Dis(i,i,3)=0;%diagonals are zeros
            for j=i+1:n
                Dis(i,j,3)=(rand(1)<Bl(Label(i),Label(j)));
                Dis(j,i,3)=Dis(i,j,3);
            end
        end
    case 60 % hierarchy SBM with 3 classes
        fileName='SBM';
        d=10;
        diff=ceil(n/d);n=diff*d;
        bd=0.3; %0.13 at n=2000;0.12 at n=5000
        %pp=1/d*ones(d,1);
        Bl=0.05*ones(d,d);
        Dis=zeros(n,n);
        Label=ones(n,3);
        for i=1:d
            Label((i-1)*diff+1:i*diff,3)=i;
            Bl(i,i)=bd+0.03*d;
        end
        for i=1:d/2;
            Label((i-1)*diff*2+1:i*diff*2,2)=i;
        end
        Label(1:4*diff,1)=1;Label(4*diff+1:end,1)=2;
        for i=1:n
            Dis(i,i)=0;%diagonals are zeros
            for j=i+1:n
                Dis(i,j)=rand(1)<Bl(Label(i,3),Label(j,3));
                Dis(j,i)=Dis(i,j);
            end
        end
        X=Dis;
     case 61 % hierarchy SBM with 3 classes
        fileName='SBM';
        d=10;
        diff=ceil(n/d);n=diff*d;
        bd=0.2; %0.13 at n=2000;0.12 at n=5000
        %pp=1/d*ones(d,1);
        Bl=0.1*ones(d,d);
        Dis=zeros(n,n);
        Label=ones(n,3);
        for i=1:d
            Label((i-1)*diff+1:i*diff,3)=i;
            Bl(i,i)=bd+0.01*d;
        end
        for i=1:d/2;
            Label((i-1)*diff*2+1:i*diff*2,2)=i;
        end
        Label(1:4*diff,1)=1;Label(4*diff+1:end,1)=2;
        for i=1:n
            Dis(i,i)=0;%diagonals are zeros
            for j=i+1:n
                Dis(i,j)=rand(1)<Bl(Label(i,3),Label(j,3));
                Dis(j,i)=Dis(i,j);
            end
        end
        X=Dis;
    case 70 % SBM outlier: 100 vertices global sending outliers
        fileName='SBM';
        diffSBM=d;
        d=20;numOut=100;
        diff=ceil(n/d);n=diff*d;
        bd=0.2; %0.13 at n=2000;0.12 at n=5000
        %pp=1/d*ones(d,1);
        Bl=0.05*ones(d,d);
        Dis=zeros(n,n);
        Label=ones(n,4);
        theta=betarnd(1,4,n,1);
        %theta=ones(n,1);
        for i=1:d
            Label((i-1)*diff+1:i*diff,3)=i;
            Bl(i,i)=bd;
        end
        for i=1:d/2;
            Label((i-1)*diff*2+1:i*diff*2,2)=i;
        end
        Label(1:4*diff,1)=1;Label(4*diff+1:end,1)=2;
        for i=1:n
            for j=1:n
                Dis(i,j)=rand(1)<theta(i)*theta(j)*Bl(Label(i,3),Label(j,3));
%                 Dis(j,i)=Dis(i,j);
            end
            Dis(i,i)=0;%diagonals are zeros
        end
        X=Dis;
        per=randperm(n);
        Label(per(1:numOut),4)=2;
        outlier=find(Label(:,4)==2);
        jend=n;
        %jend=2*length(outlier);
        if repeat==1
            for i=1:n
                for j=1:n
                    Dis(i,j)=rand(1)<theta(i)*theta(j)*Bl(Label(i,3),Label(j,3));
                    %                 Dis(j,i)=Dis(i,j);
                end
                Dis(i,i)=0;%diagonals are zeros
            end
        end
        for i=1:length(outlier)
            for j=1:jend
                Dis(outlier(i),j)=rand(1)<theta(outlier(i))*theta(j)*(Bl(Label(outlier(i),3),Label(j,3))+diffSBM);
            end
            Dis(i,i)=0;%diagonals are zeros
        end
        X=[X,Dis];
        Dis=X;
    case 71 % SBM outlier: 100 vertices global receiving outliers
        fileName='SBM';
        diffSBM=d;
        d=20;numOut=100;
        diff=ceil(n/d);n=diff*d;
        bd=0.2; %0.13 at n=2000;0.12 at n=5000
        %pp=1/d*ones(d,1);
        Bl=0.05*ones(d,d);
        Dis=zeros(n,n);
        Label=ones(n,4);
        theta=betarnd(1,4,n,1);
%         theta=ones(n,1);
        for i=1:d
            Label((i-1)*diff+1:i*diff,3)=i;
            Bl(i,i)=bd;
        end
        for i=1:d/2;
            Label((i-1)*diff*2+1:i*diff*2,2)=i;
        end
        Label(1:4*diff,1)=1;Label(4*diff+1:end,1)=2;
        for i=1:n
            for j=1:n
                Dis(i,j)=rand(1)<theta(i)*theta(j)*Bl(Label(i,3),Label(j,3));
%                 Dis(j,i)=Dis(i,j);
            end
            Dis(i,i)=0;%diagonals are zeros
        end
        X=Dis;
        per=randperm(n);
        Label(per(1:numOut),4)=2;
        outlier=find(Label(:,4)==2);
        if repeat==1
            for i=1:n
                for j=1:n
                    Dis(i,j)=rand(1)<theta(i)*theta(j)*Bl(Label(i,3),Label(j,3));
                    %                 Dis(j,i)=Dis(i,j);
                end
                Dis(i,i)=0;%diagonals are zeros
            end
        end
        jend=n;
        %jend=2*length(outlier);
        for i=1:length(outlier)
            for j=1:jend
                Dis(j,outlier(i))=rand(1)<theta(outlier(i))*theta(j)*(Bl(Label(j,3),Label(outlier(i),3))+diffSBM);
            end
            Dis(j,j)=0;%diagonals are zeros
        end
        X=[X,Dis];
        Dis=X;
    case 72 % SBM outlier: 100 vertices global mixed outliers
        fileName='SBM';
        diffSBM=d;
        d=20;numOut=100;
        diff=ceil(n/d);n=diff*d;
        bd=0.2; %0.13 at n=2000;0.12 at n=5000
        %pp=1/d*ones(d,1);
        Bl=0.05*ones(d,d);
        Dis=zeros(n,n);
        Label=ones(n,4);
        theta=betarnd(1,4,n,1);
%         theta=ones(n,1);
        for i=1:d
            Label((i-1)*diff+1:i*diff,3)=i;
            Bl(i,i)=bd;
        end
        for i=1:d/2;
            Label((i-1)*diff*2+1:i*diff*2,2)=i;
        end
        Label(1:4*diff,1)=1;Label(4*diff+1:end,1)=2;
        for i=1:n
            for j=1:n
                Dis(i,j)=rand(1)<theta(i)*theta(j)*Bl(Label(i,3),Label(j,3));
%                 Dis(j,i)=Dis(i,j);
            end
            Dis(i,i)=0;%diagonals are zeros
        end
        X=Dis;
        per=randperm(n);
        Label(per(1:numOut),4)=2;
        outlier=find(Label(:,4)==2);
        if repeat==1
            for i=1:n
                for j=1:n
                    Dis(i,j)=rand(1)<theta(i)*theta(j)*Bl(Label(i,3),Label(j,3));
                    %                 Dis(j,i)=Dis(i,j);
                end
                Dis(i,i)=0;%diagonals are zeros
            end
        end
        jend=n;
        %jend=2*length(outlier);
        for i=1:length(outlier)
            for j=1:jend
                Dis(j,outlier(i))=rand(1)<theta(j)*theta(outlier(i))*(Bl(Label(j,3),Label(outlier(i),3))+diffSBM);
                Dis(outlier(i),j)=rand(1)<theta(j)*theta(outlier(i))*(Bl(Label(outlier(i),3),Label(j,3))+diffSBM);
            end
            Dis(i,i)=0;%diagonals are zeros
            Dis(j,j)=0;%diagonals are zeros
        end
        X=[X,Dis];
        Dis=X;
    case 73 % SBM outlier: 100 vertices local sending outliers
        fileName='SBM';
        diffSBM=d;
        d=20;numOut=100;
        diff=ceil(n/d);n=diff*d;
        bd=0.2; %0.13 at n=2000;0.12 at n=5000
        %pp=1/d*ones(d,1);
        Bl=0.05*ones(d,d);
        Dis=zeros(n,n);
        Label=ones(n,4);
        theta=betarnd(1,4,n,1);
%         theta=ones(n,1);
        for i=1:d
            Label((i-1)*diff+1:i*diff,3)=i;
            Bl(i,i)=bd;
        end
        for i=1:d/2;
            Label((i-1)*diff*2+1:i*diff*2,2)=i;
        end
        Label(1:4*diff,1)=1;Label(4*diff+1:end,1)=2;
        for i=1:n
            for j=1:n
                Dis(i,j)=rand(1)<theta(i)*theta(j)*Bl(Label(i,3),Label(j,3));
%                 Dis(j,i)=Dis(i,j);
            end
            Dis(i,i)=0;%diagonals are zeros
        end
        X=Dis;
        per=randperm(n);
        Label(per(1:numOut),4)=2;
        outlier=find(Label(:,4)==2);
        %jend=2*length(outlier);
        if repeat==1
            for i=1:n
                for j=1:n
                    Dis(i,j)=rand(1)<theta(i)*theta(j)*Bl(Label(i,3),Label(j,3));
                    %                 Dis(j,i)=Dis(i,j);
                end
                Dis(i,i)=0;%diagonals are zeros
            end
        end
        jend=n;
        per2=randperm(n);
        per2=per2(1:n/5);
        for i=1:length(outlier)
            for j=1:length(per2)
                Dis(outlier(i),per2(j))=rand(1)<theta(outlier(i))*theta(per2(j))*(Bl(Label(outlier(i),3),Label(per2(j),3))+diffSBM);
            end
            Dis(outlier(i),outlier(i))=0;%diagonals are zeros
        end
        X=[X,Dis];
        Dis=X;
    case 74 % SBM outlier: 100 vertices local receiving outliers
        fileName='SBM';
        diffSBM=d;
        d=20;numOut=100;
        diff=ceil(n/d);n=diff*d;
        bd=0.2; %0.13 at n=2000;0.12 at n=5000
        %pp=1/d*ones(d,1);
        Bl=0.05*ones(d,d);
        Dis=zeros(n,n);
        Label=ones(n,4);
        theta=betarnd(1,4,n,1);
%         theta=ones(n,1);
        for i=1:d
            Label((i-1)*diff+1:i*diff,3)=i;
            Bl(i,i)=bd;
        end
        for i=1:d/2;
            Label((i-1)*diff*2+1:i*diff*2,2)=i;
        end
        Label(1:4*diff,1)=1;Label(4*diff+1:end,1)=2;
        for i=1:n
            for j=1:n
                Dis(i,j)=rand(1)<theta(i)*theta(j)*Bl(Label(i,3),Label(j,3));
%                 Dis(j,i)=Dis(i,j);
            end
            Dis(i,i)=0;%diagonals are zeros
        end
        X=Dis;
        per=randperm(n);
        Label(per(1:numOut),4)=2;
        outlier=find(Label(:,4)==2);
        if repeat==1
            for i=1:n
                for j=1:n
                    Dis(i,j)=rand(1)<theta(i)*theta(j)*Bl(Label(i,3),Label(j,3));
                    %                 Dis(j,i)=Dis(i,j);
                end
                Dis(i,i)=0;%diagonals are zeros
            end
        end
        jend=n;
        %jend=2*length(outlier);
        per2=randperm(n);
        per2=per2(1:n/5);
        for i=1:length(outlier)
            for j=1:length(per2)
                Dis(per2(j),outlier(i))=rand(1)<theta(outlier(i))*theta(per2(j))*(Bl(Label(per2(j),3),Label(outlier(i),3))+diffSBM);
            end
            Dis(outlier(i),outlier(i))=0;%diagonals are zeros
        end
        X=[X,Dis];
        Dis=X;
    case 75 % SBM outlier: 100 vertices local mixed outliers
        fileName='SBM';
        diffSBM=d;
        d=20;numOut=100;
        diff=ceil(n/d);n=diff*d;
        bd=0.2; %0.13 at n=2000;0.12 at n=5000
        %pp=1/d*ones(d,1);
        Bl=0.05*ones(d,d);
        Dis=zeros(n,n);
        Label=ones(n,4);
        theta=betarnd(1,4,n,1);
%         theta=ones(n,1);
        for i=1:d
            Label((i-1)*diff+1:i*diff,3)=i;
            Bl(i,i)=bd;
        end
        for i=1:d/2;
            Label((i-1)*diff*2+1:i*diff*2,2)=i;
        end
        Label(1:4*diff,1)=1;Label(4*diff+1:end,1)=2;
        for i=1:n
            for j=1:n
                Dis(i,j)=rand(1)<theta(i)*theta(j)*Bl(Label(i,3),Label(j,3));
%                 Dis(j,i)=Dis(i,j);
            end
            Dis(i,i)=0;%diagonals are zeros
        end
        X=Dis;
        per=randperm(n);
        Label(per(1:numOut),4)=2;
        outlier=find(Label(:,4)==2);
        if repeat==1
            for i=1:n
                for j=1:n
                    Dis(i,j)=rand(1)<theta(i)*theta(j)*Bl(Label(i,3),Label(j,3));
                    %                 Dis(j,i)=Dis(i,j);
                end
                Dis(i,i)=0;%diagonals are zeros
            end
        end
        jend=n;
        per2=randperm(n);per2=per2(1:n/5);
        per3=randperm(n);per3=per3(1:n/5);
        %jend=2*length(outlier);
        for i=1:length(outlier)
            for j=1:length(per2)
                Dis(outlier(i),per2(j))=rand(1)<theta(outlier(i))*theta(per2(j))*(Bl(Label(outlier(i),3),Label(per2(j),3))+diffSBM);
                Dis(per3(j),outlier(i))=rand(1)<theta(outlier(i))*theta(per3(j))*(Bl(Label(per3(j),3),Label(outlier(i),3))+diffSBM);
            end
            Dis(outlier(i),outlier(i))=0;%diagonals are zeros
        end
        X=[X,Dis];
        Dis=X;
    case 76 % SBM outlier: 30 vertices affects muln*30 adjacency
        fileName='SBM';
        diffSBM=d;
        d=5;numOut=100; muln=5;
        diff=ceil(n/d);n=diff*d;
        bd=0.2; %0.13 at n=2000;0.12 at n=5000
        %pp=1/d*ones(d,1);
        Bl=0.05*ones(d,d);
        Dis=zeros(n,n);
        Label=ones(n,4);
%         theta=betarnd(1,4,n,1);
        theta=ones(n,1);
        for i=1:d
            Label((i-1)*diff+1:i*diff,3)=i;
            Bl(i,i)=bd;
        end
%         for i=1:d/2;
%             Label((i-1)*diff*2+1:i*diff*2,2)=i;
%         end
%         Label(1:4*diff,1)=1;Label(4*diff+1:end,1)=2;
        for i=1:n
            for j=1:n
                Dis(i,j)=rand(1)<theta(i)*theta(j)*Bl(Label(i,3),Label(j,3));
%                 Dis(j,i)=Dis(i,j);
            end
            Dis(i,i)=0;%diagonals are zeros
        end
        X=Dis;
        per=randperm(n);
        Label(per(1:numOut),4)=2;
        outlier=find(Label(:,4)==2);
        jend=n;
%         if repeat==1
            for i=1:n
                for j=1:n
                    Dis(i,j)=rand(1)<theta(i)*theta(j)*Bl(Label(i,3),Label(j,3));
                    %                 Dis(j,i)=Dis(i,j);
                end
                Dis(i,i)=0;%diagonals are zeros
            end
%         end
%         jend=muln*length(outlier);
        for i=1:length(outlier)
            for j=1:jend
                Dis(outlier(i),j)=rand(1)<theta(outlier(i))*theta(j)*(Bl(Label(outlier(i),3),Label(j,3))+diffSBM);
            end
            Dis(outlier(i),outlier(i))=0;%diagonals are zeros
        end
        X=[X,Dis];
        Dis=X;
     case 77 % SBM outlier: 30 vertices affects muln*30 adjacency
        fileName='SBM';
        diffSBM=d;
        d=5;numOut=100; muln=5;
        diff=ceil(n/d);n=diff*d;
        bd=0.2; %0.13 at n=2000;0.12 at n=5000
        %pp=1/d*ones(d,1);
        Bl=0.05*ones(d,d);
        Dis=zeros(n,n);
        Label=ones(n,4);
%         theta=betarnd(1,4,n,1);
        theta=ones(n,1);
        for i=1:d
            Label((i-1)*diff+1:i*diff,3)=i;
            Bl(i,i)=bd;
        end
%         for i=1:d/2;
%             Label((i-1)*diff*2+1:i*diff*2,2)=i;
%         end
%         Label(1:4*diff,1)=1;Label(4*diff+1:end,1)=2;
        for i=1:n
            for j=1:n
                Dis(i,j)=rand(1)<theta(i)*theta(j)*Bl(Label(i,3),Label(j,3));
%                 Dis(j,i)=Dis(i,j);
            end
            Dis(i,i)=0;%diagonals are zeros
        end
        X=Dis;
        per=randperm(n);
        Label(per(1:numOut),4)=2;
        outlier=find(Label(:,4)==2);
        jend=n;
%         if repeat==1
            for i=1:n
                for j=1:n
                    Dis(i,j)=rand(1)<theta(i)*theta(j)*Bl(Label(i,3),Label(j,3));
                    %                 Dis(j,i)=Dis(i,j);
                end
                Dis(i,i)=0;%diagonals are zeros
            end
%         end
%         jend=muln*length(outlier);
        for i=1:length(outlier)
            for j=1:jend
                Dis(j,outlier(i))=rand(1)<theta(outlier(j))*theta(i)*(Bl(Label(outlier(i),3),Label(j,3))+diffSBM);
            end
            Dis(outlier(i),outlier(i))=0;%diagonals are zeros
        end
        X=[X,Dis];
        Dis=X;
     case 78 % SBM outlier: 30 vertices affects muln*30 adjacency
        fileName='SBM';
        diffSBM=d;
        d=5;numOut=100; muln=5;
        diff=ceil(n/d);n=diff*d;
        bd=0.2; %0.13 at n=2000;0.12 at n=5000
        %pp=1/d*ones(d,1);
        Bl=0.05*ones(d,d);
        Dis=zeros(n,n);
        Label=ones(n,4);
%         theta=betarnd(1,4,n,1);
        theta=ones(n,1);
        for i=1:d
            Label((i-1)*diff+1:i*diff,3)=i;
            Bl(i,i)=bd;
        end
%         for i=1:d/2;
%             Label((i-1)*diff*2+1:i*diff*2,2)=i;
%         end
%         Label(1:4*diff,1)=1;Label(4*diff+1:end,1)=2;
        for i=1:n
            for j=1:n
                Dis(i,j)=rand(1)<theta(i)*theta(j)*Bl(Label(i,3),Label(j,3));
%                 Dis(j,i)=Dis(i,j);
            end
            Dis(i,i)=0;%diagonals are zeros
        end
        X=Dis;
        per=randperm(n);
        Label(per(1:numOut),4)=2;
        outlier=find(Label(:,4)==2);
        jend=n;
%         if repeat==1
            for i=1:n
                for j=1:n
                    Dis(i,j)=rand(1)<theta(i)*theta(j)*Bl(Label(i,3),Label(j,3));
                    %                 Dis(j,i)=Dis(i,j);
                end
                Dis(i,i)=0;%diagonals are zeros
            end
%         end
%         jend=muln*length(outlier);
        for i=1:length(outlier)
            for j=1:jend
                Dis(outlier(i),j)=rand(1)<theta(outlier(i))*theta(j)*(Bl(Label(outlier(i),3),Label(j,3))+diffSBM);
                Dis(j,outlier(i))=rand(1)<theta(outlier(j))*theta(i)*(Bl(Label(outlier(i),3),Label(j,3))+diffSBM);
            end
            Dis(outlier(i),outlier(i))=0;%diagonals are zeros
        end
        X=[X,Dis];
        Dis=X;
   case 101 % DC-SBM with 20 classes
        fileName='DCSBM';
        pp=1/d*ones(d,1);
        Bl=0.1*ones(d,d);
        %             Bl=rand(clas,clas);
        for i=1:d
            Bl(i,i)=0.5;
        end
        if edge==0
%             if n>30000
%            Dis=sparse(n,n);
%             else
                Dis=zeros(n,n);
%             end
        else
            Dis=[];
        end
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
%         theta=betarnd(1,10,n,1);
%         ind=(Label==2);
%         theta(ind)= betarnd(10,10,sum(ind),1);
%         ind=(Label==3);
%         theta(ind)= betarnd(10,1,sum(ind),1);
        theta=betarnd(1,4,n,1);
%         theta=theta;
        for i=1:d
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        if edge==0
            for i=1:n
                Dis(i,i)=0;%diagonals are zeros
                for j=i+1:n
                    weight=1;%randi(10);
                    Dis(i,j)=weight*(rand(1)<theta(i)*theta(j)*Bl(Label(i),Label(j)));
                    Dis(j,i)=Dis(i,j);
                end
            end
        else
            for i=1:n
                for j=i+1:n
                    tmp=weight*(rand(1)<theta(i)*theta(j)*Bl(Label(i),Label(j)));
                    if tmp==1;
                        Dis=[Dis;i,j,1];
                    end
                end
            end
        end
        X=Dis;
    case 120 % DC-SBM with 3 classes
        fileName='DCSBM';
        pp=[0.2,0.3,0.5];
        d=3;
        Bl=zeros(d,d);
        %             Bl=rand(clas,clas);
        Bl(:,1)=[0.1,0.1,0.1];
        Bl(:,2)=[0.1,0.1,0.1];
        Bl(:,3)=[0.1,0.1,0.9];
        if edge==0
           Dis=zeros(n,n);
        else
            Dis=[];
        end
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        theta=betarnd(1,4,n,1);
%         theta=ones(n,1);
        % theta=unifrnd(0.5,1.5,n,1);
%         theta=theta;
        for i=1:d
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        if edge==0
            for i=1:n
                Dis(i,i)=0;%diagonals are zeros
                for j=i+1:n
                    weight=1;%randi(10);
                    Dis(i,j)=weight*(rand(1)<theta(i)*theta(j)*Bl(Label(i),Label(j)));
                    Dis(j,i)=Dis(i,j);
                end
            end
        else
            for i=1:n
                for j=i+1:n
                    tmp=(rand(1)<theta(i)*theta(j)*Bl(Label(i),Label(j)));
                    if tmp==1;
                        Dis=[Dis;i,j,1];
                    end
                end
            end
        end
        X=Dis;
     case 121 % DC-SBM with 3 classes
        fileName='DCSBM';
        pp=1/d*ones(1,d);
        Bl=0.1*ones(d,d);
        %             Bl=rand(clas,clas);
        Bl(:,d)=0.2*ones(1,d);interval=(0.9-0.2)/d;
        for i=1:d
            Bl(i,d)=Bl(i,d)+i*interval;
        end
        if edge==0
           Dis=zeros(n,n);
        else
            Dis=[];
        end
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        theta=betarnd(1,4,n,1);
        %         theta=ones(n,1);
        % theta=unifrnd(0.5,1.5,n,1);
        %         theta=theta;
        for i=1:d
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        if edge==0
            for i=1:n
                Dis(i,i)=0;%diagonals are zeros
                for j=i+1:n
                    weight=1;%randi(10);
                    Dis(i,j)=weight*(rand(1)<theta(i)*theta(j)*Bl(Label(i),Label(j)));
                    Dis(j,i)=Dis(i,j);
                end
            end
        else
            for i=1:n
                for j=i+1:n
                    tmp=(rand(1)<theta(i)*theta(j)*Bl(Label(i),Label(j)));
                    if tmp==1;
                        Dis=[Dis;i,j,1];
                    end
                end
            end
        end
        X=Dis;
    case {130,131}% DC-SBM with 3 classes
        fileName='DCSBM';
        pp=[0.2,0.2,0.6];
        d=3;
        Bl=zeros(d,d);
        %             Bl=rand(clas,clas);
        Bl(:,1)=[0.9,0.1,0.2];
        Bl(:,2)=[0.1,0.9,0.2];
        Bl(:,3)=[0.5,0.5,0.2];
        if edge==0
            Dis=zeros(n,n);
        else
            Dis=[];
        end
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        theta=betarnd(1,4,n,1);
        if option==131
           theta=ones(n,1);
        end
        % theta=unifrnd(0.5,1.5,n,1);
        %         theta=theta;
        for i=1:d
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        if edge==0
            for i=1:n
                Dis(i,i)=0;%diagonals are zeros
                for j=i+1:n
                    weight=1;%randi(10);
                    Dis(i,j)=weight*(rand(1)<theta(i)*theta(j)*Bl(Label(i),Label(j)));
                    Dis(j,i)=Dis(i,j);
                end
            end
        else
            for i=1:n
                for j=i+1:n
                    tmp=(rand(1)<theta(i)*theta(j)*Bl(Label(i),Label(j)));
                    if tmp==1;
                        Dis=[Dis;i,j,1];
                    end
                end
            end
        end
        X=Dis;
        Label(Label==2)=1;
        Label(Label==3)=2;
%         Dis=diag(sum(Dis))-Dis;
%         X=Dis;
%     case 33 % RDPG
%         fileName='RDPG';
%         pp=0.5;
%         p=1;
%         Label=(rand(n,1)>pp);
%         ind=(Label==1);
%         X = betarnd(3,6,n,p);
%         X(ind,:)= betarnd(6,3,sum(ind),p);
%         A=X*X';
%         Dis=double(rand(n,n)<A);
%         D=diag(sum(Dis));
%         Dis=eye(n)-D^(-0.5)*Dis*D^(-0.5);d=p;
%     case 3 
%         pp=0.5;
%         Label=(rand(n,1)>pp)+1;
%         ind=(Label==2);
%         d=2;
%         X = betarnd(3,5,n,d);
%         X(ind,:)= betarnd(5,3,sum(ind),d);
%         Dis=squareform(pdist(X));
%     case 34 % Gaussian Mixture
%                 pp=0.5;
%         p=1;
%         Label=(rand(n,1)>pp);
%         ind=(Label==1);
%         X = betarnd(3,6,n,p);
%         X(ind,:)= betarnd(6,3,sum(ind),p);
%         Dis=DCorInput(X,'euclidean');
% %         
% %         pp=[0.3,0.4,0.3];
% %         tmp=rand(n,1);
% %         Label=ones(n,1)+(tmp>pp(1))+(tmp>pp(1)+pp(2));
% %         X = mvnrnd(zeros(d,1),eye(d),n);
% %         ind=(Label==2);
% %         X(ind,:)= mvnrnd(1*ones(d,1),eye(d),sum(ind));
% %         ind=(Label==3);
% %         X(ind,:)= mvnrnd(-1*ones(d,1),eye(d),sum(ind));
% %         Dis=DCorInput(X,'euclidean');
% %         %Dis=DCorInput(X,'hsic');
%          d=p;
%     case 35 % Gaussian Mixture
%                         pp=0.5;
%         p=1;
%         Label=(rand(n,1)>pp);
%         ind=(Label==1);
%         X = betarnd(3,6,n,p);
%         X(ind,:)= betarnd(6,3,sum(ind),p);
% %         pp=[0.3,0.4,0.3];
% %         tmp=rand(n,1);
% %         Label=ones(n,1)+(tmp>pp(1))+(tmp>pp(1)+pp(2));
% %         X = mvnrnd(zeros(d,1),eye(d),n);
% %         ind=(Label==2);
% %         X(ind,:)= mvnrnd(1*ones(d,1),eye(d),sum(ind));
% %         ind=(Label==3);
% %         X(ind,:)= mvnrnd(-1*ones(d,1),eye(d),sum(ind));
% %         %Dis=DCorInput(X,'euclidean');
%         Dis=DCorInput(X,'hsic');
%         %Dis=squareform(pdist(X));
%         d=p;
%     case 36 % three half circle
%         pp=[0.3,0.4,0.3];
%         d=2;
%         tmp=rand(n,1);
%         Label=ones(n,1)+(tmp>pp(1))+(tmp>pp(1)+pp(2));
%         X=zeros(n,d);
%         X(:,1)=unifrnd(-1,1,n,1);
%         X(:,2)=sqrt(1-X(:,1).^2);
%         ind=(Label==2);
%         X(ind,:)= X(ind,:)*1.2;
%         ind=(Label==3);
%         X(ind,:)= X(ind,:)*1.5;
%         Dis=DCorInput(X,'euclidean');
end
%     case 6 % three spirals
%         pp=[0.3,0.4,0.3];
%         d=2;
%         tmp=rand(n,1);
%         Label=ones(n,1)+(tmp>pp(1))+(tmp>pp(1)+pp(2));
%         X=zeros(n,d);
%         z=unifrnd(0,5,n,1);
%         X(:,1)=z.*cos(z*pi);
%         X(:,2)=z.*sin(z*pi);
%         ind=(Label==2);
%         X(ind,:)= X(ind,:)*1.2;
%         ind=(Label==3);
%         X(ind,:)= X(ind,:)*1.5;
%         Dis=DCorInput(X,'euclidean');
% %         plot(X(:,1),X(:,2),'.');
%         %Dis=DCorInput(X,'hsic');
%         %Dis=squareform(pdist(X));end
% for i=1:n
%     Dis(i,i)=sum(Dis(i,:));
% end
% tmp=vecnorm(Dis);
% idx=(tmp>0);
% Dis=Dis(idx,idx);
% Label=Label(idx);