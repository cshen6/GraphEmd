function [Dis,Label,d,X]=simGenerate(option,n,d)
if nargin<3
    d=3;
end
switch option
    case 1 % SBM with 3 classes
        fileName='SBM';
        d=3;
        pp=[0.2,0.3,0.5];
        Bl=zeros(d,d);
        %             Bl=rand(clas,clas);
        Bl(:,1)=[0.13,0.1,0.1];
        Bl(:,2)=[0.1,0.13,0.1];
        Bl(:,3)=[0.1,0.1,0.13];
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
     case 2 % DC-SBM with 3 classes
        fileName='DCSBM';
        pp=[0.2,0.3,0.5];
        Bl=zeros(d,d);
        %             Bl=rand(clas,clas);
        Bl(:,1)=[0.9,0.1,0.1];
        Bl(:,2)=[0.1,0.6,0.1];
        Bl(:,3)=[0.1,0.1,0.3];
        Dis=zeros(n,n);
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        theta=betarnd(1,4,n,1);
%         theta=theta;
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
        X=Dis;
%         deg=diag(sum(X,1));
%         X=deg^(-0.5)*X*deg^(-0.5);
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
    case 3 % RDPG
        fileName='RDPG';
        p=1;
        pp=[0.2,0.3,0.5];
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:size(pp,2)
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        X = betarnd(1,3,n,p);
        ind=(Label==2);
        X(ind,:)= betarnd(3,3,sum(ind),p);
        ind=(Label==3);
        X(ind,:)= betarnd(3,1,sum(ind),p);
        A=X*X';
        Dis=double(rand(n,n)<A);
        d=p;
%         X=Dis;
    case 4 % RDPG with dist 
        fileName='RDPG';
        p=1;
        pp=[0.2,0.3,0.5];
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:size(pp,2)
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        X = betarnd(1,3,n,p);
        ind=(Label==2);
        X(ind,:)= betarnd(3,3,sum(ind),p);
        ind=(Label==3);
        X(ind,:)= betarnd(3,1,sum(ind),p);
        A=exp(-squareform(pdist(X)))/4;
        Dis=double(rand(n,n)<A);
        d=p;
    case 5 % RDPG with kernel
        fileName='RDPG';
        p=1;
        pp=[0.2,0.3,0.5];
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:size(pp,2)
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        X = betarnd(1,3,n,p);
        ind=(Label==2);
        X(ind,:)= betarnd(3,3,sum(ind),p);
        ind=(Label==3);
        X(ind,:)= betarnd(3,1,sum(ind),p);
        A=exp(-squareform(pdist(X)))/4;
        Dis=double(rand(n,n)<A);
        d=p;
%         Dis=diag(sum(Dis))-Dis;
%         X=Dis;
    case 6 % RDPG
        fileName='RDPG';
        pp=0.5;
        p=1;
        Label=(rand(n,1)>pp);
        ind=(Label==1);
        X = betarnd(3,6,n,p);
        X(ind,:)= betarnd(6,3,sum(ind),p);
        A=X*X';
        Dis=double(rand(n,n)<A);
        D=diag(sum(Dis));
        Dis=eye(n)-D^(-0.5)*Dis*D^(-0.5);d=p;
%     case 3 
%         pp=0.5;
%         Label=(rand(n,1)>pp)+1;
%         ind=(Label==2);
%         d=2;
%         X = betarnd(3,5,n,d);
%         X(ind,:)= betarnd(5,3,sum(ind),d);
%         Dis=squareform(pdist(X));
    case 6 % Gaussian Mixture
                pp=0.5;
        p=1;
        Label=(rand(n,1)>pp);
        ind=(Label==1);
        X = betarnd(3,6,n,p);
        X(ind,:)= betarnd(6,3,sum(ind),p);
        Dis=DCorInput(X,'euclidean');
%         
%         pp=[0.3,0.4,0.3];
%         tmp=rand(n,1);
%         Label=ones(n,1)+(tmp>pp(1))+(tmp>pp(1)+pp(2));
%         X = mvnrnd(zeros(d,1),eye(d),n);
%         ind=(Label==2);
%         X(ind,:)= mvnrnd(1*ones(d,1),eye(d),sum(ind));
%         ind=(Label==3);
%         X(ind,:)= mvnrnd(-1*ones(d,1),eye(d),sum(ind));
%         Dis=DCorInput(X,'euclidean');
%         %Dis=DCorInput(X,'hsic');
         d=p;
    case 7 % Gaussian Mixture
                        pp=0.5;
        p=1;
        Label=(rand(n,1)>pp);
        ind=(Label==1);
        X = betarnd(3,6,n,p);
        X(ind,:)= betarnd(6,3,sum(ind),p);
%         pp=[0.3,0.4,0.3];
%         tmp=rand(n,1);
%         Label=ones(n,1)+(tmp>pp(1))+(tmp>pp(1)+pp(2));
%         X = mvnrnd(zeros(d,1),eye(d),n);
%         ind=(Label==2);
%         X(ind,:)= mvnrnd(1*ones(d,1),eye(d),sum(ind));
%         ind=(Label==3);
%         X(ind,:)= mvnrnd(-1*ones(d,1),eye(d),sum(ind));
%         %Dis=DCorInput(X,'euclidean');
        Dis=DCorInput(X,'hsic');
        %Dis=squareform(pdist(X));
        d=p;
    case 8 % three half circle
        pp=[0.3,0.4,0.3];
        d=2;
        tmp=rand(n,1);
        Label=ones(n,1)+(tmp>pp(1))+(tmp>pp(1)+pp(2));
        X=zeros(n,d);
        X(:,1)=unifrnd(-1,1,n,1);
        X(:,2)=sqrt(1-X(:,1).^2);
        ind=(Label==2);
        X(ind,:)= X(ind,:)*1.2;
        ind=(Label==3);
        X(ind,:)= X(ind,:)*1.5;
        Dis=DCorInput(X,'euclidean');
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
%         %Dis=squareform(pdist(X));
     case 11 % RDPG with 3 data
        fileName='SBM';
        Dis=zeros(n,n,3);
        pp=[0.2,0.3,0.2,0.3];
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:size(pp,2)
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
    case 12 % RDPG with 3 data
        fileName='RDPG';
        p=1;
        Dis=zeros(n,n,3);
        pp=[0.2,0.3,0.5];
        tt=rand([n,1]);
        Label=ones(n,1);
        thres=0;
        for i=1:size(pp,2)
            thres=thres+pp(i);
            Label=Label+(tt>thres); %determine the block of each data
        end
        X = betarnd(1,3,n,p);
        ind=(Label==3);
        X(ind,:)= betarnd(3,1,sum(ind),p);
        A=X*X';
        Dis(:,:,1)=double(rand(n,n)<A);
        X = betarnd(3,1,n,p);
        ind=(Label==1);
        X(ind,:)= betarnd(1,3,sum(ind),p);
        A=X*X';
        Dis(:,:,2)=double(rand(n,n)<A);
        X = betarnd(1,3,n,p);
        A=X*X';
        Dis(:,:,3)=double(rand(n,n)<A);   
        d=p;
end
% for i=1:n
%     Dis(i,i)=sum(Dis(i,:));
% end
% tmp=vecnorm(Dis);
% idx=(tmp>0);
% Dis=Dis(idx,idx);
% Label=Label(idx);