function [Dis,Label,d,X]=generateSims(option,n,d)
if nargin<3
    d=2;
end
switch option
    case 1 % SBM with 3 classes
        fileName='SBM3';
        clas=3;
        pp=[0.3,0.4,0.3];
        Bl=zeros(clas,clas);
        %             Bl=rand(clas,clas);
        Bl(:,1)=[0.3,0.1,0.1];
        Bl(:,2)=[0.1,0.3,0.1];
        Bl(:,3)=[0.1,0.1,0.3];
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
        d=3;
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
    case 2 % RDPG
        fileName='RDPG';
        pp=0.5;
        d=1;
        Label=(rand(n,1)>pp);
        ind=(Label==1);
        X = betarnd(3,6,n,d);
        X(ind,:)= betarnd(6,3,sum(ind),d);
        A=X*X';
        Dis=double(rand(n,n)<A);
%         X=Dis;
    case 10 % RDPG
        fileName='RDPG';
        pp=0.5;
        d=1;
        Label=(rand(n,1)>pp);
        ind=(Label==1);
        X = betarnd(3,6,n,d);
        X(ind,:)= betarnd(6,3,sum(ind),d);
        A=X*X';
        Dis=double(rand(n,n)<A);
        Dis=diag(sum(Dis))-Dis;
    case 11 % RDPG
        fileName='RDPG';
        pp=0.5;
        d=1;
        Label=(rand(n,1)>pp);
        ind=(Label==1);
        X = betarnd(3,6,n,d);
        X(ind,:)= betarnd(6,3,sum(ind),d);
        A=X*X';
        Dis=double(rand(n,n)<A);
        D=diag(sum(Dis));
        Dis=eye(n)-D^(-0.5)*Dis*D^(-0.5);
%     case 3 
%         pp=0.5;
%         Label=(rand(n,1)>pp)+1;
%         ind=(Label==2);
%         d=2;
%         X = betarnd(3,5,n,d);
%         X(ind,:)= betarnd(5,3,sum(ind),d);
%         Dis=squareform(pdist(X));
    case 4 % Gaussian Mixture
                pp=0.5;
        d=1;
        Label=(rand(n,1)>pp);
        ind=(Label==1);
        X = betarnd(3,6,n,d);
        X(ind,:)= betarnd(6,3,sum(ind),d);
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
    case 5 % Gaussian Mixture
                        pp=0.5;
        d=1;
        Label=(rand(n,1)>pp);
        ind=(Label==1);
        X = betarnd(3,6,n,d);
        X(ind,:)= betarnd(6,3,sum(ind),d);
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
    case 6 % three half circle
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
end
% for i=1:n
%     Dis(i,i)=sum(Dis(i,:));
% end