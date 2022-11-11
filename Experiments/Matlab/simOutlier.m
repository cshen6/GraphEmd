function simOutlier

% Compare supGEE, unsupGEE, unfolded ASE for outlier detection
ll=3;diffSBM=0.05;type=70; unfolded=true;
n=5000;level=1;numOut=100; 
directed=3;opts = struct('Directed',directed);
[Adj,Y]=simGenerate(type,n,diffSBM);
d=max(max(Y));
Adj1=Adj(:,1:n);X1=adj2edge(Adj1);
Adj2=Adj(:,n+1:2*n);X2=adj2edge(Adj2);
outlier=find(Y(:,4)==2);
acc=zeros(7,1); time=zeros(7,1); 
tic
% [ZC1]=GraphEncoderConcat(X1,Y(:,ll),level,1,directed);
% [ZC2]=GraphEncoderConcat(X2,Y(:,ll),level,1,directed);
[ZS1]=GraphEncoderConcat(X1,Y(:,ll),level,1,directed);
[ZS2]=GraphEncoderConcat(X2,Y(:,ll),level,1,directed);
t1=toc;
tic
ZU=GraphEncoder({X1,X2},d,opts);
ZU1=ZU(:,1:size(ZU,2)/2,:);ZU2=ZU(:,size(ZU,2)/2+1:end,:);
% ZU1=ZS1;ZU2=ZS2;
t2=toc;
%%%%
for i=1:7
    tic
    if i==7
        if unfolded==false
            break;
        end
        if type==70
           Z1=ASE([Adj1',Adj2'],d);
        else
           Z1=ASE([Adj1,Adj2],d);
        end
        Z2=Z1(n+1:2*n,d);
        Z1=Z1(1:n,d);
        [~,Z2] = procrustes(Z1,Z2);
        %         Z2=ASE(Adj2,d);
    else
        if i<=3
           Z1=ZS1(:,:,i);Z2=ZS2(:,:,i);
        else
           Z1=ZU1(:,:,(i-3));Z2=ZU2(:,:,(i-3));
        end
    end
    res=vecnorm(Z1-Z2,2,2);
    [~,ind]=sort(res,'descend');
    acc(i)=mean(ismember(ind(1:numOut),outlier));
    time(i)=toc; 
end
time(1:3)=time(1:3)+t1;
time(4:6)=time(4:6)+t2;


% n=5000;level=1;numOut=100;ll=3;diffSBM=0.05;
% [Adj,Y]=simGenerate(71,n,diffSBM;
% Adj1=Adj(:,1:n);
% Adj2=Adj(:,n+1:2*n);
% outlier=find(Y(:,4)==2);
% [Z1]=GraphEncoderConcat(Adj1,Y(:,ll),level);
% [Z2]=GraphEncoderConcat(Adj2,Y(:,ll),level);
% res2=vecnorm(Z1-Z2,2,2);
% [~,ind2]=sort(res2,'descend');
% acc2=0; 
% for i=1:30
%     if ind2(i)<=30
%         acc2=acc2+1/30;
%     end
% end

% n=5000;level=3;
% [Adj,Y]=simGenerate(71,n);
% Adj1=Adj(:,1:n);
% Adj2=Adj(:,n+1:2*n);
% [Z1]=GraphEncoderConcat(Adj1,Y,level);
% [Z2]=GraphEncoderConcat(Adj2,Y,level);
% res2=vecnorm(Z1-Z2,2,2);
% [~,ind2]=sort(res2,'descend');
% acc2=0; 
% for i=1:30
%     if ind2(i)<=30
%         acc2=acc2+1/30;
%     end
% end
