function simOutlier

% Compare supGEE, unsupGEE, unfolded ASE for outlier detection
ll=3;diffSBM=0.1;type=70; unfolded=true; % 70-72 at 0.01 diff, n=5000; 73-75 at 0.1 diff, n goes from 500 to 5000; 76-79 at 0.2 diff
n=3000;level=1;numOut=100; 
directed=3;opts = struct('Directed',directed,'Normalize',true);
[Adj,Y]=simGenerate(type,n,diffSBM);
d=max(max(Y));
Adj1=Adj(:,1:n);X1=adj2edge(Adj1);
Adj2=Adj(:,n+1:2*n);X2=adj2edge(Adj2);
outlier=find(Y(:,4)==2);
inlier=find(Y(:,4)==1);
acc=zeros(9,1); time=zeros(9,1); 
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
tic
if unfolded==true
ZUA1=ASE([Adj1',Adj2'],d);
ZUA2=ASE([Adj1,Adj2],d);
ZA1=zeros(n,d,directed); ZA1(:,:,1)=ZUA1(1:n,:);ZA1(:,:,2)=ZUA2(1:n,:);ZA1(:,:,3)=ZUA1(1:n,:)+ZUA2(1:n,:);
ZA2=zeros(n,d,directed); ZA2(:,:,1)=ZUA1(n+1:2*n,:);ZA2(:,:,2)=ZUA2(n+1:2*n,:);ZA2(:,:,3)=ZUA1(n+1:2*n,:)+ZUA2(n+1:2*n,:);
% ZU1=ZS1;ZU2=ZS2;
end
t3=toc;
%%%%
for i=1:9
    tic
    if i>=7
        if unfolded==false
            break;
        end
        Z1=ZA1(:,:,(i-6));Z2=ZA2(:,:,(i-6));
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
time(7:9)=time(7:9)+t3;
acc=array2table(reshape(acc,3,3),'RowNames', {'Sender', 'Receiver','Mixed'},'VariableNames', {'sup GEE','unsup GEE','unfolded ASE'});
time=array2table(reshape(time,3,3),'RowNames', {'Sender', 'Receiver','Mixed'},'VariableNames', {'sup GEE','unsup GEE','unfolded ASE'});


map2 = brewermap(128,'PiYG'); % brewmap
i=5;
Z1=ZU1(:,:,(i-3));Z2=ZU2(:,:,(i-3));
% [~,Z1]=pca(Z1,'NumComponents',2);[~,Z2]=pca(Z2,'NumComponents',2);
% Z1=run_umap([Z1,Y(:,ll)],'label_column','end','contour_percent',0, 'verbose','none','randomize',false);
% Z2=run_umap([Z2,Y(:,ll)],'label_column','end','contour_percent',0, 'verbose','none','randomize',false);
subplot(1,2,1)
hold on
plot(Z1(inlier,1),Z1(inlier,2),'o');
plot(Z2(inlier,1),Z2(inlier,2),'.');
axis('square')
title('Inliers')
subplot(1,2,2)
hold on
plot(Z1(outlier,1),Z1(outlier,2),'o');
plot(Z2(outlier,1),Z2(outlier,2),'*');
axis('square')
title('Outliers')
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
