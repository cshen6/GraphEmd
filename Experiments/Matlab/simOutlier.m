function simOutlier

n=5000;level=1;
[Adj,Y]=simGenerate(70,n);
Adj1=Adj(:,1:n);
Adj2=Adj(:,n+1:2*n);
outlier=find(Y(:,4)==2);
[Z1]=GraphEncoderConcat(Adj1,Y(:,3),level);
[Z2]=GraphEncoderConcat(Adj2,Y(:,3),level);
res1=vecnorm(Z1-Z2,2,2);
[~,ind1]=sort(res1,'descend');
acc1=0; 
for i=1:30
    if ind1(i)<=30
        acc1=acc1+1/30;
    end
end

n=5000;level=1;
[Adj,Y]=simGenerate(71,n);
Adj1=Adj(:,1:n);
Adj2=Adj(:,n+1:2*n);
outlier=find(Y(:,4)==2);
[Z1]=GraphEncoderConcat(Adj1,Y(:,3),level);
[Z2]=GraphEncoderConcat(Adj2,Y(:,3),level);
res2=vecnorm(Z1-Z2,2,2);
[~,ind2]=sort(res2,'descend');
acc2=0; 
for i=1:30
    if ind2(i)<=30
        acc2=acc2+1/30;
    end
end

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
