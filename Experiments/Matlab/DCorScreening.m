function [ind,corr,pval] = DCorScreening(X,Y,thres)

% if nargin<3
%     thres=0.05;
% end
% d=size(X,2);
% pval=zeros(d,1);
% corr=zeros(d,1);
% for i=1:d
%     [corr(i),pval(i)]=DCorFastTest(X(:,i),Y);
% end
% ind=(pval<thres);

if nargin<3
    thres=0.05;
end
option=1;
[~,d]=size(X);
corr=zeros(d,1);
pval=zeros(d,1);
for i=1:d
    if option==2
        [tmp,pp]=corrcoef(X(:,i),Y);
        corr(i)=abs(tmp(1,2));
        pval(i)=pp(1,2);
    else
        [corr(i),pval(i)]=DCorFastTest(X(:,i),Y);
    end
end

if thres<1
   ind=(pval<thres);
else
   [~,ind]=sort(pval);
   ind=ind(1:ceil(thres));
end
% ind2=(pval>=thres);