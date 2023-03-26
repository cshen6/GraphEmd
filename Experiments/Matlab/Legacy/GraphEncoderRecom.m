function [n1,n2,n3]=GraphEncoderRecom(Z,Y,ind,k)


[idx,D] = knnsearch(Z,Z(ind,:),'K',k); %%IPhoneX, 53, 
n1=[idx;D];
m=length(ind);
n2=zeros(2*m,k);
n3=zeros(k,2*m);
[n2(m+1:end,:),n2(1:m,:)]=maxk(Z(ind,:),k,2);
[n3(:,m+1:end),n3(:,1:m)]=maxk(Z(:,Y(ind)),k,1);
n3=n3';