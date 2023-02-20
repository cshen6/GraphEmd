function [Z,VD,Y,time]=simDynamicSBM(type,n,K,t)

[A,Y]=simGenerate(type,n,K);
E=adj2edge(A);
E(:,3)=randi(100,size(E,1),1);
G={E};s=size(E,1);
for i=1:t-1
    inlier=binornd(1,0.5,s,1);
    outlier=(~inlier);
    E(outlier,3)=E(outlier,3)+randi([-20,20],sum(outlier),1);
    E(E<1)=1;
    G=[G,E];
end
[Z,VD,Y,time]=GraphDynamics(G,Y);