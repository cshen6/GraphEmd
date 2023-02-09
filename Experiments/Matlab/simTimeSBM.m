function [Z,Z_d,Y,time,ind]=simTimeSBM(type,n,K,t)

[A,Y]=simGenerate(type,n,K);
E=adj2edge(A);
E(:,3)=randi(100,size(E,1),1);
G={E};s=size(E,1);
inlier=binornd(1,0.5,s,1);
outlier=(~inlier);
for i=1:t-1
    E(outlier,3)=E(outlier,3)+randi([-10,10],sum(outlier),1);
    E(E<1)=1;
    G=[G,E];
end
[Z,Z_d,Y,time,ind]=GraphDynamics(G,Y);