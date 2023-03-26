function simModel

%%% Bootstrap
n=2000; type=30;k=10;
[Adj,Y]=simGenerate(type,n,k);
[pi,B,theta,Z]=GraphSBMEst(Adj,Y);
[Adj2,Y2]=GraphSBMGen(pi,B,theta,n);
[pi2,B2,theta2,Z2]=GraphSBMEst(Adj2,Y2);
figure
subplot(1,2,1);
plot(Z(Y==1,1),Z(Y==1,2),'o');
hold on
plot(Z(Y==2,1),Z(Y==2,2),'x');
hold off
subplot(1,2,2);
hold on
plot(Z2(Y2==1,1),Z2(Y2==1,2),'o');
plot(Z2(Y2==2,1),Z2(Y2==2,2),'x');
hold off

% [pval,stat]=GraphTwoSampleTest(Adj,Adj2,Y,Y2)

% n=2000; type=22;k=10;
% [Adj3,Y3]=simGenerate(type,n,k);
% [pi,B,theta,Z]=GraphSBMEst(Adj3,Y3);
% [Adj4,Y4]=GraphSBMGen(pi,B,theta,n);
% [pi2,B2,theta2,Z2]=GraphSBMEst(Adj2,Y2);
% 
% [pval,stat]=GraphTwoSampleTest(Adj,Adj3,Y,Y3)
% [pval,stat]=GraphTwoSampleTest(Adj2,Adj4,Y2,Y4)
%%
load('polblogs.mat'); Y=Label+1;  
load('graphCElegans.mat'); Adj=Ac; Y=vcols; 
load('adjnoun.mat');Y=Label; 
load('email.mat'); 
load('pubmed.mat'); 
load('CoraAdj.mat'); 
load('Gene.mat'); 

n=3000;
[pi,B,theta,Z]=GraphSBMEst(Adj,Y);
[Adj2,Y2]=GraphSBMGen(pi,B,theta,n);
[pi2,B2,theta2,Z2]=GraphSBMEst(Adj2,Y2);
figure
subplot(1,2,1);
plot(Z(Y==1,1),Z(Y==1,2),'o');
hold on
plot(Z(Y==2,1),Z(Y==2,2),'x');
hold off
subplot(1,2,2);
hold on
plot(Z2(Y2==1,1),Z2(Y2==1,2),'o');
plot(Z2(Y2==2,1),Z2(Y2==2,2),'x');
hold off