function [pval,stat]=GraphTwoSampleTest(Adj,Adj2,Y,Y2)

opts1=2;
opts = struct('metric','hsic','max',0); % default parameters
if opts1==1;
d=10;
[U,S,~]=svds(Adj,d);
Z=U(:,1:d)*S(1:d,1:d)^0.5;

[U,S,~]=svds(Adj2,d);
Z2=U(:,1:d)*S(1:d,1:d)^0.5;
end

if opts1==2;
    [Z,~]=GraphEncoder(Adj,Y);
    [Z2,~]=GraphEncoder(Adj2,Y2);
    figure
    plot(Z(Y==1,1),Z(Y==1,2),'o');
    hold on
    plot(Z(Y==2,1),Z(Y==2,2),'x');
    hold off
    figure
    hold on
    plot(Z2(Y2==1,1),Z2(Y2==1,2),'o');
    plot(Z2(Y2==2,1),Z2(Y2==2,2),'x');
    hold off
end
% [stat, pval,localCor,optimalScale]=MGCFastTest(Z,Z2);
[stat,pval]=DCorFastTest(Z,Z2,opts);