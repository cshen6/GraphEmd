function [corr,pval,corrCom,pvalCom]=GraphCorr(A,B,Y)

opts = struct('Normalize',0,'Unbiased',1,'DiagAugment',0,'Principal',0,'Laplacian',0,'Discriminant',1);
std=1;

[~,Z12]=GraphEncoder(A.*B,Y,opts);
% [~,Z11]=GraphEncoder(Dis{1}.*Dis{1},Label,opts);
% [~,Z22]=GraphEncoder(Dis{2}.*Dis{2},Label,opts);
[~,Z1]=GraphEncoder(A,Y,opts);
[~,Z2]=GraphEncoder(B,Y,opts);
nk=Z12.nk;
K=length(nk);
nk=nk*nk';

covCom=Z12.mu-Z1.mu.*Z2.mu;
cov=mean(mean(sqrt(nk).*covCom));

Var1=Z1.mu.*(1-Z1.mu);
Var2=Z2.mu.*(1-Z2.mu);
% Var1=Z1.mu.*Z1.mu;
% Var2=Z2.mu.*Z2.mu;
corrCom=sqrt(nk).*covCom./sqrt(Var1.*Var2);
corr=mean(mean(corrCom));%cov./sqrt(mean(mean(sqrt(nk).*Var1))*mean(mean(sqrt(nk).*Var2)));
pvalCom=1-normcdf(corrCom,0,std);
pval=1-normcdf(corr,0,std/sqrt(K*(K+1)/2));
% nZ=sqrt(min(nk)*min(nk)');
%
% z=Corr12*nZ+1;
%
% % compute the pvalue via chi-square test
%pval0=1-chi2cdf(z,1);
%pval1=[1-chi2cdf(max(max(z)),1)^p;1-chi2cdf(p*mean(mean(z)),p)];