function [stat,pval,corrCom,pvalCom,nk]=CorrGEE(A,B,Y)

n=size(A,1);
if nargin<3
    Y=ones(n,1);
end

opts = struct('Normalize',0,'Unbiased',0,'DiagAugment',0,'Principal',0,'Laplacian',0,'Discriminant',1);
std=2;eps=0.01;directed=1;p=1;
if issymmetric(A) && issymmetric(B)
    directed=0;
end
[~,out1]=GraphEncoder(A,Y,opts);
[~,out2]=GraphEncoder(B,Y,opts);
[~,out12]=GraphEncoder(A.*B,Y,opts);
[~,out11]=GraphEncoder(A.*A,Y,opts);
[~,out22]=GraphEncoder(B.*B,Y,opts);
nk=out12.nk;
K=length(nk);
nk=nk*nk';

% community correlation computation
covCom=out12.mu-out1.mu.*out2.mu;
Var1=out11.mu-out1.mu.*out1.mu;
Var2=out22.mu-out2.mu.*out2.mu;
corrCom=covCom./sqrt(Var1.*Var2);
if size(corrCom,1)>1
    % thresholding to exclude correlations of very sparse communities
    ind=(Var1<eps)|(Var2<eps);
    ind=ind | ind';
    ind(boolean(eye(K)))=0;
    % tmp=corrCom;
    corrCom(ind)=0;
    p2=sum(sum(ind));
    % p2=0;
    % directed or not
    if ~directed
        corrCom(boolean(eye(K)))=diag(corrCom);
        p=K*(K+1)/2-p2/2;
    else
        p=K^2-p2;
    end
end
% final dependence measure and hypothesis testing
tmp=sqrt(nk).*abs(corrCom);
pvalCom=2-2*normcdf(tmp,0,std);
stat=max(max(tmp));
pval=1-(2*normcdf(stat,0,std)-1)^(p);

% stat=mean(mean(tmp));
% pval=1-normcdf(stat,0,std/p);
% end
% 
% if choice>0
%     K=max(Y);
%     corrCom=zeros(K,1);
%     pvalCom=zeros(K,1);
%     nk=zeros(K,1);
%     for i=1:K
%        indi=(Y==i);nk(i)=sum(indi);
%        [corrCom(i),pval(i)]=DCorFastTest(Z1(indi,:),Z2(indi,:));
%     end
%     stat=max(max(nk.*corrCom+1));
%     pval=1-chi2cdf(stat,1)^K;
% end
% if NumPerms==0
    % pvalCom=1-normcdf(tmp,0,std);
    % pval=1-normcdf(stat,0,std)^(p);
% else
%     stat2=max(max(covCom));pvalCom=0;pval=0;
%     for r=1:NumPerms
%         per=randperm(n);
%         [~,Z12Per]=GraphEncoder(A.*B(per,:),Y,opts);
%         [~,Z2Per]=GraphEncoder(B(per,:),Y,opts);
%         covComPer=Z12Per.mu-Z1.mu.*Z2Per.mu;
%         pvalCom=pvalCom+(covCom>covComPer)/NumPerms;
%         stat2Per=max(max(covComPer));
%         pval=pval+(stat2>stat2Per)/NumPerms;
%     end
% end



% nZ=sqrt(min(nk)*min(nk)');
%
% z=Corr12*nZ+1;
%
% % compute the pvalue via chi-square test
%pval0=1-chi2cdf(z,1);
%pval1=[1-chi2cdf(max(max(z)),1)^p;1-chi2cdf(p*mean(mean(z)),p)];