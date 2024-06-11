function simCorr(choice, corrInd, reps)

n=100;
reps=100;
power0=0;
power1=zeros(4,4);
alpha=0.05;
corrInd=0.05;
corr=zeros(reps,1);
for r=1:reps
    r
    [Dis,Y]=simGenerate(18+(choice-1)*10,n,1,0);
    Dis{2}=Dis{1}*corrInd+Dis{2}*(1-corrInd);
    [corr(r),cov,pval,corrCom,covCom,pvalCom]=GraphCorr(Dis{1},Dis{2},Y);
    power0=power0+(pval<alpha)/reps;
    power1=power1+(pvalCom<alpha)/reps;
end
hist(corr)
power0
power1

save(strcat('GEECorSim',num2str(choice),'n',num2str(n),'Corr',num2str(corrInd),'.mat'),'choice','corrInd','reps','power0','power1','n','corr');

