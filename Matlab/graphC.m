function error=graphC(X,Y)
%rng('default')
num=10;
indices = crossvalind('Kfold',Y,num);
error=0;
% rng('default')
for i = 1:num
    test = (indices == i);
    train = ~test;
    mdl=fitcecoc(X(train,:),Y(train));
    %mdl=fitcdiscr(X(train,:),Y(train));
    tt=predict(mdl,X(test,:));
    error=error+mean(Y(test)~=tt)/num;
end