function error=GraphRandom(X,Y)
%rng('default')
num=10;
indices = crossvalind('Kfold',Y,num);
error=0;
% rng('default')
for i = 1:num
    test = (indices == i);
    train = ~test;
    for r=1:rep
        %[U,S,V]=svd(unifrnd(0,1,n,n));
        V=unifrnd(0,1,n,n);
        for i=1:dr
            %     tic
            W=adj*V(:,1:i);
            %     toc
            % plot(W(1:n1,1),W(1:n1,2),'r.')
            % hold on
            % plot(W(n1+1:n,1),W(n1+1:n,2),'b.')
            % hold off
            %error2(i) = error2(i)+GraphNNLearn(W,Y)/rep;
            error2(i) = error2(i)+graphC(W,Y)/rep;
        end
    end
    %mdl=fitcdiscr(X(train,:),Y(train));
    tt=predict(mdl,X(test,:));
    error=error+mean(Y(test)~=tt)/num;
end