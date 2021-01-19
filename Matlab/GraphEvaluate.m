function [error]=GraphEvaluate(X,Y,M,option,num)

if nargin<3
    M=0; % 10 fold cross validation by default
end
if nargin<4
    option=2; % 10 fold cross validation by default
end
if nargin<5
    num=10; % 10 fold cross validation by default
end
% try object-wise probability updating!
% data structure at finite n: whether that has info to improve
% classification
if min(Y)<1
    Y=Y+1-min(Y);
end

%X=X+eye(size(X));
indices = crossvalind('Kfold',Y,num);
error=0;
% rng('default')
% error2=zeros(5,1);
% error3=zeros(5,1);
thres=0.9;

% opt='euclidean';
% % opt='hsic';
% [~,ind]=checkDist(X);
% if ind==0
%     X=DCorInput(X,opt);
% end

for i = 1:num
    %     test = (indices == i); % test indices
    %     train = ~test; % training indices
    
    test = (indices == i); % test indices
    train = ~test; % training indices
%          [~,filter]=GraphFilter(X(train,:),Y(train));
    %mdl=fitgmdist(X,k);
         mdl = fitcdiscr(X(train,:),Y(train),'discrimType','pseudoLinear');
         tt=predict(mdl,X(test,:));
        error=error+mean(Y(test)~=tt)/num;
%     if size(M,1)==size(X,1)
%         [mdl,filter]=GraphNN(X(train,train),Y(train),M(train,:),option); % build model
%         labelV = GraphNNPredict(X(test,train),mdl,filter,M(test,:),option);
%     else
%         [mdl,filter]=GraphNN(X(train,train),Y(train),0,option); % build model
%         labelV = GraphNNPredict(X(test,train),mdl,filter,0,option);
%     end
%     label = vec2ind(labelV)';
%     error=error+mean(Y(test)~=label)/num; % error rate

    % try LDA
%     if option==0
%         [Z,filter]=GraphFilter(X(train,train),Y(train));
%         %mdl=fitgmdist(X,k);
%         mdl = fitcdiscr(Z,Y(train));
%         tt=predict(mdl,X(test,train)*filter);
%         error=error+mean(Y(test)~=tt)/num;
%     else
%         % try GNN
%         [mdl,filter]=GraphNN(X(train,train),Y(train),M(train,:),option); % build model
%         labelV = GraphNNPredict(X(:,train),mdl,filter);
%         label = vec2ind(labelV)';
%         error=error+mean(Y(test)~=label(test))/num; % error rate
%         
%         for r=1:5
%             label(train)=Y(train);
%             tmp=(max(labelV,[],1)>thres);
%             tmp(train)=1;
%             [mdl,filter]=GraphNN(X(tmp,tmp),label(tmp)); % build model
%             labelV = GraphNNPredict(X(:,tmp),mdl,filter);
%             label2 = vec2ind(labelV)';
%             error2(r)=error2(r)+mean(Y(test)~=label2(test))/num;
% 
%             [mdl,filter]=GraphNN(X,label); % build model
%             labelV = GraphNNPredict(X,mdl,filter);
%             label2 = vec2ind(labelV)';
%             error3(r)=error3(r)+mean(Y(test)~=label2(test))/num;
%         end
%     end
end