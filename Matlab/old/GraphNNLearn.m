function [error1]=GraphNNLearn(X,Y,num)

if nargin<3
    num=10; % 10 fold cross validation by default
end

if min(Y)<1
    Y=Y+1-min(Y);
end

% option1=1;
% thres=0.5;
%X=X+eye(size(X));
indices = crossvalind('Kfold',Y,num);
error1=0;
% error2=zeros(10,1);
% error3=zeros(10,1);
% rng('default')

for i = 1:num
    test = (indices == i); % test indices
    train = ~test; % training indices
    
    % try GNN
    [mdl]=GraphNN2(X(train,:),Y(train)); % build model
    labelV = mdl(X(test,:)'); 
    label = vec2ind(labelV)';
    error1=error1+mean(Y(test)~=label)/num;
    
%     for r=1:10
%         label(train)=Y(train);
%         tmp=(max(labelV,[],1)>thres);
%         tmp(train)=1;
%         [mdl]=GraphNN2(X(tmp,:),label(tmp)); % build model
%         labelV = mdl(X');
%         label2 = vec2ind(labelV)';
%         error2(r)=error2(r)+mean(Y(test)~=label2(test))/num;
%         
%         [mdl]=GraphNN2(X,label); % build model
%         labelV = mdl(X');
%         label2 = vec2ind(labelV)';
%         error3(r)=error3(r)+mean(Y(test)~=label2(test))/num;
%     end
end