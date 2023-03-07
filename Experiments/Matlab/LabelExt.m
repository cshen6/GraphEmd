function Y2=LabelExt(Y, newC)

n=length(Y);
[K,~,Y]=unique(Y);
K=length(K);
Y2=Y;

for i =1:K
    ind=(Y==i);
    tmp=randi([1,newC],sum(ind),1);
    Y2(ind)=tmp+newC*(i-1);
end

% Y2=randi([1,newC*K],n,1);


% function [y] = label_propagation(W, y, max_iter)
% % Inputs:
% % W: adjacency matrix of the graph
% % y: initial label vector
% % max_iter: maximum number of iterations
% %
% % Output:
% % y: final label vector after label propagation
% 
% N = size(W, 1); % number of nodes
% for i = 1:max_iter
%     y_new = y;
%     for j = 1:N
%         % compute weighted sum of neighboring labels
%         y_new(j) = sum(W(j, :) .* y) / sum(W(j, :));
%     end
%     y = y_new;
% end


% function [C,Q] = louvain(A)
% % A: adjacency matrix
% % C: community assignments
% % Q: modularity value
% 
% n = size(A,1);
% C = (1:n)';
% Q = -inf;
% improve = 1;
% while improve
%     improve = 0;
%     for u = randperm(n)
%         Ci = C(u);
%         neighbors = find(A(u,:));
%         Ci_neighbors = unique(C(neighbors));
%         Qmax = -inf;
%         Cmax = Ci;
%         for i = 1:length(Ci_neighbors)
%             Cj = Ci_neighbors(i);
%             S = sum(A(u, C==Cj));
%             Qtemp = (S / sum(sum(A))) - sum(sum(A(C==Cj,:)))^2 / (4*sum(sum(A))^2);
%             if Qtemp > Qmax
%                 Qmax = Qtemp;
%                 Cmax = Cj;
%             end
%         end
%         if Qmax > 0
%             C(u) = Cmax;
%             Q = Q + Qmax;
%             improve = 1;
%         end
%     end
% end
% end
