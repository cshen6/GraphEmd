function accuracy = metric_accuracy(P, Y_label)
% P:       Prediction probability matrix
% Y_label: ground truth labels

[~, predict] = max(P,[],2);
accuracy = length(find(predict==Y_label)) / size(P,1);
