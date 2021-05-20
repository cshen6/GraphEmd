function dLdX = backward_dropout(dLdY, mask)
% Y = X .* mask

dLdX = dLdY .* mask;
