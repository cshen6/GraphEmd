function [loss,gradients] = GCNmodelLoss(parameters,X,A,T)

Y = GCNmodel(parameters,X,A);
loss = crossentropy(Y,T,DataFormat="BC");
gradients = dlgradient(loss, parameters);

end