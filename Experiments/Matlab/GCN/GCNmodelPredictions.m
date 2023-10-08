function predictions = GCNmodelPredictions(parameters,coulombData,adjacencyData,mu,sigsq,classes)

predictions = {};
numObservations = size(coulombData,3);

for i = 1:numObservations
    % Extract unpadded data.
    numNodes = find(any(adjacencyData(:,:,i)),1,"last");
    A = adjacencyData(1:numNodes,1:numNodes,i);
    X = coulombData(1:numNodes,1:numNodes,i);

    % Preprocess data.
    [A,X] = preprocessPredictors(A,X);
    X = (X - mu)./sqrt(sigsq);
    X = dlarray(X);

    % Make predictions.
    Y = model(parameters,X,A);
    Y = onehotdecode(Y,classes,2);
    predictions{end+1} = Y;
end

end