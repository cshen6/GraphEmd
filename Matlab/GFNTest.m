k=3;
n=1000;
[Adj,Y]=simGenerate(1,n);
Y2=zeros(n,k);
for i=1:n
Y2(i,Y(i))=1;
end
[Z,W]=GraphEncoder(Adj,Y);
netGFN1 = patternnet(max(opts.neuron,k),'trainscg','crossentropy'); % number of neurons, Scaled Conjugate Gradient, cross entropy
trnSize=0.5;valSize=0.25;
netGFN1.divideParam.trainRatio = trnSize;
netGFN1.divideParam.valRatio   = valSize;
netGFN1.divideParam.testRatio  = 1-trnSize-valSize;
mdl1 = train(netGFN1,Z',Y2');

netGFN2 = patternnet(k,'trainscg','crossentropy');
netGFN2=configure(netGFN2,Adj,Y2');
netGFN2.divideParam.trainRatio = trnSize;
netGFN2.divideParam.valRatio   = valSize;
netGFN2.divideParam.testRatio  = 1-trnSize-valSize;
indNew=(netGFN2.inputs{1,1}.range(:,2)~=0);
netGFN2.IW{1,1} = repmat(W(indNew,:)',1,1);
netGFN2.b{1,1} = zeros(1*k,1);
mdl2 = train(netGFN2,Adj,Y2');