function simAttributeGraph()

%%% Graph Processing into Annual
load('CTDC.mat')
Year=table2array(X(:,1));
Data=X(:,7:end); % remove categorical variables
Y=Y(7:end);Y=Y-2;
idx=~isnan(Year); Year=Year(idx,:); Data=Data(idx,:);% index by year
%Data(isnan(Data))=0; % all NaN numerical entries set to 0
[a,~,~]=unique(Year); % find all years
Year=Year-min(a)+1; % re-arrange starting year to 1

% initialize graph 
G=cell(size(a,1),1);
sz=size(Data,2);
for i=1:size(a,1)
    G{i}=zeros(sz,sz);
end

D1=table2array(Data);
for i=1:size(Data,1)
    tmpG=squareform(pdist(D1(i,:)'))+1;
    tmpG(isnan(tmpG))=0;
    ind=Year(i);
    G{ind}=G{ind}+tmpG;
end

for i=1:size(a,1)
    for j=1:length(Y)
    G{i}(j,j)=0;
    end
end

opts = struct('Normalize',true,'Refine',0,'Principal',0,'Laplacian',false,'Discriminant',false,'Softmax',false);
[Z,out]=GraphEncoder(G,Y,opts);
Z=reshape(Z,21,4,21);

xplot=zeros(21,2);
for i=1:size(a,1)
    tmp=1-squareform(pdist(Z(:,:,i),'cosine'));
    tmp(isnan(tmp))=0;
    G{i}=tmp;
    xplot(i,:)=tmp(1,2:3);
end

plot(1:21,xplot(:,1),1:21,xplot(:,2));