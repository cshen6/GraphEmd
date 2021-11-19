function XE=edgeEmd(edge,XV)

%%% Vertex to Edge embedding:
s=size(edge,1);
[~,d]=size(XV);
XE=zeros(s,2*d);
for i=1:s
    XE(i,:)=[XV(edge(i,1),:),XV(edge(i,2),:)];
end
end