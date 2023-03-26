function D = DistSparse(X,thres)

if nargin<2
    thres=0.2;
end
% if nargin<3
%     dist='cosine';
% end

% 
% Dist1='cosine';Dist2=Dist1; spec=1;GNN=0;
lim=1000;
[n,d]=size(X);
D=zeros(lim*n,3);
for j=1:n
    X(j,:) = normalize(X(j,:),2,'norm');
end

s=1;
for i=1:n
    D(s,1)=i;D(s,2)=i;D(s,3)=1;
    s=s+1;
    t=1;
    for j=i+1:n
        if t>lim
            break;
        else
            tmp=1-dot(X(i,:),X(j,:));
            if tmp>thres
                D(s,1)=i;D(s,2)=j;D(s,3)=tmp;t=t+1;s=s+1;
            end
        end
    end
end
D=D(1:s-1,:);
