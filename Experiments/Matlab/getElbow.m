
% Given a decreasingly sorted vector, return the given number of elbows
% dat: a input vector (e.g. a vector of standard deviations), or a input feature matrix.
% n: the number of returned elbows.
% q: a vector of length n. Typically 1st or 2nd elbow suffices
% Reference: Zhu, Mu and Ghodsi, Ali (2006), "Automatic dimensionality selection from the scree plot via the use of profile likelihood", Computational Statistics & Data Analysis, Volume 51 Issue 2, pp 918-930, November, 2006.
function q=getElbow(d, n)
if nargin<2
    n=3;
end
p=length(d);
q=getElbow2(d);
for i=2:n
    if q(i-1)>=p
        break;
    else
        q=[q,q(i-1)+getElbow2(d(q(i-1)+1:end))];
    end
end
if length(q)<n
    q=[q,q(end)*ones(1,n-length(q))];
end

function q=getElbow2(d)
p=length(d);
lq=zeros(p,1);
for i=1:p
    mu1 = mean(d(1:i));
    mu2 = mean(d(i+1:end));
    sigma2 = (sum((d(1:i) - mu1).^2) + sum((d(i+1:end) - mu2).^2)) / (p - 1 - (i < p));
    lq(i) = sum( log(normpdf(  d(1:i), mu1, sqrt(sigma2)))) + sum( log(normpdf(  d(i+1:end), mu2, sqrt(sigma2))));
end
[~,q]=max(lq);