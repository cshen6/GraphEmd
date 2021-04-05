function [error1,error2,error3,error4,time1,time2,time3,time4,std1,std2,std3,std4]=GraphNNSim(option,rep,n,d,num)
% encoder error, LDA error, SVM error, RF error

if nargin<1
    option=1;
end
if nargin<2
    rep=1;
end
if nargin<3
    n=1000;
end
if nargin<4
    d=2;
end
if nargin<5
    num=5;
end
% if nargin<3
%     lim=0.5;
% end
%numRange=n;
error1=zeros(rep,1);
error2=zeros(rep,1);
error3=zeros(rep,1);
error4=zeros(rep,1);
time1=0;
time2=0;
time3=0;
time4=0;
for r=1:rep
    [dis,Y,d,X]=generateSims(option,n,d);
    tic
    if option<2
        error1(r)=GraphNNEvaluate(dis,Y,num);
    else
        error1(r)=GraphNNEvaluate(X,Y,num);
    end
    time1=time1+toc;
    
    tic
    if option<2
        error2(r)=GraphSVM(dis,Y,2,d,num);
    else
        error2(r)=GraphSVM(X,Y,1,d,num);
    end
    time2=time2+toc;
    
    if option>2
        tic
        error3(r)=GraphSVM(X,Y,0); 
        time3=time3+toc;
        
        tic
        error4(r)=GraphSVM(X,Y,3);
        time4=time4+toc;
    end
end
std1=std(error1);
std2=std(error2);
std3=std(error3);
std4=std(error4);
error1=mean(error1);
error2=mean(error2);
error3=mean(error3);
error4=mean(error4);

