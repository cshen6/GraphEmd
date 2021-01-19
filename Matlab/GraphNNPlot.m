function [error1,error2,time1,time2]=GraphNNPlot(option)

% Plot time and error for GraphNN vs ASE * LDA with respect to sample size
if option==1
    ll=30;
    inte=100;
    error1=zeros(ll,1);
    error2=zeros(ll,1);
    time1=zeros(ll,1);
    time2=zeros(ll,1);
    for i=1:ll
        n=inte*i;
        [error1(i),error2(i),time1(i),time2(i)]=GraphNNSim(1,1,n,0);
    end
    subplot(1,2,1);
    plot(1:ll,error1,1:ll,error2,'LineWidth',2)
    ticks=1:ll;
    ticks=ticks*inte;
    xticklabels(ticks)
    %ylim([0,1])
    ylabel('Classification Error');
    xlabel('Sample Size');
    legend('Convolution * NN','ASE * LDA');
    subplot(1,2,2);
    plot(1:ll,time1,1:ll,time2,'LineWidth',2)
    xticklabels(ticks)
    ylabel('Running Time');
    xlabel('Sample Size');
    legend('Convolution * NN','ASE * LDA');
end

% Plot error for GraphNN vs ASE * LDA with respect to contamination
if option==2
    ll=10;
    n=200;
    inte=0.02;
    error1=zeros(ll,1);
    error2=zeros(ll,1);
    std1=zeros(ll,1);
    std2=zeros(ll,1);
    time1=zeros(ll,1);
    time2=zeros(ll,1);
    for i=1:ll
        contam=2-inte*i;
        [error1(i),error2(i),time1(i),time2(i),std1(i),std2(i)]=GraphNNSim(1,3,n,contam);
    end
    hold on
    errorbar(1:ll,error1,std1,'LineWidth',2)
    errorbar(1:ll,error2,std2,'LineWidth',2)
    hold off
    xticklabels(0.02:0.02:0.4)
    %ylim([0,1])
    ylabel('Classification Error');
    xlabel('Contamination');
    legend('Convolution * NN','ASE * LDA');
end