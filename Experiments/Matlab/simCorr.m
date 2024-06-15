function simCorr(choice, n,rho, reps)

if nargin<1
    choice=1;
end

if choice<20
    if nargin<2
        n=100;
    end
    if nargin<3
        rho=0;
    end
    if nargin<4
        reps=300;
    end
    spec=true;
    K=10;
    type=600+(choice);
    if choice>10
        type=type-10;
    end
    nn=100;lim=6;dd=20;
    % if choice>3
    %     nn=200;
    % end
    power0=zeros(lim,1);
    power1=zeros(K,K,lim);
    power2=zeros(lim,1);
    power3=zeros(lim,1);
    time1=zeros(lim,reps);
    time2=zeros(lim,reps);
    time3=zeros(lim,reps);
    alpha=0.05;
    stat=zeros(lim,reps);
    corrCom=zeros(K,K,reps,lim);
    for l=1:lim;
        n=nn*l
        for r=1:reps
            % r
            [Dis,Y]=simGenerate(type,n,K,0);
            A=Dis{1}; B=Dis{2};
            % A=rand(n,1);B=rand(n,1);
            % A=squareform(pdist(A));B=squareform(pdist(B));Y=randi(K,n,1);
            % [B,~]=simGenerate(21,n,K,0);
            tic
            if choice<=10
                [stat(l,r),pval,corrCom(:,:,r,l),pvalCom]=GraphCorr(A,B,Y);
            else
                [stat(l,r),pval,corrCom(:,:,r,l),pvalCom]=GraphCorr(A,B,K);
            end
            time1(l,r)=toc;
            if spec
                tic
                Z1=ASE(A,dd);Z2=ASE(B,dd);
                tmp=toc;
                tic
                [~,pval2]=DCorFastTest(Z1,Z2);
                % [~,pval2]=GraphCorr(A,B,K,1);
                time2(l,r)=toc+tmp;
                power2(l)=power2(l)+(pval2<alpha)/reps;
                tic
                [~,pval3]=DCorPartialTest(Z1,Z2,Y);
                % [~,pval3]=GraphCorr(A,B,K,10);
                time3(l,r)=toc+tmp;
                power3(l)=power3(l)+(pval3<alpha)/reps;
            end
            power0(l)=power0(l)+(pval<alpha)/reps;
            power1(:,:,l)=power1(:,:,l)+(pvalCom<alpha)/reps;
        end
    end

    save(strcat('GEECorSim',num2str(choice),'Corr',num2str(rho),'.mat'),'nn','lim','type','rho','reps','power0','power1','power2','n','stat','corrCom','time1','time2','time3','power3');
end


if choice>20 && choice<30
    switch choice
        case 21
            load('CElegans.mat');G1=Ac;G2=Ag;Y=vcols;
        % case 12
        %     load('anonymized_msft.mat');G1=edge2adj(G{1});G2=edge2adj(G{3});Y=label;
        case 23
            load('Wiki_Data.mat'); G1=GEAdj;G2=GFAdj;Y=Label;
        case 24
            load('Letter.mat'); G1=edge2adj(Edge1); G2=edge2adj(Edge2);Y=Label1;n=10507;G2=G2(1:n,1:n);
        case 25
            load('Letter.mat'); G1=edge2adj(Edge1); G2=edge2adj(Edge3);Y=Label1;n=10507;G2=G2(1:n,1:n);
        case 26
            load('Wiki_Data.mat'); G1=max(max(TE))-TE;G2=GEAdj;Y=Label;
        case 27
            load('Wiki_Data.mat'); G1=max(max(TF))-TF;G2=GFAdj;Y=Label;
        case 28
            load('Wiki_Data.mat'); G1=max(max(TE))-TE;G2=max(max(TF))-TF;Y=Label;
    end
    [stat,pval,corrCom,pvalCom]=GraphCorr(G1,G2,Y);
    d=10;
    Z1=ASE(G1,d);Z2=ASE(G2,d);
    [stat2,pval2]=DCorFastTest(Z1,Z2);
     [stat3,pval3]=DCorPartialTest(Z1,Z2,Y);
     save(strcat('GEECorSim',num2str(choice),'.mat'),'pval','pvalCom','pval2','pval3','corrCom','pvalCom','stat','stat2','stat3');
end
