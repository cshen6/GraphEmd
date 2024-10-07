%% From https://github.com/WangJiangzhou/Fast-Network-Community-Detection-with-Profile-Pseudo-Likelihood-Methods
%% If you use this code, please check the above link and cite the proper paper

function [e dT] = initLabel5b(As,K,type,varargin)
% Generate initial labeling
% - As    sparse adjacency matrix
% - K     number of communities 
% - type  'spc' spectral clustering of the supernodes, the fourth argument
% will be "d"
%

options = struct('verbose',false,'perturb',false,'rhoPert',0.25, ...
                 'normalize',false,'itrNum',5, 'degPert', 0.01);  %default options
if nargin > 3
    % process options
    optNames = fieldnames(options);
    passedOpt = varargin{end};
  
    if ~isstruct(passedOpt), 
        error('Last argument should be an options struct.')
    end
    for fi = reshape(fieldnames(passedOpt),1,[])
        if any(strmatch(fi,optNames))
            options.(fi{:}) = passedOpt.(fi{:});
        else
            error('Unrecognized option name: %s', fi{:})
        end
    end
end

n = size(As,1);
avgDeg = full(mean(sum(As,2))); % average degree of As

% if options.perturb
%     Bs = genBlkMod(ones(n,1),log(n)/n,log(n)*n);
%     Bs = Bs + Bs';
%     
%     
%     As = As + Bs * options.rhoPert * avgDeg/log(n);
%     if options.verbose
%         fprintf(1,'Adj. matrix perturbed by %3.3f\n', options.rhoPert)
%     end
% else
%     if options.verbose
%         disp('No perturbation.')
%     end
% end

switch lower(type)
    case 'hic'
        % hierarchical clustering
        tic
        
        Deg = sum(As,2);
        zdI = Deg == 0;
        np = sum(~zdI);
        
        Cs = As(~zdI,~zdI);
        
        W = Cs'*Cs;
        dW = diag(diag(W).^(-1/2));
        Wn = dW * W * dW;

        D = full(1-Wn(tril(true(np),-1))');

        Z = linkage(D,'average');
        T = cluster(Z,'maxclust',K);
        %dendrogram(Z)

        e = zeros(n,1);
        e(~zdI) = T;
        e(zdI) = randsrc(n-np,1,1:K);
        
        dT = toc;

    case 'true'
        % set the initial condition to be a percent of the true labels
        % forth argument should be the vector of true labels
        % fifth argument should be the percentage of random assignments, 
        %   i.e. roughly incorrect assignments 
        tic
        c = varargin{1};
        alpha = varargin{2};
        
        n = numel(c);
        n0 = floor(alpha*n);
        
        I = randperm(n);
        e = c(:);
        
        e(I(1:n0)) = randsrc(n0,1,1:K);
        dT = toc;
     
    case 'gibbs'
        % approximate gibbs sampling
        tic
        e = ones(n,1);
        alpha = varargin{1};
        gamma = varargin{2};
        n = size(As,1);

        Nt = 10*n;
%         err = zeros(Nt,1);
%         T = zeros(7,1);
        for t = 1:Nt
%           tic
            %i = randsrc(1,1,1:n);
            i = 1 + mod(t-1,n);
%           T(1) = T(1) + toc; 

%            tic
            Neib = find(As(i,:));
            NeibSize = numel(Neib);
            NeibC = setdiff(1:n,[Neib i]);
            NeibCSize = numel(NeibC);
%            T(2) = T(2) + toc;

            condPi = zeros(1,K);
            for k = 1:K
%                 tic
                NeibMatch = sum(e(Neib) == k);
%                 T(3) = T(3) + toc;

%                 tic
                NeibCMatch = sum(e(NeibC) == k);
%                 T(4) = T(4) + toc;

%                 tic
                condPi(k) = alpha^NeibMatch * gamma^(NeibSize-NeibMatch) ...
                    * (1-alpha)^NeibCMatch * (1-gamma)^(NeibCSize - NeibCMatch);
%                 T(5) = T(5) + toc;
            end
%             tic
            condPi = condPi / sum(condPi);
            %disp(condPi)
            %e(i) = randsrc(1,1,[1:K;condPi]);
            [e(i),~] = find(mnrnd(1,condPi,1)');
%             T(6) = T(6) + toc;

%             tic
%             err(t) = compErr(e,c);
%             T(7) = T(7) + toc;
        end
        dT = toc;
        
    case 'sc'
        % spectral clustering
        tic
        
        G = infHandle(diag(sum(As,2).^(-0.5)));
        
        L = G*As*G;
        fun = @(x) L*x;
        
        opts.issym = 1;
        if options.verbose
            opts.disp = 2;
        end
        [U, ~] = eigs(fun,n,K,'LM',opts);      
        
        if options.verbose
            kmopts = statset('Display','iter');
        else
            kmopts = statset('Display','off'); 
        end
        kmIDX = kmeans(U(:,2:K),K,'replicates',10,...
            'onlinephase','off','Options',kmopts);
        
        e = reshape(repmat(kmIDX,1,1)',n,[]);
        dT = toc;
        

    case 'scp'
        % spectral clusteting with perturbation
        tic
        alpha0 = (options.rhoPert)*avgDeg;
             
        degh = sparse(sum(As,2) + alpha0);        
        Gh = infHandle(diag(degh.^(-0.5)));
        
        bh = sparse(Gh*ones(n,1));
        bhn = (alpha0/n)*bh;
        Lh = Gh*As*Gh;
        
        fun = @(x) Lh*x + (bh'*x)*bhn;
        
        opts.issym = 1;
        if options.verbose
            opts.disp = 2;
        end
        [U, ~] = eigs(fun,n,K,'LM',opts);
        
        if options.verbose
            kmopts = statset('Display','iter'); 
        else
            kmopts = statset('Display','off'); 
        end
        kmIDX = kmeans(U(:,2:K),K,'replicates',10, ...
            'onlinephase','off','Options',kmopts);
        
        e = reshape(repmat(kmIDX,1,1)',n,[]);
          
        dT = toc;    
        
    case 'degkm'
        tic
        deg = sum(As,2);
        
        e = kmeans(deg + options.degPert *randn(n,1),K,'replicates',10);
        dT = toc;
        
    case 'bi_deg'
        % degree clusterin
        tic
        degv = [sum(As,2) sum(As^2,2)];
        X = degv + options.degPert * randn(n,2);
        
        if options.verbose
           fprintf(1,'bi_deg: Degree perterb. = %3.3f\n',options.degPert)
        end
     
        if options.normalize
           X = X*diag((1./sum(X)));
           if options.verbose
               disp('bi_deg: Degrees normalized.')
           end
        end
        
        e = kmeans(X,K,'replicates',10,'onlinephase','off');
        dT = toc;
        
    case 'approx_spc'
        
        tic
        G = diag(sum(As,2).^(-0.5));
        L = G*As*G;
        
        Q = randn(n,ceil(log2(n)))/sqrt(n);
        F = [sum(As,2) sum(As^2,2)];
        Q(:,1:2) = F*diag((1./sum(F)));
        
        for k = 1:options.itrNum
             [Q,~] = qr(L*Q,0);
        end
        
        e = kmeans(Q + options.degPert*randn(size(Q)),K,'replicates',10) ;
        dT = toc;
        
    
        
    otherwise
       e = randsrc(n,1,1:K);  
end

end 


function y = infHandle(x)

y = x;
y(isinf(x)) = 0;
end
