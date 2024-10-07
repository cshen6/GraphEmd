%% From https://github.com/WangJiangzhou/Fast-Network-Community-Detection-with-Profile-Pseudo-Likelihood-Methods
%% If you use this code, please check the above link and cite the proper paper

function [chatF,err,dT,post,log_CPL] = cpl4c(As,K,e,c,type,varargin)
% compute (un)conditional pseudo-likelihood estimate
% As  sparse Adjacency matrix
% e   the initial labeling
% c   the rue labeling (used only for computing/tracking error across iterations
% T   number of iterations on top of EM

%[As,K,e,c,type,varargin]=[mo.As, mo.K, e, mo.c, 'upm', cpl_opts]

options = struct('verbose',false,'verb_level',1,'conv_crit','pl', ...
    'em_max',100,'delta_max',0,'itr_num',20,'track_err',true); %default options
if nargin > 5
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

if options.track_err
    compErr = @(c,e) compMuI(compCM(c,e,K));         % use mutual info as a measure of error/sim.
end

rowSumNZ = @(X) X./repmat(sum(X,2),1,size(X,2)); % normalize row sum to 1
% swap= @(varargin) varargin{nargin:-1:1};         % swap two variables
epsilon = 1e-3;
regularize = @(x) (x+epsilon).*(x == 0) + x.*(x ~= 0);

LOG_REAL_MAX = log(realmax)-1;

T = options.itr_num;

% remove 0-degree nodes
zdNodes = sum(As,2) == 0;
nOrg = size(As,1);
% zdNum = nnz(zdIDX);
% AsOrg = As;
% eOrg = e;

As = As(~zdNodes,~zdNodes);
e = e(~zdNodes);
if options.track_err
    c = c(~zdNodes);  % not a very good idea !
end
n = size(As,1);


% Compute initial Phat
Phat = zeros(K);
for k = 1:K
    for ell = 1:K
        Phat(k,ell) = mean( reshape( As( e == k, e == ell ), [], 1) );
        
        Phat(k,ell) = regularize( Phat(k,ell) );
    end
end


% Compute initial Rhat
Rhat = zeros(K);
for k = 1:K
    Rhat(k,k) = sum( e == k ) / n;
    
    Rhat(k,k) = regularize( Rhat(k,k) );
    
end

% Compute inital Lambda hat and Theta hat
Lambdah = n*Phat*Rhat'; 
Thetah = rowSumNZ(Lambdah);

% Compute inital community prior estimates
pih = diag(Rhat);
%pih = rand(K,1); pih = pih/sum(pih);


% Compute block compressions
Bs = compBlkCmprss(As,e,K);

% initial the err vector
if options.track_err
    err = zeros(1,T+1);
    err(1) = compErr(c,e);
else
    err = 0;
end
% chat = zeros(n,1);

emN = options.em_max;           % max. EM steps
%deltaMax = 2/n;
if options.delta_max == 0
    switch options.conv_crit
        case 'param'
            deltaMax = 1e-3; % max. error below which EM terminates
        case 'label'
            deltaMax = 1e-2;    
        case 'pl'
            deltaMax = 1e-2;    
    end
else
    deltaMax = options.delta_max;
end
% initial chat
chat = e;
% chatOld = chat;

switch lower(type)
    case 'upl'
        % unconditional PL
        if options.verbose
            fprintf(1,'\nupl: %3d iterations\n',T)
        end
        tic
        for t = 2:(T+1)
            
            delVec = zeros(emN,1);
            OVF_FLAG = false;
            
            delta = inf;
            nu = 1;
            CONVERGED = false;
            while (nu <= emN) && (~CONVERGED) %(delta > deltaMax)
                % Z is K x n
                Z = -Lambdah * ones(K,n) + log(Lambdah)*Bs';
                Zmean = mean(Z);
                Z = Z - repmat(Zmean,K,1);
                 
                [ZZ, OVF] = handleOverflow(Z,LOG_REAL_MAX); 
                
                U = exp( ZZ );
                if OVF
                    OVF_FLAG = true;
                end
                
                alpha = repmat(pih(:),1,n).*U;
                
                post_denom = sum(alpha);                             
                
%                 alphatemp = alpha;
                
                alpha = alpha ./ regularize( repmat(post_denom,K,1) );
                
                plVal = sum( log(post_denom) ) + sum(Zmean);
                %plVal = sum( log(post_denom) );
                
                % alpha is K x n -- This is posterior prob. of labels
                % Bs is n x K
                % Lambdah is K x K
                
                if any(isnan(alpha(:)))
                     error('Something went wrong, pih will havve NaN entries.')
                 end

                
%                 pihold = pih;

                pih = mean(alpha,2); 
%                 disp(pih)
                
%                  if any(isnan(pih))
%                      error('Something went wrong, pih has NaN entries.')
%                  end
%                             
                Lambdah = regularize( ...
                        diag(1./ regularize(pih) )*(alpha*Bs/n) );
                

%                  [~, chat] = max(alpha',[],2);
                [~,chat] = max(alpha,[],1);
                chat = chat(:);
                
%                 if options.verb_level > 1
%                     disp(pih)
%                     disp(Lambdah)
%                 end
%                 delta = mean( chat ~= chatOld );
%                 chatOld = chat;
                
                if nu ~= 1
                    delta = abs((plVal - plValOld)/plValOld);
                    CONVERGED = delta < deltaMax;
                    delVec(nu-1) = delta;
                end
                plValOld = plVal;
                
%                 if nu == 1
% %                     switch options.conv_crit
% %                         case 'param'
% %                             
% %                         case 'label'
% %                         case 'pl'
% %                     end
%                     if options.c_term
%                         chatOld = chat;
%                     else
%                         LambdahOld = Lambdah;
%                         pihOld = pih;
%                     end
%                 else
%                     if options.c_term
%                         delta = mean( chat ~= chatOld );
%                     else
%                          delta = max([ ...
%                              norm(Lambdah - LambdahOld)/norm(LambdahOld) ...
%                              norm(pih - pihOld)/norm(pihOld) ...
%                          ]);
%                         
%                     end
%                 end

                if options.verbose && (options.verb_level > 1)
                    print_pl_decay(nu,delta,CONVERGED)
                end
                
                nu = nu + 1;
            end % end while
            
%             if options.verbose && (options.verb_level > 1)
%                 figure(1), clf,
%                 plot(1:(nu-1),delVec(1:(nu-1)),'b.-')
%                 
%                 if OVF_FLAG
%                     title('OVERFLOW')
%                 end
%                 pause(1)
%             end
            
            if options.track_err
                err(t) = compErr(c,chat);
            end
            
            Bs = compBlkCmprss(As,chat,K);
                       
            if options.verbose
                fprintf(1,'  > itr. %2d >> EM terminated after %2d steps, final delta = %3.6f\n',t-1, nu-1, delta)
            end
        end
        
%         if ~options.track_err
%             % we report error on 0-degree removed graph
%             err = compErr(c,chat);
%         end
        
        chatF = zeros(nOrg,1);
        chatF(~zdNodes) = chat;
        zdPrior = (pih(:).*sum(exp(-Lambdah),2))';
        zdPrior = zdPrior/sum(zdPrior);
        
        if any(isnan(zdPrior))
%             chatF(zdNodes) = randsrc(nnz(zdNodes),1,[1:K]);
              tempP = mnrnd(1,(1:K)/K, nnz(zdNodes))';
        else
%             chatF(zdNodes) = randsrc(nnz(zdNodes),1,[1:K;zdPrior]);
              zdPrior = zdPrior / sum(zdPrior);
              tempP = mnrnd(1,zdPrior, nnz(zdNodes))';
        end
        [chatF(zdNodes),~] = find(tempP);
        
        post = zeros(K, nOrg);
        post(:,~zdNodes) = alpha;
        post(:,zdNodes) = tempP;
         
        
        dT = toc;
        if options.verbose
            fprintf(1,'... in dT = %5.5f\n',dT)
        end

    case 'upm'
        Bs = compBlkCmprss(As,e,K);
        % unconditional PM
        if options.verbose
            fprintf(1,'\nupm: %3d iterations\n',T)
        end
        tic
        
        Iter=zeros(1,T+1);
        Iter(1)=100;        
        for t = 2:(T+1)
            n_K=ones(1,K);
            for k=1:K
                n_K(k)=sum(chat==k);
            end
            
            delVec = zeros(emN,1);
            OVF_FLAG = false;
            
            delta = inf;
            nu = 1;
            CONVERGED = false;
            while (nu <= emN) && (~CONVERGED) %(delta > deltaMax)
                % Z is K x n
                Z = Bs * log(Phat+1e-30)'+ (repmat(n_K,n,1)-Bs)*log(1-Phat+1e-30)'+repmat(log(pih'+1e-30),n,1);
                [maxZ,~]=max(Z,[],2);
                Z=Z-repmat(maxZ,1,K);
                [ZZ, OVF] = handleOverflow(Z,LOG_REAL_MAX);
                U = exp( ZZ );
                if OVF
                    OVF_FLAG = true;
                end                
                alpha=U./(repmat(sum(U,2),1,K)+1e-30);
                
                
                
                
                plVal=sum(sum((alpha'*Bs).*log(Phat+1e-30)))+...
                    sum(sum(alpha'*(repmat(n_K,n,1)-Bs).*log(1-Phat+1e-30)))+sum(alpha,1)*log(pih+1e-30);
                %plVal = sum( log(post_denom) );
                
                % alpha is n x K -- This is posterior prob. of labels
                % Bs is n x K
                % Lambdah is K x K
                
                if any(isnan(alpha(:)))
                     error('Something went wrong, pih will havve NaN entries.')
                 end

                
%                 pihold = pih;

                pih = mean(alpha,1)'; 
%                 disp(pih)
                
%                  if any(isnan(pih))
%                      error('Something went wrong, pih has NaN entries.')
%                  end
%               
                Phat=(alpha'*Bs)./(sum(alpha,1)'*n_K+1e-30);
                

%                  [~, chat] = max(alpha',[],2);
%                 [~,chat] = max(alpha,[],2);
%                 chat = chat(:);
                
%                 if options.verb_level > 1
%                     disp(pih)
%                     disp(Lambdah)
%                 end
%                 delta = mean( chat ~= chatOld );
%                 chatOld = chat;
                
                if nu ~= 1
                    delta = abs((plVal - plValOld)/plValOld);
                    CONVERGED = delta < deltaMax;
                    delVec(nu-1) = delta;
                end
                plValOld = plVal;
                
%                 if nu == 1
% %                     switch options.conv_crit
% %                         case 'param'
% %                             
% %                         case 'label'
% %                         case 'pl'
% %                     end
%                     if options.c_term
%                         chatOld = chat;
%                     else
%                         LambdahOld = Lambdah;
%                         pihOld = pih;
%                     end
%                 else
%                     if options.c_term
%                         delta = mean( chat ~= chatOld );
%                     else
%                          delta = max([ ...
%                              norm(Lambdah - LambdahOld)/norm(LambdahOld) ...
%                              norm(pih - pihOld)/norm(pihOld) ...
%                          ]);
%                         
%                     end
%                 end

                if options.verbose && (options.verb_level > 1)
                    print_pl_decay(nu,delta,CONVERGED)
                end
                
                nu = nu + 1;
            end % end while
            
%             if options.verbose && (options.verb_level > 1)
%                 figure(1), clf,
%                 plot(1:(nu-1),delVec(1:(nu-1)),'b.-')
%                 
%                 if OVF_FLAG
%                     title('OVERFLOW')
%                 end
%                 pause(1)
%             end
            
            n_v=sum(alpha,1);
            B_star=As*alpha;
            Score=B_star*log(Phat+1e-30)'+(repmat(n_v,n,1)-B_star)*log(1-Phat+1e-30)';
            [~,chat] = max(Score,[],2);
            chat = chat(:);            

            if options.track_err
                err(t) = compErr(c,chat);
            end
            
            Bs = compBlkCmprss(As,chat,K);
                       
            if options.verbose
                fprintf(1,'  > itr. %2d >> EM terminated after %2d steps, final delta = %3.6f\n',t-1, nu-1, delta)
            end
            
            Iter(t)=nu;            
            if Iter(t-1)==3
                break
            end            
        end
        
%         if ~options.track_err
%             % we report error on 0-degree removed graph
%             err = compErr(c,chat);
%         end
        
        chatF = zeros(nOrg,1);
        chatF(~zdNodes) = chat;
        zdPrior = (pih(:).*sum(exp(-Lambdah),2))';
        zdPrior = zdPrior/sum(zdPrior);
        
        if any(isnan(zdPrior))
%             chatF(zdNodes) = randsrc(nnz(zdNodes),1,[1:K]);
              tempP = mnrnd(1,(1:K)/K, nnz(zdNodes))';
        else
%             chatF(zdNodes) = randsrc(nnz(zdNodes),1,[1:K;zdPrior]);
              zdPrior = zdPrior / sum(zdPrior);
              tempP = mnrnd(1,zdPrior, nnz(zdNodes))';
        end
        [chatF(zdNodes),~] = find(tempP);
        
        post = zeros(K, nOrg);
        post(:,~zdNodes) = alpha';
        post(:,zdNodes) = tempP;
         
        
        dT = toc;
        if options.verbose
            fprintf(1,'... in dT = %5.5f\n',dT)
        end
        
    case 'cpm'
        % conditional PM
        if options.verbose
            fprintf(1,'\ncpl: %3d iterations\n',T)
        end
        
        
        tic
        for t = 2:(T+1)
            
            delVec = zeros(emN,1);
            
            delta = inf;
            nu = 1;
            CONVERGED = false;
            while (nu <= emN) && (~CONVERGED) %(delta > deltaMax)
                
                betah = zeros(n,K);
                for ell = 1:K
%                     betah(:,ell) = pih(ell)*prod(repmat(Thetah(ell,:),n,1).^Bs,2);
                    temp = repmat(Thetah(ell,:),n,1);
                    
                    betah(:,ell) = ...
                        pih(ell)*exp(sum(Bs.*log(temp),2));
                end
                
                post_denom = sum(betah,2);
                 
                betah = betah ./ regularize( repmat(post_denom,1,K) );
                % betah = rowSumNZ( regularize(betah) );

                pih = mean(betah,1)';
                
                Thetah = rowSumNZ( regularize(betah'*Bs) );

                [~, chat] = max(betah,[],2);
                 
                cplVal = sum( log(post_denom) );
                
%                 delta = mean( chat ~= chatOld );
%                 chatOld = chat;

                if nu ~= 1
                    delta = abs((cplVal - cplValOld)/(cplValOld+eps));
                    CONVERGED = delta < deltaMax; 
                    delVec(nu-1) = delta;
                end
                cplValOld = cplVal;

%                 if nu == 1
%                     if options.c_term
%                         chatOld = chat;
%                     else
%                         ThetahOld = Thetah;
%                         pihOld = pih;
%                     end
%                 else
%                     if options.c_term
%                         delta = mean( chat ~= chatOld );
%                     else
%                          delta = max([ ...
%                              norm(Thetah - ThetahOld)/norm(ThetahOld) ...
%                              norm(pih - pihOld)/norm(pihOld)]);
% %                         delta = max([ ...
% %                             norm(Thetah - ThetahOld) ...
% %                             norm(pih - pihOld)]);
%                     end
%                 end                
                             
                       
                
                if options.verbose && (options.verb_level > 1)
                    print_pl_decay(nu,delta,CONVERGED)
                end
                
                nu = nu + 1;  
            end
            
%             if options.verbose && (options.verb_level > 1)
%                 figure(1), clf,
%                 plot(1:(nu-1),delVec(1:(nu-1)),'r.-')
%                 pause(1)
%             end
            
            if options.track_err
                err(t) = compErr(c,chat);
            end
            
            Bs=As*betah;
            if options.verbose
                fprintf(1,'  > itr. %2d >> EM terminated after %2d steps, final delta = %3.6f\n',t-1, nu-1, delta)
            end

        end
        
%         if ~options.track_err
%             % we report error on 0-degree removed graph
%             err = compErr(c,chat);
%         end
        
        chatF = zeros(nOrg,1);
        chatF(~zdNodes) = chat;
        zdPrior = pih(:)';
        
        if any(isnan(zdPrior))
%             chatF(zdNodes) = randsrc(nnz(zdNodes),1,[1:K]);
              tempP = mnrnd(1,(1:K)/K, nnz(zdNodes))';
        else
%             chatF(zdNodes) = randsrc(nnz(zdNodes),1,[1:K;zdPrior]);
              zdPrior = zdPrior / sum(zdPrior);
              tempP = mnrnd(1,zdPrior, nnz(zdNodes))';
        end
        [chatF(zdNodes),~] = find(tempP);
        
        post = zeros(K, nOrg);
        post(:,~zdNodes) = betah';
        post(:,zdNodes) = tempP;
                   
        
        dT = toc;
        if options.verbose
            fprintf(1,'... in dT = %5.5f\n',dT)
        end  
        
        
    case 'cppl'
        % conditional PPL
        if options.verbose
            fprintf(1,'\ncpl: %3d iterations\n',T)
        end
        
        
        tic
        for t = 2:(T+1)
            
            delVec = zeros(emN,1);
            
            delta = inf;
            nu = 1;
            CONVERGED = false;
            while (nu <= emN) && (~CONVERGED) %(delta > deltaMax)
                
                betah = zeros(n,K);
                for ell = 1:K
%                     betah(:,ell) = pih(ell)*prod(repmat(Thetah(ell,:),n,1).^Bs,2);
                    temp = repmat(Thetah(ell,:),n,1);
                    
                    betah(:,ell) = ...
                        pih(ell)*exp(sum(Bs.*log(temp),2));
                end
                
                post_denom = sum(betah,2);
                 
                betah = betah ./ regularize( repmat(post_denom,1,K) );
                % betah = rowSumNZ( regularize(betah) );

                pih = mean(betah,1)';
                
                Thetah = rowSumNZ( regularize(betah'*Bs) );

                %[~, chat] = max(betah,[],2);
                 
                cplVal = sum( log(post_denom) );
                
%                 delta = mean( chat ~= chatOld );
%                 chatOld = chat;

                if nu ~= 1
                    delta = abs((cplVal - cplValOld)/(cplValOld+eps));
                    CONVERGED = delta < deltaMax; 
                    delVec(nu-1) = delta;
                end
                cplValOld = cplVal;

%                 if nu == 1
%                     if options.c_term
%                         chatOld = chat;
%                     else
%                         ThetahOld = Thetah;
%                         pihOld = pih;
%                     end
%                 else
%                     if options.c_term
%                         delta = mean( chat ~= chatOld );
%                     else
%                          delta = max([ ...
%                              norm(Thetah - ThetahOld)/norm(ThetahOld) ...
%                              norm(pih - pihOld)/norm(pihOld)]);
% %                         delta = max([ ...
% %                             norm(Thetah - ThetahOld) ...
% %                             norm(pih - pihOld)]);
%                     end
%                 end                
                             
                       
                
                if options.verbose && (options.verb_level > 1)
                    print_pl_decay(nu,delta,CONVERGED)
                end
                
                nu = nu + 1;  
            end
            
%             if options.verbose && (options.verb_level > 1)
%                 figure(1), clf,
%                 plot(1:(nu-1),delVec(1:(nu-1)),'r.-')
%                 pause(1)
%             end
            Score=(As*betah)*log(Thetah);
            [~, chat] = max(Score,[],2);
            
            %%%%
            if options.track_err
                err(t) = compErr(c,chat);
            end
            
            Bs = compBlkCmprss(As,chat,K);
            
            if options.verbose
                fprintf(1,'  > itr. %2d >> EM terminated after %2d steps, final delta = %3.6f\n',t-1, nu-1, delta)
            end

        end
        
%         if ~options.track_err
%             % we report error on 0-degree removed graph
%             err = compErr(c,chat);
%         end
        
        chatF = zeros(nOrg,1);
        chatF(~zdNodes) = chat;
        zdPrior = pih(:)';
        
        if any(isnan(zdPrior))
%             chatF(zdNodes) = randsrc(nnz(zdNodes),1,[1:K]);
              tempP = mnrnd(1,(1:K)/K, nnz(zdNodes))';
        else
%             chatF(zdNodes) = randsrc(nnz(zdNodes),1,[1:K;zdPrior]);
              zdPrior = zdPrior / sum(zdPrior);
              tempP = mnrnd(1,zdPrior, nnz(zdNodes))';
        end
        [chatF(zdNodes),~] = find(tempP);
        
        post = zeros(K, nOrg);
        post(:,~zdNodes) = betah';
        post(:,zdNodes) = tempP;
                   
        
        dT = toc;
        if options.verbose
            fprintf(1,'... in dT = %5.5f\n',dT)
        end          
    otherwise
        % conditional PL
        if options.verbose
            fprintf(1,'\ncpl: %3d iterations\n',T)
        end
        
        
        tic
        log_CPL=zeros(1,T+1);
        for t = 2:(T+1)
            
            delVec = zeros(emN,1);
            
            delta = inf;
            nu = 1;
            CONVERGED = false;
            while (nu <= emN) && (~CONVERGED) %(delta > deltaMax)
                logbetah=log(repmat(pih',n,1))+Bs*log(Thetah');
                betah=exp(logbetah-repmat(max(logbetah,[],2),1,K));
%                 betah = zeros(n,K);
%                 for ell = 1:K
% %                     betah(:,ell) = pih(ell)*prod(repmat(Thetah(ell,:),n,1).^Bs,2);
%                     temp = repmat(Thetah(ell,:),n,1);
%                     
%                     betah(:,ell) = ...
%                         pih(ell)*exp(sum(Bs.*log(temp),2));
%                 end
                
                post_denom = sum(betah,2);
                 
                betah = betah ./ regularize( repmat(post_denom,1,K) );
                % betah = rowSumNZ( regularize(betah) );

                pih = mean(betah,1)';
                
                Thetah = rowSumNZ( regularize(betah'*Bs) );

                [~, chat] = max(betah,[],2);
                 
                cplVal = sum( log(post_denom) );
                
%                 delta = mean( chat ~= chatOld );
%                 chatOld = chat;

                if nu ~= 1
                    delta = abs((cplVal - cplValOld)/(cplValOld+eps));
                    CONVERGED = delta < deltaMax; 
                    delVec(nu-1) = delta;
                end
                cplValOld = cplVal;

%                 if nu == 1
%                     if options.c_term
%                         chatOld = chat;
%                     else
%                         ThetahOld = Thetah;
%                         pihOld = pih;
%                     end
%                 else
%                     if options.c_term
%                         delta = mean( chat ~= chatOld );
%                     else
%                          delta = max([ ...
%                              norm(Thetah - ThetahOld)/norm(ThetahOld) ...
%                              norm(pih - pihOld)/norm(pihOld)]);
% %                         delta = max([ ...
% %                             norm(Thetah - ThetahOld) ...
% %                             norm(pih - pihOld)]);
%                     end
%                 end                
                             
                       
                
                if options.verbose && (options.verb_level > 1)
                    print_pl_decay(nu,delta,CONVERGED)
                end
                
                nu = nu + 1;  
            end
            
%             if options.verbose && (options.verb_level > 1)
%                 figure(1), clf,
%                 plot(1:(nu-1),delVec(1:(nu-1)),'r.-')
%                 pause(1)
%             end
            
            if options.track_err
                err(t) = compErr(c,chat);
            end
            
            Bs = compBlkCmprss(As,chat,K);
            log_CPL(t)=logCPL(Bs, betah, Thetah);
            if options.verbose
                fprintf(1,'  > itr. %2d >> EM terminated after %2d steps, final delta = %3.6f\n',t-1, nu-1, delta)
            end

        end
        
%         if ~options.track_err
%             % we report error on 0-degree removed graph
%             err = compErr(c,chat);
%         end
        
        chatF = zeros(nOrg,1);
        chatF(~zdNodes) = chat;
        zdPrior = pih(:)';
        
        if any(isnan(zdPrior))
%             chatF(zdNodes) = randsrc(nnz(zdNodes),1,[1:K]);
              tempP = mnrnd(1,(1:K)/K, nnz(zdNodes))';
        else
%             chatF(zdNodes) = randsrc(nnz(zdNodes),1,[1:K;zdPrior]);
              zdPrior = zdPrior / sum(zdPrior);
              tempP = mnrnd(1,zdPrior, nnz(zdNodes))';
        end
        [chatF(zdNodes),~] = find(tempP);
        
        post = zeros(K, nOrg);
        post(:,~zdNodes) = betah';
        post(:,zdNodes) = tempP;
                   
        
        dT = toc;
        if options.verbose
            fprintf(1,'... in dT = %5.5f\n',dT)
        end        
end


end

function Bs = compBlkCmprss(As,e,K)
% Compute block compression
%
% As  sparse adj. matrix
% e   labeling to use
% K   number of communities

n = size(As,1);

[~, I] = sort(e);
Bs = spalloc(n,K,nnz(As));
for k = 1:K
   Bs(:,k) = sum(As(:,I(e(I) == k)),2);
end
%Ds = sum(Bs,2)

end

function Bs = softcompBlkCmprss(As,e,~)
% Compute block compression
%
% As  sparse adj. matrix
% e   labeling posterior probability to use n by K
% K   number of communities
Bs=As*e;
end

function [ZZ,OVF] = handleOverflow(Z,LOG_MAX)
    Zmax = max(abs(Z(:)));
   
    if Zmax > LOG_MAX
        ZZ = (LOG_MAX / Zmax) * Z;
        OVF = true;
    else
        ZZ = Z;
        OVF = false;
    end
end

function print_pl_decay(nu,delta,CONVERGED)
    if nu == 1
        fprintf(1,'    .... ')
    else
        fprintf(1,'%3.5f ',delta)
        if (mod(nu,5) == 0) || CONVERGED 
            fprintf(1,'\n    .... ');
        end
    end
end

function normMuI = compMuI(CM)
% normMUI Computes the normalized mutual information between two clusters
%
% CM   confusion matrix

N = sum(CM(:));
normCM = CM/N; % normalized confusion matrix

IDX = CM == 0; % index of zero elements of CM so that we can avoid them

jointEnt = - sum( (normCM(~IDX)).*log(normCM(~IDX)) );

indpt = sum(normCM,2) * sum(normCM,1);
muInfo = sum(normCM(~IDX) .* log(normCM(~IDX) ./ indpt(~IDX)) );

normMuI = muInfo / jointEnt;
end


function M = compCM(c,e,K)
% Compute the confusion matrix between labels "c" and "e"
%
% c,e Two sets of labels
% K   number of labels in both "c" and "e"

M = zeros(K);
for k = 1:K
    for r = 1:K
        M(k,r) = sum( (c(:) == k) .* ( e(:) == r ) );
    end
end
end

function [CPL_logLikelihood,res]=logCPL(Bs, betah, Thetah)

c_matrix=betah;% n-by-K matrix
K=size(c_matrix,2);

Delta=zeros(1,K);
for k=1:K
    Delta(k)=Bs(:,k)'*c_matrix*log(Thetah(:,k)+1e-40);
end

CPL_logLikelihood=-sum(sum(log(factorial(Bs))))+sum(Delta);
res=3;
end