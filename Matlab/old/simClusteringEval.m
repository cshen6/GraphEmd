function simClusteringEval(nmax, opts)

% opts = struct('model','SBM','AEE',1,'ASE',1,'interval',20,'reps',200);
% simClustering(10000, opts)
% opts = struct('model','DCSBM','AEE',1,'ASE',1,'interval',20,'reps',200);
% simClustering(10000, opts)
% opts = struct('model','RDPG','AEE',1,'ASE',1,'interval',20,'reps',200);
% simClustering(10000, opts)

if nargin < 2
    opts = struct('model','SBM','AEE',1,'ASE',1,'interval',1,'reps',1); % default parameters
end
if ~isfield(opts,'model'); opts.model='SBM'; end
if ~isfield(opts,'AEE'); opts.AEE=1; end
if ~isfield(opts,'ASE'); opts.ASE=1; end
if ~isfield(opts,'interval'); opts.interval=1; end
if ~isfield(opts,'reps'); opts.reps=1; end
fs=15; d=3;K=3;
% theorem 1: SBM clustering
t_AEE=zeros(opts.interval,opts.reps); t_ASE=zeros(opts.interval,opts.reps);
RI_AEE=zeros(opts.interval,opts.reps); RI_ASE=zeros(opts.interval,opts.reps);RI=zeros(opts.interval,opts.reps);

nmin=floor(nmax/opts.interval);
nrange=nmin:nmin:nmin*opts.interval;

for i=1:opts.interval
    n=nrange(i);
    for r=1:opts.reps
        if strcmpi(opts.model,'SBM')
            [Adj,Y]=simGenerate(1,n);
        end
        if strcmpi(opts.model,'DCSBM')
            [Adj,Y]=simGenerate(2,n);
        end
        if strcmpi(opts.model,'RDPG')
            [Adj,Y]=simGenerate(3,n);
        end
        if opts.AEE==1
            tic
        ind_AEE=GraphClustering(Adj,K);
        t_AEE(i,r)=toc;
        RI_AEE(i,r)=RandIndex(Y,ind_AEE);
        end
        if opts.ASE==1
        tic
        [U,S,~]=svds(Adj,d);
        Z_ASE=U(:,1:d)*S(1:d,1:d)^0.5;
        ind_ASE = kmeans(Z_ASE, K);
        t_ASE(i,r)=toc;
        RI_ASE(i,r)=RandIndex(Y,ind_ASE);
        RI(i,r)=RandIndex(ind_AEE,ind_ASE);
        end
    end
end

fpath = mfilename('fullpath');
fpath=strrep(fpath,'\','/');
findex=strfind(fpath,'/');
rootDir=fpath(1:findex(end-1));
pre=strcat(rootDir,'Matlab/results/');% The folder to save figures
filename=strcat(pre,'AEE',opts.model,num2str(n),'.mat');
save(filename,'opts','nrange','t_AEE','t_ASE','RI_AEE','RI_ASE','RI');
