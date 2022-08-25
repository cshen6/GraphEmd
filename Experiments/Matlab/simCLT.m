function []=simCLT(opt)

% Visualization
fpath = mfilename('fullpath');
fpath=strrep(fpath,'\','/');
findex=strfind(fpath,'/');
rootDir=fpath(1:findex(end));
pre=strcat(rootDir,'');% The folder to save figures
fs=30;
lw=3;
rep=100;
opts0 = struct('DiagA',true,'Normalize',false,'Laplacian',false,'Replicates',1);
opts1 = struct('DiagA',true,'Normalize',true,'Laplacian',false,'Replicates',1);
opts2 = struct('DiagA',true,'Normalize',true,'Laplacian',false,'Replicates',10);
% map2 = brewermap(128,'PiYG'); % brewmap
% colormap(gca,map2);

if opt==0
end

if opt==1
end

if opt==2
end