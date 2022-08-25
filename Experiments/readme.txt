Each folder consists of simulated and real data experiments using GraphEncoder for this paper: 
Cencheng Shen, Qizhe Wang, Carey E Priebe, "One-Hot Grah Encoder Embedding", submitted, https://arxiv.org/pdf/2109.13098.pdf

Most experiments are carried out in /Matlab. Here we provide descriptions for some core scripts:
1. GraphEncoderEvaluate.m: the 10-fold classification comparison script, outputs classification error and running time.
   Simulation example: [Adj,Y]=simGenerate(10,2000,k); SBM=GraphEncoderEvaluate(Adj,Y);
   Real data example: load('email.mat'); email=GraphEncoderEvaluate(Adj,Y);

2. GraphClusteringEvaluate.m: the clustering comparison, outputs ARI and running time.
   Simulation example: [Adj,Y]=simGenerate(10,2000,k); SBM=GraphClusteringEvaluate(Adj,Y);
   Real data example: load('email.mat'); email=GraphClusteringEvaluate(Adj,Y);

3. simGenerate.m: generate simulated data. Options used in paper are type 10, 11, 20, 21, 30, 31.

4. generatePlotAEE.m: the function that plots all figures in the paper. 
   Type generatePlotAEE(i) for i=-1,0,1,3,5 will generate figures in paper. 

Some data and figures are already generated within the /Matlab folder. 
Other scripts are mostly auxiliary functions and experimental trials.
Please open an issue if any of the code is obsolete and does not run.