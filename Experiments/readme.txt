This folder consists of simulated and real data experiments for this paper: 
Cencheng Shen, Qizhe Wang, Carey E Priebe, "One-Hot Grah Encoder Embedding", submitted, https://arxiv.org/pdf/2109.13098.pdf

Most experiments are carried out in MATLAB2022a. The core GEE function is under Main/GraphEncoder.m

Given a edgelist E and corresponding label vector Y (all or partial), the following runs supervised GEE:
>> [Z,Y]=GraphEncoder(E,Y);
Given edgelist and desired number of class K, the following runs un-supervised GEE:
>> [Z,Y]=GraphEncoder(E,K);

----------------------------------------------------------------------------------------------------------

To replicate the experiments in the main paper, the scripts are located under /Experiments/Matlab:
1. GraphEncoderEvaluate.m: the 10-fold classification comparison script, outputs classification error and running time.
   Simulation example: 
>> [Adj,Y]=simGenerate(20,2000); SBM=GraphEncoderEvaluate(Adj,Y);
   Real data example: 
>> load('email.mat'); email=GraphEncoderEvaluate(Adj,Y);
   More classification examples can be found in simClassification.m.


2. GraphClusteringEvaluate.m: the clustering comparison, outputs ARI and running time.
   Simulation example: 
>> [Adj,Y]=simGenerate(10,2000); SBM=GraphClusteringEvaluate(Adj,Y);
   Real data example: 
>> load('email.mat'); email=GraphClusteringEvaluate(Adj,Y);
   More clustering examples can be found in simClassification.m.

3. simGenerate.m: generate simulated data. Options used in paper are type 10/11 (SBM), 20/21 (DC-SBM), 30/31 (RDPG).
   Example to generate adjacency matrix, label vector, and transform to edgelist: 
>> [Adj,Y]=simGenerate(21,2000,5);
>> Edge=adj2edge(Adj); 

4. generatePlotAEE.m: the function that plots all figures in the paper. 
   Type generatePlotAEE(i) for i=-1,0,1,3,5 will generate figures in paper. 

Some data and figures are already generated within the /Matlab folder. Other scripts are mostly auxiliary functions and experimental trials.
If any of the code is not working, this is typically caused by code update. Please report in github issue and we will correct them asap.