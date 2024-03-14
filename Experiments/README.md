This folder consists of simulated and real data experiments for the reference papers (see readme in the parent folder).

All experiments are carried out in MATLAB2022a, located under Experiments/Matlab. We used a Windows 10 PC with Intel i-7 6850K and 64GB memory.

As the GEE is continuously updated, certain experiment functions may be out-of-date. Please report in the issue if you would like to replicate an experiment, but the script does not run as intended.

-----------------------------------------------------------------------------------
**Paper 1**:

To replicate the experiments in paper 1, the scripts are located under /Experiments/Matlab:

1. GraphEncoderEvaluate.m: the 10-fold classification comparison script, outputs classification error and running time.
   Simulation example: 
> [Adj,Y]=simGenerate(20,2000); SBM=GraphEncoderEvaluate(Adj,Y);

   Real data example: 
   
> load('email.mat'); email=GraphEncoderEvaluate(Adj,Y);

   More classification examples can be found in simClassification.m.


2. GraphClusteringEvaluate.m: the clustering comparison, outputs ARI and running time.
   Simulation example: 
   
> [Adj,Y]=simGenerate(10,2000); SBM=GraphClusteringEvaluate(Adj,Y);

   Real data example: 
   
> load('email.mat'); email=GraphClusteringEvaluate(Adj,Y);

   More clustering examples can be found in simClassification.m.
   

3. simGenerate.m: generate simulated data. Options used in paper are type 10/11 (SBM), 20/21 (DC-SBM), 30/31 (RDPG).

   Example to generate adjacency matrix, label vector, and transform to edgelist: 
   
> [Adj,Y]=simGenerate(21,2000,5);

> Edge=adj2edge(Adj); 


4. generatePlotAEE.m: the function that plots all figures in the paper. 

> generatePlotAEE(i);

for i=-1,0,1,3,5 will generate figures in paper. 

Some data and figures are already generated within the /Matlab folder. Other scripts are mostly auxiliary functions and experimental trials.

If any of the experiment code is not working, this is typically caused by code update that may render certain scripts outdated. 
Please report in github issue and we will correct them asap.

-----------------------------------------------------------------------------------
**Paper 2**:

All experiments in paper 2 are in simImprove.m:

> simImprove(i) 

for i=0,1,2,3 will generate the figures in paper.

-----------------------------------------------------------------------------------
**Paper 3**:

All experiments and plots in paper 3 are in simDynamicPlot.m:

> simDynamicPlot(i) 

for i=1,2 will generate the simulation figures in paper.
Option 3,4,5 is for the Microsoft data, which is not publically available but can be available for individual upon request.

-----------------------------------------------------------------------------------
**Paper 4**:

All experiments and plots in paper 4 are in simFusion.m:

> simDynamicPlot(i) 

Setting i = 1,2,3,4 will generate the simulation results in paper. i=100 or 101 plots the simulation figures
i = 13 for the real data in paper.
