# Main
This folder contains the main code for GEE in three languages

1. MATLAB:   GraphEncoder.m
             GraphDynamics.m
2. Python:   GraphEncoder.ipynb
3. R:        GraphEncoder.R

On a standard PC, the MATLAB code is typically the fastest, followed by Python then R.

-------------------------------------------------------------
**Basic Usage in MATLAB:**

Given an edgelist E and corresponding label vector Y (all or partial), the following runs the supervised GEE based on paper 1:
> Z=GraphEncoder(E,Y);
where Z is the vertex embedding.

Given the edgelist and desired number of class K (or a range), the following runs the GEE ensemble based on paper 2:
> [Z,Y]=GraphEncoder(E,K);
where Z is the vertex embedding and Y is the estimated class for each vertex.

Given a time-series graph, say a 1*T cell of edgelists E and a label vector Y, the following runs the Temporal embedding based on paper 3:
> [Z,Dynamic]=GraphDynamic(E,Y);
where Dynamic contains the vertex, community, and graph dynamic in a 1*3 cell output.

Other optional parameters include using laplacian transform or not, normalize to unit circle or not, number of random replicates, etc.

-------------------------------------------------------------

Reference Papers:

1. C. Shen and Q. Wang and C. E. Priebe, **"One-Hot Graph Encoder Embedding"**, IEEE Transactions on Pattern Analysis and Machine Intelligence, accepted, 2023. arXiv:2109.13098

2. C. Shen and Y. Park and C. E. Priebe, **"Graph Encoder Ensemble for Simultaneous Vertex Embedding and Community Detection"**, submitted, 2023. arXiv:2301.11290

3. C. Shen, J. Larson, H. Trinh, X. Qin, Y. Park, and C. E. Priebe, **"Discovering Communication Pattern Shifts in Large-Scale Networks using Encoder Embedding and Vertex Dynamics
"**, submitted, 2023. 

4. C. Shen, C. E. Priebe, J. Larson, H. Trinh, **"Synergistic Graph Fusion via Encoder Embedding"**, submitted, 2023. 

