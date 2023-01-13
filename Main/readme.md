# Main
This folder contains the main code for GEE in three languages

1. MATLAB:   GraphEncoder.m
2. Python:   GraphEncoder.ipynb
3. R:        GraphEncoder.R


--------------Basic Usage in MATLAB--------------------------

Given an edgelist E and corresponding label vector Y (all or partial), the following runs the supervised GEE based on paper 1:
>> Z=GraphEncoder(E,Y);
where Z is the vertex embedding.

Given the edgelist and desired number of class K (or a range), the following runs the GEE ensemble based on paper 2:
>> [Z,Y]=GraphEncoder(E,K);
where Z is the vertex embedding and Y is the estimated class for each vertex.

Other optional parameters include using laplacian transform or not, normalize to unit circle or not, number of random replicates, etc.

On a standard PC, the MATLAB code is typically the fastest, followed by Python then R.

-------------------------------------------------------------

Reference Papers:

**1. C. Shen and Q. Wang and C. E. Priebe, "One-Hot Graph Encoder Embedding", IEEE Transactions on Pattern Analysis and Machine Intelligence, accepted, 2023. arXiv:2109.13098**

**2. C. Shen and Y. Park and C. E. Priebe, "Graph Encoder Ensemble for Simultaneous Vertex Embedding and Community Detection", submitted, 2023. **
