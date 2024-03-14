# Graph Encoder Embedding

-----------------------------------------------------------------

The **Main** folder contains the core GraphEncoder function in three languages:
- MATLAB:   GraphEncoder.m
- Python:   GraphEncoder.ipynb
- R:        GraphEncoder.R
plus several extension functions in MATLAB:
-- TemporalGraph.m
-- UnsupGraph.m

The **Data** folder contains the public real data used in the reference papers. 

The **Experiments** folder contains various experiments, plots, and auxiliary functions for the reference papers.

-------------------------------------------------------------
**Basic Usage in MATLAB:**

Given a graph A (either an n*n square matrix or an s*3 edgelist) and corresponding label vector Y (n*1 vector with K classes), the following outputs the supervised graph encoder embedding
> Z=GraphEncoder(A,Y);
where Z is the n*K vertex embedding.

Given a time-series graph A (stored in a 1*T cell, and each cell can be either square matrix of edgelist), and a label vector Y, the following outputs the temporal embedding:
> [Z,Dynamic]=TemporalGraph(E,Y);
where Dynamic contains the vertex, community, and graph dynamic in a 1*3 cell output.

Given a graph A and desired number of class K (or a range), the following outputs the unsupervised embedding:
> [Z,Y]=UnsupGraph(A,K);
where Z is the unsupervised vertex embedding, and Y is the estimated class label vector for each vertex.


-------------------------------------------------------------


References:

1. C. Shen and Q. Wang and C. E. Priebe, **"One-Hot Graph Encoder Embedding"**, IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(6):7933 - 7938, 2023. DOI: https://doi.org/10.1109/TPAMI.2022.3225073, arXiv:2109.13098

2. C. Shen and Y. Park and C. E. Priebe, **"Graph Encoder Ensemble for Simultaneous Vertex Embedding and Community Detection"**, in 2023 2nd International Conference on Algorithms, Data Mining, and Information Technology, pp. 13-18, ACM, 2023. DOI: https://doi.org/10.1145/3625403.3625407, arXiv:2301.11290

3. C. Shen, J. Larson, H. Trinh, X. Qin, Y. Park, and C. E. Priebe, **"Discovering Communication Pattern Shifts in Large-Scale Labeled Networks using Encoder Embedding and Vertex Dynamics"**, IEEE Transactions on Network Science and Engineering, 11(2):2100 - 2109, 2024. DOI: https://doi.org/10.1109/TNSE.2023.3337600, arXiv:2305.02381.

4. C. Shen, C. E. Priebe, J. Larson, H. Trinh, **"Synergistic Graph Fusion via Encoder Embedding"**, submitted, 2023. arXiv:2303.18051


