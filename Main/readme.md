# Main
This folder contains the main code for GEE in three languages

- MATLAB:   GraphEncoder.m
- Python:   GraphEncoder.ipynb
- R:        GraphEncoder.R

and additional experimental functions on MATLAB:
-- TemporalGraph.m

On a standard PC, the MATLAB code is typically the fastest, followed by Python then R.

-------------------------------------------------------------
**Basic Usage in MATLAB:**

Given a graph A (either an n*n square matrix or an s*3 edgelist) and corresponding label vector Y, the following runs the supervised GEE based on paper 1:
> Z=GraphEncoder(A,Y);
where Z is the vertex embedding.

Given a graph A and desired number of class K (or a range), the following runs the GEE clustering based on paper 2:
> [Z,Y]=GraphEncoder(A,K);
where Z is the vertex embedding and Y is the estimated class for each vertex.

Given a time-series graph A (stored in a 1*T cell, and each cell can be either square matrix of edgelist), and a label vector Y, the following runs the temporal embedding based on paper 3:
> [Z,Dynamic]=TemporalGraph(E,Y);
where Dynamic contains the vertex, community, and graph dynamic in a 1*3 cell output.

-------------------------------------------------------------
