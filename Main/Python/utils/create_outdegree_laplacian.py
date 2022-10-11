# This function loads an edgelist graph, and for each vertex, finds its out-degree (nr. of outgoing edges)
# 	and adds that number as a weighted self-edge to the edgelist in the form (v1, v2, outdeg_v1)


import numpy as np

G_edgelist = G_edgelist = np.loadtxt("../../../../Thesis-Graph-Data/twitch-SNAP-weighted.csv", delimiter=" ", dtype=np.int32)

# First column of the edgelist = list of source vertices (not unique)
all_source_vertices = G_edgelist[:, 0]


# Don't use this, it's O(n^2)
# out_degree = [np.sum(G_edgelist[:,0] == v) for v in vertices]


# O(nlogn) + O(n) solution-------------------------------------

# O(nlogn) Sort
all_source_vertices = np.sort(all_source_vertices)

vertex_outdegrees = []

# O(n) count sorted occurrences
curr_vertex = all_source_vertices[0]
curr_vertex_count = 1 # Count itself
for i in range(1, len(all_source_vertices)):
	if curr_vertex == all_source_vertices[i]:
		curr_vertex_count += 1
	else:
		vertex_outdegrees.append((curr_vertex, curr_vertex, curr_vertex_count)) # Save in edgelist format
		curr_vertex_count = 1 # Again, count itself
		curr_vertex = all_source_vertices[i]


laplacian_edgelist = np.vstack((G_edgelist, vertex_outdegrees))

np.savetxt("laplacian_edgelist.csv", laplacian_edgelist, fmt='%d')
