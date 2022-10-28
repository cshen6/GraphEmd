# This function loads an edgelist graph, and for each vertex, finds its out-degree (nr. of outgoing edges)
# 	and adds that number as a weighted self-edge to the edgelist in the form (v1, v2, outdeg_v1)


import numpy as np


def count_edges_for_laplacian():
	# This function creates the Degree vector/diag matrix by counting edges in which vertex v occurs

	print("Loading Graph")
	G_edgelist = np.loadtxt("../../../../Thesis-Graph-Data/twitch-SNAP-weighted.csv", delimiter=" ", dtype=np.int32)

	# First column of the edgelist = list of source vertices (not unique)
	all_source_vertices = G_edgelist[:, 0]

	print("Counting edges by source vertex")

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

def sum_weights_for_laplacian():
	# This function creates the Degree vector/diag matrix by summing weights of edges in which vertex v occurs
	# Trying to imitate GEE, but failed, RESULTS DON'T MATCH GEE

	print("Loading Graph")
	G_edgelist = np.loadtxt("../../../../Thesis-Graph-Data/twitch-SNAP-weighted.csv", delimiter=" ", dtype=np.int32)

	# TODO Sort array by only 1st column. See https://numpy.org/doc/stable/user/basics.rec.html

	print("Counting weights of all edges by vertex")

	vertex_outdegrees = []

	# O(n) count sorted occurrences
	curr_vertex = G_edgelist[0, 0] # First source vertex
	curr_vertex_weight = G_edgelist[0, 2]  # First edge's weight

	# Loop over edges
	for i in range(1, G_edgelist.shape[0]):
		if curr_vertex == G_edgelist[i][0]:
			curr_vertex_weight += G_edgelist[i][2]
		else:
			vertex_outdegrees.append((curr_vertex, curr_vertex, curr_vertex_weight))  # Save in edgelist format
			curr_vertex_weight = 1  # Again, count itself
			curr_vertex = G_edgelist[i][0]

	laplacian_edgelist = np.vstack((G_edgelist, vertex_outdegrees))

	np.savetxt("laplacian_edgelist_weighted.csv", laplacian_edgelist, fmt='%d')



if __name__ == '__main__':
    sum_weights_for_laplacian()
