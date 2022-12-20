import networkx as nx


def edgelist2adjmatrix(path):
    G = nx.read_edgelist(path)
    return G
