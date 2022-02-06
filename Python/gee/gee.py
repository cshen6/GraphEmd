import numpy as np

class GraphEncoderEmbedding:
	"""
	Implementation of the graph encoder embedding proposed by Shen at al.
	"""
	def __init__(self, lap=False, n_nodes=None):
		self.lap=lap
		self.n_nodes = n_nodes


	def fit(self, X, y=None):
		"""
		Input
		X - An edge list of shape (n_edges, {2,3}) or an adjacency matrix of shape (n_nodes, n_nodes).
		y - Labels for each node in implied by X
		"""

		self.classes = np.unique(y)

		a, b = X.shape

		if b == 2:
			X = np.vstack((X, np.ones(a)))
			b = 3

		if b == 3:
			if self.n_nodes is None:
				self.n_nodes = len(np.unique(np.concatenate((X[:, 0], X[:, 1]))))
			edge_list = True

		elif a == b:
			self.n_nodes = a
			edge_list = False

		self.W = np.zeros((self.n_nodes, len(self.classes)))
		self.encoder_embedding = np.zeros((len(self.classes), self.n_nodes))

		for i, k in enumerate(self.classes):
			inds = np.where(y == k)[0]
			self.W[inds, i] = 1 / len(inds)

		if edge_list:
			self._fit_matrix(X, y)
		else:
			self._fit_matrx(X, y)

		self.pinv = np.linalg.pinv(self.encoder_embedding)

		return self


	def _fit_matrix(self, X, y=None):
		for i in range(X.shape[0]):
			y1 = y[i]
			y1_index = int(np.where(self.classes == y1)[0][0])

			for j in range(X.shape[1]):
				y2 = y[j]
				edge_weight = X[i,j]

				y2_index = int(np.where(self.classes == y2)[0][0])

				self.encoder_embedding[i, y2_index] += edge_weight * self.W[j, y2_index]
				self.encoder_embedding[j, y1_index] += edge_weight * self.W[i, y1_index]


	def _fit_edge_list(self, X, y=None):
		for i, edge in enumerate(X):
			node1 = int(edge[0])
			node2 = int(edge[1])

			y1 = y[node1]
			y2 = y[node2]
			edge_weight = edge[2]

			y1_index = int(np.where(self.classes == y1)[0][0])
			y2_index = int(np.where(self.classes == y2)[0][0])

			self.encoder_embedding[node1, y2_index] += edge_weight * self.W[node2, y2_index]
			self.encoder_embedding[node2, y1_index] += edge_weight * self.W[node1, y1_index]


	def transform(self, X):
		"""
		returns OOS embedding of X. X must be shape (n_of_sample_nodes, n_in_sample_nodes)
		"""

		return X @ self._pinv_left

	def fit_transform(self, X, y=None):
		self.fit(X, y)
		return self.transform(X)