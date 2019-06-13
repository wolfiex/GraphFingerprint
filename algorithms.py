from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap,time
import numpy as np


def do_tsne(data_subset,learning_rate=200,early_exaggeration=12.0,perplexity=30,n_iter=300):
		np.random.seed(seed=42)
		time_start = time.time()
		tsne = TSNE(n_components=2, verbose=1,perplexity=perplexity,early_exaggeration=early_exaggeration,n_iter = n_iter)
		tsne_results = tsne.fit_transform(data_subset)
		tm = time.time()-time_start
		print('t-SNE done! Time elapsed: {} seconds'.format(tm))
		return tsne_results[:,0],tsne_results[:,1],tm


def do_pca(data_subset):
		np.random.seed(seed=42)
		time_start = time.time()
		pca = PCA(n_components=2)
		pca_results = pca.fit_transform(data_subset)
		tm = time.time()-time_start
		print('PCA done! Time elapsed: {} seconds'.format(tm))
		return pca_results[:,0],pca_results[:,1],tm


def do_umap(data_subset, n_neighbors=5,min_dist=0.3,metric='correlation'):
		np.random.seed(seed=42)
		time_start = time.time()
		reducer = umap.UMAP(n_neighbors=n_neighbors,min_dist=min_dist,metric=metric)
		embedding = reducer.fit_transform(data_subset)
		tm = time.time()-time_start
		print('PCA done! Time elapsed: {} seconds'.format(tm))
		return embedding[:,0],embedding[:,1],tm
