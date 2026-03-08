
import numpy as np
from sklearn.decomposition import IncrementalPCA

class StreamingPCA:

    def __init__(self, n_components=50, batch_size=128):
        self.n_components = n_components
        self.batch_size = batch_size
        self.ipca = IncrementalPCA(n_components=n_components)

    def fit(self, features):
        n_samples = features.shape[0]
        for i in range(0, n_samples, self.batch_size):
            batch = features[i:i + self.batch_size]
            self.ipca.partial_fit(batch)

    def transform(self, features):
        n_samples = features.shape[0]
        transformed = []

        for i in range(0, n_samples, self.batch_size):
            batch = features[i:i + self.batch_size]
            transformed.append(self.ipca.transform(batch))

        return np.vstack(transformed)

    def fit_transform(self, features):
        self.fit(features)
        return self.transform(features)
