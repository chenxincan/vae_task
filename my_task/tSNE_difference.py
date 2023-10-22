import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 假设features是你从变分自编码器提取出来的SMILES特征，形状为(n_samples, n_features)
# features = encoder.predict(data)

# 使用t-SNE降维
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
tsne_results = tsne.fit_transform(features)

# 可视化结果
plt.figure(figsize=(8, 8))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=50, cmap='viridis')
plt.title('t-SNE visualization of SMILES features')
plt.show()



#这种可视化可以帮助你看到哪些SMILES在特征空间中是相近的，哪些是远离的。如果t-SNE图中的结构和你对于化合物的知识或其他外部信息相吻合，那么这可以作为特征提取有效性的一个证据。