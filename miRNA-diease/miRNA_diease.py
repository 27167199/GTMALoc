import pandas as pd
import networkx as nx
from node2vec import Node2Vec

if __name__ == '__main__':
    df_miRNA_dis = pd.read_csv('../../../data/miRNA_disease.csv', header=None)
    miRNA_dis_matrix = df_miRNA_dis.values

    G = nx.Graph()

    for i in range(1041):
        G.add_node(f"miRNA_{i}")

    for i in range(640):
        G.add_node(f"disease_{i}")

    for miRNA_index in range(1041):
        for disease_index in range(640):
            if miRNA_dis_matrix[miRNA_index, disease_index] != 0:
                G.add_edge(f"miRNA_{miRNA_index}", f"disease_{disease_index}")

    node2vec = Node2Vec(G, dimensions=128, walk_length=150, num_walks=200, workers=1)
    model = node2vec.fit()

    miRNA_features = {f"miRNA_{i}": model.wv[f"miRNA_{i}"] for i in range(1041)}

    df = pd.DataFrame.from_dict(miRNA_features, orient='index')
    df.to_csv('../../../feature/miRNA_disease_feature_128.csv')
