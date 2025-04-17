import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sp
import argparse
import tensorflow._api.v2.compat.v1 as tf
from gt_trainer import GTransformerTrainer


def sim_thresholding(matrix: np.ndarray, threshold):
    matrix_copy = matrix.copy()
    matrix_copy[matrix_copy >= threshold] = 1
    matrix_copy[matrix_copy < threshold] = 0
    print(f"rest links: {np.sum(np.sum(matrix_copy))}")
    return matrix_copy



def single_generate_graph_adj_and_feature(network, feature):
    features = sp.csr_matrix(feature).tolil().todense()
    
    # 使用新版本的 from_numpy_array
    graph = nx.from_numpy_array(network)
    
    # 生成邻接矩阵（保持稀疏格式）
    adj = nx.adjacency_matrix(graph)
    adj = sp.coo_matrix(adj)
    return adj, features



def get_gate_feature(adj, features, epochs, l):
    args = parse_args(epochs=epochs, l=l)
    args.hidden_dims = [features.shape[1]] + args.hidden_dims

    # 获取三元组
    adj_triple, S, R = prepare_graph_data(adj)

    # 传递三元组给训练器
    print("Shape of S:", S.shape)
    print("Shape of R:", R.shape)
    
    trainer = GTransformerTrainer(args)
    trainer(adj_triple, features, S, R)  # 使用 adj_triple
    embeddings, _ = trainer.infer(adj_triple, features, S, R)
    print("Features:", features.mean(), features.std())
    print("S (labels):", S.mean(), S.std())
    print("R (labels):", R.mean(), R.std())
    tf.reset_default_graph()
    return embeddings


def prepare_graph_data(adj):
    adj = adj.tocoo().astype(np.float32)

    # 按 (row, col) 排序
    sorted_idx = np.lexsort((adj.col, adj.row))  
    indices = np.vstack((adj.row[sorted_idx], adj.col[sorted_idx])).transpose().astype(np.int64)
    values = adj.data[sorted_idx]
    dense_shape = np.array(adj.shape, dtype=np.int64)

    return (indices, values, dense_shape), adj.row, adj.col



def parse_args(epochs, l):
    
    parser = argparse.ArgumentParser(description="Run gate.")

    parser = argparse.ArgumentParser(description="Run GTransformer.")
    parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads.')
    parser.add_argument('--hidden-dims', type=list, nargs='+', default=[256, 128])
    parser.add_argument('--lambda-', default=l, type=float)
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate. Default is 0.001.')

    parser.add_argument('--n-epochs', default=epochs, type=int,
                        help='Number of epochs')

    parser.add_argument('--dropout', default=0.5, type=float,
                        help='Dropout.')

    parser.add_argument('--gradient_clipping', default=5.0, type=float,
                        help='gradient clipping')

    return parser.parse_args()

if __name__ == '__main__':
    df_drug = pd.read_csv('../../../feature/miRNA_drug_network_feature_128.csv', index_col=0)
    df_func = pd.read_csv('../../../data/miRNA_func_sim.csv', header=None)

    feature = df_drug.values
    similarity = df_func.values
    #二值化
    network = sim_thresholding(similarity,0.8)
    adj, features = single_generate_graph_adj_and_feature(network, feature)
    print("Features shape:", features.shape)

    embeddings = get_gate_feature(adj, features,10, 1)
    print(embeddings.shape)

    # 指定要保存的CSV文件的路径
    file_path = '../../../feature/gate_feature_mRNA_0.8_128_0.012.csv'

    np.savetxt(file_path, embeddings, delimiter=',',)
