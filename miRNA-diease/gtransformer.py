# gtransformer.py
import tensorflow._api.v2.compat.v1 as tf
from tensorflow.keras import layers
import numpy as np


class GTransformer:
    def __init__(self, hidden_dims, num_heads=4, lambda_=1.0):
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        self.lambda_ = lambda_
        self.n_layers = len(hidden_dims) - 1
        self.weights = self._define_weights()

    def _define_weights(self):
        weights = {}
        for i in range(self.n_layers):
            # Multi-head attention kernel（权重矩阵）
            weights[f"attn_kernel_{i}"] = tf.get_variable(
                f"attn_kernel_{i}",
                shape=(self.hidden_dims[i], self.hidden_dims[i + 1] * self.num_heads)  # 输出维度需匹配多头
            )
            # 修改偏置形状为 (num_heads * hidden_dim)
            weights[f"attn_bias_{i}"] = tf.get_variable(
                f"attn_bias_{i}",
                shape=(self.hidden_dims[i + 1] * self.num_heads,)  # 形状 (num_heads * hidden_dim,)
            )
            # Feed-forward weights
            weights[f"ffn_kernel_{i}"] = tf.get_variable(
                f"ffn_kernel_{i}",
                shape=(self.hidden_dims[i + 1] * self.num_heads, self.hidden_dims[i + 1])  # 输入维度匹配多头输出
            )
        return weights

    def __call__(self, A, X, R, S):
        H = X
        for layer in range(self.n_layers):
            H = self._graph_transformer_layer(H, A, layer)

        # Final embeddings
        self.H = H

        # Structure reconstruction loss (optional)
        self.S_emb = tf.nn.embedding_lookup(self.H, S)
        self.R_emb = tf.nn.embedding_lookup(self.H, R)
        structure_loss = -tf.reduce_mean(tf.log(tf.sigmoid(tf.reduce_sum(self.S_emb * self.R_emb, axis=-1))))

        # Total loss (if needed)
        self.loss = structure_loss * self.lambda_

        return self.loss, self.H, None  # 注意力系数可忽略或自定义

    def _graph_transformer_layer(self, H, A, layer):
        attn_output = self._multi_head_attention(H, A, layer)  # (N, num_heads * out_dim)
        ffn_output = tf.matmul(attn_output, self.weights[f"ffn_kernel_{layer}"])  # (N, out_dim)
        return tf.nn.relu(ffn_output)

    def _multi_head_attention(self, H, A, layer):

        # 节点数
        N = tf.shape(H)[0]
        out_dim = self.hidden_dims[layer + 1]
        num_heads = self.num_heads


        query = tf.matmul(H, self.weights[f"attn_kernel_{layer}"])

        query = tf.reshape(query, [N, num_heads, out_dim])

        query = tf.transpose(query, [1, 0, 2])

        outputs = []

        A_dense = tf.sparse.to_dense(A, default_value=0.0)
        mask = tf.cast(tf.not_equal(A_dense, 0.0), dtype=tf.float32)

        for head in range(num_heads):
            q_head = query[head]

            attn_logits = tf.matmul(q_head, q_head, transpose_b=True)  # (N, N)

            attn_logits = attn_logits * mask + (1.0 - mask) * (-1e9)

            attn_logits = (attn_logits + tf.transpose(attn_logits)) / 2.0

            attn = tf.nn.softmax(attn_logits, axis=-1)  # (N, N)

            attn_output = tf.matmul(attn, q_head)  # (N, out_dim)
            outputs.append(attn_output)

        output = tf.concat(outputs, axis=-1)

        output += self.weights[f"attn_bias_{layer}"]

        output = tf.nn.relu(output)
        return output



