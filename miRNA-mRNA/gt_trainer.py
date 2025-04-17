import tensorflow._api.v2.compat.v1 as tf
from gtransformer import GTransformer
import scipy.sparse as sp
import numpy as np
tf.disable_eager_execution()

class GTransformerTrainer:
    def __init__(self, args):
        self.args = args
        self.build_placeholders()

        # 构造模型，添加 L2 归一化
        self.model = GTransformer(
            hidden_dims=args.hidden_dims,
            num_heads=args.num_heads,
            lambda_=args.lambda_
        )
        self.loss, self.H, _ = self.model(self.A, self.X, self.R, self.S)
        self.H = tf.nn.l2_normalize(self.H, axis=1)  # 对嵌入进行 L2 归一化
        self.optimize(self.loss)
        self.build_session()

    def build_placeholders(self):
        # 稀疏矩阵 A 的占位符
        self.A_indices = tf.placeholder(tf.int64, shape=[None, 2])
        self.A_values = tf.placeholder(tf.float32, shape=[None])
        self.A_dense_shape = tf.placeholder(tf.int64, shape=[2])
        self.A = tf.SparseTensor(self.A_indices, self.A_values, self.A_dense_shape)

        # X 的占位符
        self.X = tf.placeholder(tf.float32, shape=[None, self.args.hidden_dims[0]])

        # S 和 R 的占位符
        self.S = tf.placeholder(tf.int64, shape=[None, 1])
        self.R = tf.placeholder(tf.int64, shape=[None, 1])

    def optimize(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.args.gradient_clipping)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

    def build_session(self, gpu=True):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if not gpu:
            config.intra_op_parallelism_threads = 0
            config.inter_op_parallelism_threads = 0
        self.session = tf.Session(config=config)
        self.session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    def __call__(self, A_triple, X, S, R):
        for epoch in range(self.args.n_epochs):
            # 构造 SparseTensorValue
            sparse_A_value = tf.SparseTensorValue(
                indices=A_triple[0],
                values=A_triple[1],
                dense_shape=A_triple[2]
            )
            feed_dict = {
                self.A_indices: sparse_A_value.indices,
                self.A_values: sparse_A_value.values,
                self.A_dense_shape: sparse_A_value.dense_shape,
                self.X: X,
                self.S: np.expand_dims(S, axis=1),
                self.R: np.expand_dims(R, axis=1)
            }
            # 运行训练操作并计算损失
            loss, _ = self.session.run([self.loss, self.train_op], feed_dict=feed_dict)

            # 打印损失值，检查 nan 或 0 的问题
            print(f"Epoch {epoch}, Loss: {loss}")
            if np.isnan(loss):
                print("Loss is NaN. Debug the inputs.")
                print("Features:", np.mean(X), np.std(X))
                print("S (labels):", np.mean(S), np.std(S))
                print("R (labels):", np.mean(R), np.std(R))
                break

    def infer(self, A_triple, X, S, R):
        sparse_A_value = tf.SparseTensorValue(
            indices=A_triple[0],
            values=A_triple[1],
            dense_shape=A_triple[2]
        )
        feed_dict = {
            self.A_indices: sparse_A_value.indices,
            self.A_values: sparse_A_value.values,
            self.A_dense_shape: sparse_A_value.dense_shape,
            self.X: X,
            self.S: np.expand_dims(S, axis=1),
            self.R: np.expand_dims(R, axis=1)
        }
        # 返回归一化的嵌入
        H_normalized = self.session.run(tf.nn.l2_normalize(self.H, axis=1), feed_dict=feed_dict)
        return H_normalized, None

