from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import tensorflow as tf
from tensorflow.keras.regularizers import l2

def create_transformer_fusion_model(seq_shape, drug_shape, mRNA_shape, dis_shape, num_classes):
    # 各特征输入
    seq_input = Input(shape=(seq_shape,), name='seq_feature')
    drug_input = Input(shape=(drug_shape,), name='drug_feature')
    mRNA_input = Input(shape=(mRNA_shape,), name='mRNA_feature')
    dis_input = Input(shape=(dis_shape,), name='dis_feature')

    # 特征对齐（加上L2正则化）
    seq_dense = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(seq_input)
    drug_dense = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(drug_input)
    mRNA_dense = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(mRNA_input)
    dis_dense = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(dis_input)

    # 拼接特征
    concat_features = Concatenate()([seq_dense, drug_dense, mRNA_dense, dis_dense])

    # Transformer 编码器
    def transformer_encoder(inputs, num_heads=8, ff_dim=256, dropout_rate=0.3):
        # 自注意力
        attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(inputs, inputs)
        attention_output = Dropout(dropout_rate)(attention_output)
        attention_output = LayerNormalization(epsilon=1e-6)(attention_output + inputs)

        # 前馈网络
        ff_output = Dense(ff_dim, activation='relu')(attention_output)
        ff_output = Dense(inputs.shape[-1], activation='linear')(ff_output)
        ff_output = Dropout(dropout_rate)(ff_output)
        ff_output = LayerNormalization(epsilon=1e-6)(ff_output + attention_output)

        return ff_output

    # 为 Transformer 添加时间维度
    concat_features_expanded = tf.expand_dims(concat_features, axis=1)
    transformer_output = transformer_encoder(concat_features_expanded)
    transformer_output = tf.squeeze(transformer_output, axis=1)  # 去掉时间维度

    # 自监督模块（对比学习）
    projection_head = Dense(64, activation='relu')(transformer_output)
    projection_head = Dense(32, activation='linear', name="projection_head")(projection_head)

    # 分类头
    classification_head = Dense(64, activation='relu')(transformer_output)
    classification_head = Dropout(0.3)(classification_head)
    output = Dense(num_classes, activation='sigmoid', name='classification_head')(classification_head)

    # 模型
    model = Model(inputs=[seq_input, drug_input, mRNA_input, dis_input],
                  outputs=[output, projection_head])

    return model


def lr_schedule(epoch):
    """学习率衰减"""
    if epoch < 10:
        return 0.001
    else:
        return 0.0005

if __name__ == '__main__':
    # 读取数据
    df_seq_feature = pd.read_csv('../feature/miRNA_seq_feature_64.csv', index_col='Index')
    df_mRNA_net_feature = pd.read_csv('../feature/gt_feature_mRNA_0.8_128_0.01.csv', header=None)
    df_drug_feature = pd.read_csv('../feature/gt_feature_drug_0.8_128_0.01.csv', header=None)
    df_dis_feature = pd.read_csv('../feature/gt_feature_disease_0.8_128_0.01.csv', header=None)
    df_loc = pd.read_csv('../data/miRNA_localization.csv', header=None)
    df_loc_index = pd.read_csv('../data/miRNA_have_loc_information_index.txt', header=None)

    loc_index = df_loc_index[0].tolist()
    select_row = np.array([value == 1 for value in loc_index])

    # 提取特征
    seq_feature = df_seq_feature.values
    drug_feature = df_drug_feature.values
    dis_feature = df_dis_feature.values
    miRNA_loc = df_loc.values

    # 合并特征
    merge_feature = np.concatenate((seq_feature, drug_feature, mRNA_net_feature, dis_feature), axis=1)

    # 数据归一化
    n_splits = 10  # K折交叉验证
    scaler = StandardScaler()
    merge_feature_scaled = scaler.fit_transform(merge_feature)

    # 多标签数据
    miRNA_loc_multilabel = miRNA_loc[select_row]
    merge_feature_scaled_multilabel = merge_feature_scaled[select_row]

    num_classes = 7
    random_seed = 42
    auc_ls = [0] * 7
    aupr_ls = [0] * 7
    np.random.seed(random_seed)
    class_name = ['Cytoplasm', 'Exosome', 'Nucleolus', 'Nucleus', 'Extracellular vesicle', 'Microvesicle',
                  'Mitochondrion']
    fold_size = len(merge_feature_scaled_multilabel) // n_splits

    with open("T+C_optimized.txt", "w") as f:
        for i in range(n_splits):
            # 数据切分
            X, y = shuffle(merge_feature_scaled_multilabel, miRNA_loc_multilabel, random_state=random_seed)

            # 定义测试集和训练集索引
            test_start = i * fold_size
            test_end = (i + 1) * fold_size
            test_indices = range(test_start, test_end)
            train_indices = [j for j in range(len(X)) if j not in test_indices]

            # 分割数据
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            # 特征分拆
            seq_shape = seq_feature.shape[1]
            drug_shape = drug_feature.shape[1]
            mRNA_shape = mRNA_net_feature.shape[1]
            dis_shape = dis_feature.shape[1]

            X_train_seq, X_train_drug, X_train_mRNA, X_train_dis = \
                X_train[:, :seq_shape], \
                    X_train[:, seq_shape:seq_shape + drug_shape], \
                    X_train[:, seq_shape + drug_shape:seq_shape + drug_shape + mRNA_shape], \
                    X_train[:, seq_shape + drug_shape + mRNA_shape:seq_shape + drug_shape + mRNA_shape + dis_shape]

            X_test_seq, X_test_drug, X_test_mRNA, X_test_dis = \
                X_test[:, :seq_shape], \
                    X_test[:, seq_shape:seq_shape + drug_shape], \
                    X_test[:, seq_shape + drug_shape:seq_shape + drug_shape + mRNA_shape], \
                    X_test[:, seq_shape + drug_shape + mRNA_shape:seq_shape + drug_shape + mRNA_shape + dis_shape]

            # 创建模型
            model = create_transformer_fusion_model(seq_shape, drug_shape, mRNA_shape, dis_shape, num_classes)

            # 编译模型
            model.compile(
                optimizer='adam',
                loss={
                    'classification_head': 'binary_crossentropy',
                    'projection_head': 'mse',
                },
                metrics={'classification_head': ['accuracy']}
            )

            # 学习率衰减回调
            lr_scheduler = LearningRateScheduler(lr_schedule)

            # 训练模型
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            model.fit(
                [X_train_seq, X_train_drug, X_train_mRNA, X_train_dis],
                {'classification_head': y_train, 'projection_head': np.zeros((y_train.shape[0], 32))},
                validation_split=0.2,
                epochs=20,
                batch_size=32,
                callbacks=[early_stopping, lr_scheduler]
            )

            # 预测
            y_pred = model.predict([X_test_seq, X_test_drug, X_test_mRNA, X_test_dis])[0]

            # 评估指标
            for class_idx in range(num_classes):
                true_label = y_test[:, class_idx]
                pre_label = y_pred[:, class_idx]
                accuracy = accuracy_score(true_label, (pre_label > 0.5).astype(int))
                precision = precision_score(true_label, (pre_label > 0.5).astype(int))
                recall = recall_score(true_label, (pre_label > 0.5).astype(int))
                f1 = f1_score(true_label, (pre_label > 0.5).astype(int))
                auc_score = roc_auc_score(true_label, pre_label)
                aupr_score = average_precision_score(true_label, pre_label)
                auc_ls[class_idx] += auc_score
                aupr_ls[class_idx] += aupr_score
                f.write(
                    f"Class {class_idx} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc_score:.4f}, AUPR: {aupr_score:.4f}\n")

        avg_auc = 0
        avg_aupr = 0
        for class_idx in range(num_classes):
            f.write(
                f"Class {class_name[class_idx]} - AUC: {auc_ls[class_idx] / n_splits}, AUPR: {aupr_ls[class_idx] / n_splits}\n")
            avg_auc += auc_ls[class_idx] / n_splits
            avg_aupr += aupr_ls[class_idx] / n_splits

        avg_auc /= num_classes
        avg_aupr /= num_classes
        f.write(f"Average AUC: {avg_auc:.4f}, Average AUPR: {avg_aupr:.4f}")
