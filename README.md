GTMALoc: miRNA Subcellular Localization Prediction via Graph Transformer and Multi-Head Attention
Environment Requirements
To ensure reproducibility, the following dependencies are required:

Python: 3.7.16

TensorFlow: 2.11.0

PyTorch: 1.12.0

scikit-learn: 1.0.2

node2vec: 0.4.3

networkx: 2.6.3

pandas: 1.5.2

scipy: 1.10.0

Directory Structure and Data Description
1. Data
Basic Information

miRNA_ID_1041.txt: Unique identifiers for 1041 miRNAs used in GTMALoc.

miRNA_have_loc_information_index.txt: Index list of miRNAs with known subcellular localization annotations.

Association Data

miRNA_disease.csv: Binary association matrix between 1041 miRNAs and 640 diseases.

miRNA_drug.csv: Binary association matrix between 1041 miRNAs and 130 drugs.

miRNA_mRNA_matrix.txt: Binary interaction matrix between 1041 miRNAs and 2836 mRNAs.

Similarity Data

miRNA_seq_sim.csv: Sequence similarity matrix for the 1041 miRNAs.

miRNA_func_sim.csv: Functional similarity matrix for the same miRNA set.

Subcellular Localization Annotations

miRNA_localization.csv: Known subcellular localization labels for the 1041 miRNAs.

2. Feature Representations (Extracted via Node2Vec)
miRNA_seq_feature_64.csv: 64-dimensional sequence-based features.

miRNA_disease_feature_128.csv: 128-dimensional features from the miRNA-disease network.

miRNA_drug_feature_128.csv: 128-dimensional features from the miRNA-drug interaction network.

miRNA_mRNA_network_feature_128.csv: 128-dimensional features from the miRNA-mRNA interaction network.

Usage Instructions
Feature-Specific Preprocessing
Run the following scripts individually to preprocess and encode each network:
python miRNA_diease.py
python miRNA_drug.py
python miRNA_mRNA.py
Main Model Training and Evaluation
After generating the intermediate features, execute:
python main.py


Contact Information
For questions, issues, or collaboration inquiries, please contact:
Cheng Yan – yancheng01@hnucm.edu.cn