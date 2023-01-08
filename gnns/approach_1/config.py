model_type = 'gcn'                 #gcn/gin
learning_rate = 0.04
weight_decay = 0
epochs = 50
batch_size = 32
K = 70000
dims = [2048]
p = 0
training = False
depth = 5

g1 = 0
g2 = 0

GRAPH_PATH = '../../data/scene_graphs500_large.pkl'
TEST_GRAPH_PATH = ''#'../../data/scene_graphs500_large.pkl'
TEST_SYN_N_PATH = ''#'../../data/syn_n500_large.pkl'
SYN_N_PATH = '../../data/syn_n500_large.pkl'
SYN_E_PATH = '../../data/syn_e500_large.pkl'
EMBEDDING_PATH = '../../../drive/MyDrive/glove_emb_300.pkl'#embeddings_path2vec_wup.pkl'
GED_PATH = '../../outs/geds_bipartite_costs.pkl'
IDX_PATH = '../../data/scene500_large_idx.pkl'
LOAD_PATH = ''#'../../outs/gcn_70k_test.pth' #'../../outs/model_gcn_70k.pth'
EMB_SAVE_PATH = '../../outs/emb2.pkl'
SAVE_PATH = '../../outs/gcn_70k_2.pth'
SIM = ''

