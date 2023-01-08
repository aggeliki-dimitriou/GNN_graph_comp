import pickle as pkl
import scipy.sparse as sp
import networkx as nx
import numpy as np
import itertools
import random
import torch
from torch_geometric.utils.convert import from_networkx
from models import GNN_model
from tqdm import tqdm, trange
from torch_geometric.loader import DataLoader
from nltk.corpus import wordnet as wn


def find_embedding_glove(names, embedding_map, depth):
    # max_retry = 5
    # hypers = []

    embs = []
    for i in names:
        name = i.split('.')[0]
        try:
            # print(name)
            embs.append(embedding_map[name])
            # print(type(embedding_map[name]))
        except:
            components = name.split('_')
            # hypers.append([c.name() for c in wn.synset(i).hypernyms()])
            if len(components)>1:
              emb = []
              for j in components:
                  try:
                      emb.append(embedding_map[j])
                  except:
                      continue
              emb = np.mean(np.array(emb), axis=0)
              embs.append(emb)  
            else:    
              continue   

    if len(embs) < depth+1:
      mean = np.mean(np.array(embs), axis=0)
      pad = [embs.append(mean) for i in range(depth+1-len(embs))]
    return embs

def find_hierarchy(k, c):
    c_set = set()

    prev = set([c])
    c_set.add(c)

    for _ in range(k):
        hyps = set()
        for j in prev:
            hyp = set(j.hypernyms())
            hyps = hyps.union(hyp)

        c_set = c_set.union(hyps)

        prev = hyps

    backups = [wn.synsets(i.name()) for i in c_set]
    b_s = set()
    for l in backups:
      for j in l:
        b_s = b_s.union(j)

    res = c_set.union(b_s)

    return res


class Trainer:
    def __init__(self, args):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_type = args.model_type
        self.args = args
        self.load_data()
        self.compute_targets()
        self.setup_model()
        print(sum(param.numel() for param in self.model.parameters()))


    def setup_model(self):
        print('Setting up Model...')
        self.model = GNN_model(self.args.dims, self.num_features, self.args.p, self.args.training, self.model_type, self.device, self.args.depth).to(self.device)

    def load_data(self):
        print('Loading Data...')

        self.graphs = pkl.load(open(self.args.GRAPH_PATH, "rb"))

        if self.args.TEST_GRAPH_PATH:
            self.test_graphs = pkl.load(open(self.args.TEST_GRAPH_PATH, "rb"))
            syn_n_test = pkl.load(open(self.args.TEST_SYN_N_PATH, "rb"))

        # load and assign attributes
        syn_n = pkl.load(open(self.args.SYN_N_PATH, "rb"))
        # syn_e = pkl.load(open(SYN_E_PATH, "rb"))
        embeddings = pkl.load(open(self.args.EMBEDDING_PATH, "rb"))

        self.num_features = len(list(embeddings.values())[0])

        # self.graph_attributes = []
        self.attach_features(embeddings, syn_n, self.graphs)

        if self.args.TEST_GRAPH_PATH:
            self.attach_features(embeddings, syn_n_test, self.test_graphs)
            # print(self.test_graphs[0].nodes.data())

        # 124750 possible pairs pick 2000 pairs
        pairs = list(itertools.combinations(list(range(500)), 2))
        self.train_graph_pair_idx = random.sample(pairs, k=self.args.K)

        # print(self.train_graph_pair_idx[:5])

    def attach_features(self, embeddings, syn_n, gs):
        for g_idx, G in tqdm(enumerate(gs)):
            for i in list(G.nodes()):
                names = syn_n[g_idx][i]
                # if len(names) != 1:
                #     print("Multiple Synsets", names)
                # names_c = set()
                names_c = list(find_hierarchy(self.args.depth, wn.synset(names[0])))[:self.args.depth+1]
                names_c = list([i.name() for i in names_c])
                if 'glove' in self.args.EMBEDDING_PATH:
                    embs = find_embedding_glove(list(names_c), embeddings, self.args.depth)
                    assert len(embs) == 6
                else:
                    embs = []
                    for n in names:
                        emb = embeddings[n]
                        embs.append(emb)
                    embs = embs
                
                embs = np.array(embs)
                del G.nodes()[i]['label']  ##
                G.nodes()[i]['feature'] = embs
            for i in list(G.edges()):  ##
                del G.edges()[i]['label']  ##
        return gs

    def compute_targets(self):
        geds = pkl.load(open(self.args.GED_PATH, 'rb'))
        self.real_idx = pkl.load(open(self.args.IDX_PATH, 'rb'))
        self.targets = dict()
        for pair in self.train_graph_pair_idx:
            new_pair = (self.real_idx[pair[0]][0], self.real_idx[pair[1]][0])
            self.targets[pair] = geds[new_pair][0]

    def format_graph_pairs(self):
        # convert to PyG format

        pairs_1 = []
        pairs_2 = []

        for idx in self.train_graph_pair_idx:
            # new_data = dict()
            G1 = self.graphs[idx[0]]
            G2 = self.graphs[idx[1]]

            pairs_1.append(from_networkx(G1).to(self.device))
            pairs_2.append(from_networkx(G2).to(self.device))

        train_loader_1 = DataLoader(pairs_1, batch_size=self.args.batch_size)
        train_loader_2 = DataLoader(pairs_2, batch_size=self.args.batch_size)
        target_idxs = [self.train_graph_pair_idx[i * self.args.batch_size:(i + 1) * self.args.batch_size] for i, _ in
                       enumerate(train_loader_1)]
        targets = []
        for idx_list in target_idxs:
            targets.append(torch.FloatTensor([self.targets[i] for i in idx_list]).to(self.device))
        return (train_loader_1, train_loader_2, targets)

    def format_graph_single(self, gs):
        graphs = []
        for idx in range(len(gs)):
            # new_data = dict()
            G = gs[idx]

            new_G = from_networkx(G).to(self.device)

            graphs.append(new_G)

        graph_loader = DataLoader(graphs, batch_size=self.args.batch_size)

        return graph_loader

    def process_batch(self, batch1, batch2, targets):
        self.optimizer.zero_grad()
        # losses = 0

        # pairs = self.format_graph_pairs(batch)

        train_pred = self.model(batch1, batch2)

        losses = torch.nn.functional.mse_loss(targets, train_pred).to(self.device)
        losses.backward(retain_graph=True)
        self.optimizer.step()
        loss = losses.item()
        return loss

    def fit(self):
        print("\nModel training...\n")

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)

        self.model.train()
        epochs = trange(self.args.epochs, leave=True, desc="Epoch")
        for epoch in epochs:

            train_loader_1, train_loader_2, targets = self.format_graph_pairs()

            # train
            self.loss_sum = 0
            batch_index = 0
            for index, zipped in enumerate(zip(train_loader_1, train_loader_2, targets)):
                batch1, batch2, target_list = zipped
                loss_score = self.process_batch(batch1, batch2, target_list)
                batch_index += len(batch1)
                self.loss_sum += (loss_score * len(batch1))
                loss = self.loss_sum / batch_index
                epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))

    def predict(self):
        print("\n\nProducing embeddings...\n")
        graph_loader = self.format_graph_single(self.graphs)
        embeddings = []

        for graph_batch in graph_loader:
            embedding_batch = self.model.conv_pass(graph_batch.feature.float(), graph_batch.edge_index,
                                                   graph_batch.batch)
            for i in embedding_batch:
                embeddings.append(i.cpu().detach().numpy())

        pkl.dump(embeddings, open(self.args.EMB_SAVE_PATH, 'wb'))

    def test(self):
        print("\n\nTEST: Producing embeddings for unseen graphs...\n")
        graph_loader = self.format_graph_single(self.test_graphs)
        embeddings = []

        for graph_batch in graph_loader:
            embedding_batch = self.model.conv_pass(graph_batch.feature.float(), graph_batch.edge_index,
                                                   graph_batch.batch)
            for i in embedding_batch:
                embeddings.append(i.cpu().detach().numpy())

        pkl.dump(embeddings, open(self.args.EMB_SAVE_PATH, 'wb'))

    def find_similarity(self):
        print("\n\nFind Similarity...\n")
        self.real_idx = pkl.load(open(self.args.IDX_PATH, 'rb'))

        G1 = self.graphs[self.real_idx.index(self.args.g1)]
        G2 = self.graphs[self.real_idx.index(self.args.g2)]

        train_loader_1 = DataLoader([from_networkx(G1)], batch_size=1)
        train_loader_2 = DataLoader([from_networkx(G2)], batch_size=1)
        sim = self.model(train_loader_1, train_loader_2)

        print(sim)

    def load(self):
        print('Loading pre-existing model...')
        self.model.load_state_dict(torch.load(self.args.LOAD_PATH))

    def save(self):
        print('Saving model in {}...'.format(self.args.SAVE_PATH))
        torch.save(self.model.state_dict(), self.args.SAVE_PATH)