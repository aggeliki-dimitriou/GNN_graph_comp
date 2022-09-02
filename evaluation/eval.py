import numpy as np
from sklearn.metrics import ndcg_score
import pickle as pkl
import rbo

##image_data.json.zip must be unzipped in its directory
##download embeddings_path2vec.pkl

def get_ranking(idx, geds, idx_list, rank = False):
  sim_keys = [key for key in geds.keys() if idx in key]
  
  similarities = []
  sim_idx = []
  for key in sim_keys:
    similarities.append(geds[key][0])
    if key[0] == idx:
      sim_idx.append(key[1])
    else:
      sim_idx.append(key[0])

  ordered_sim = [similarities[sim_idx.index(i)] if i != idx else 0.0 for i in idx_list]
  if rank:
    idx_ord_sim = list(np.argsort(ordered_sim))
    ordered_sim = [idx_ord_sim.index(i) for i in range(len(idx_ord_sim))]

  return ordered_sim

def find_all_ranks(geds,idx_list, rank = False):
  result = []
  idx_list = [elem[0] for elem in idx_list]
  for i in idx_list:
    a = get_ranking(i, geds, idx_list, rank)
    a_max = max(a)
    a = [j/a_max for j in a]
    result.append(a)

  return result

def find_ranks_kernels(sims, rank = False):
  sims_ranks = []

  for i in range(len(sims)):

    max_sim = max(sims[i])
    a = [j/max_sim for j in sims[i]]
    order_ranks = [1-j for j in a]

    if rank:
      idx_ord_sim = list(np.argsort(order_ranks))
      idx_ord_sim.reverse()
      order_ranks = [idx_ord_sim.index(i) for i in range(len(idx_ord_sim))]

    sims_ranks.append(order_ranks)

  return sims_ranks

def hit_percentage(a, b):
  assert np.array(a).shape == np.array(b).shape
  intermediate_hit = []
  for i in range(len(a)):
    hits = set(a[i]).intersection(set(b[i]))
    hits_per = len(hits)/len(a[i])
    intermediate_hit.append(hits_per)

  return intermediate_hit, np.mean(intermediate_hit)

def score_stats(gd, sims):
  hps = dict()
  rbos = dict()
  for k in [10, 5, 2]:
    hps[k] = list()
    rbos[k] = list()
    print('---- k = {} ----'.format(k))
    for sim in sims:
      all_hps, mean_hp = hit_percentage(np.array(sim)[:,:k], np.array(gd)[:,:k])
      hps[k].append(all_hps)
      print("Hit Percentage: {}".format(mean_hp))
      all_rbos = list()
      for i in range(amount):
        all_rbos.append(rbo.RankingSimilarity(gd[i][:k], sim[i][:k]).rbo())
      
      rbos[k].append(all_rbos)
      print("Mean RBO: {}".format(np.mean(all_rbos)))

  return hps, rbos

#most similar to 0
def most_similar_graph(idx, emb, ten=False):
  sorted_emb = list(np.argsort(emb[idx]))
  sorted_emb.remove(idx) # remove self
  if ten:
    sim = sorted_emb[-10:]
  else:
    sim = sorted_emb[-1] 
  return sim


def find_all_similar(emb, ten = False):
  size = emb.shape[0]
  similarities = []
  for i in range(size):
    sim = most_similar_graph(i, emb, ten)
    similarities.append(sim)
  return similarities

def main():
    large_idx = pkl.load(open('../data/scene500_large_idx.pkl', 'rb'))

    geds = pkl.load(open('../outs/geds_bipartite_costs.pkl', 'rb')) 

    gd_rank = find_all_ranks(geds, large_idx)
    gd_dict = pkl.load(open('../outs/ten_most_similar_large_bipartite.pkl', 'rb'))

    sims_rank = []
    sims_rank.append(pkl.load(open('../outs/sim_wl_rank.pkl', 'rb')))
    sims_rank.append(pkl.load(open('../outs/sim_pm_rank.pkl', 'rb')))
    sims_rank.append(pkl.load(open('../outs/sim500_pa_attr_rank.pkl', 'rb')))
    sims_rank.append(pkl.load(open('../outs/sim500_sm_attr_rank.pkl', 'rb')))
    sims_rank.append(pkl.load(open('../outs/sim500_gh_attr_rank.pkl', 'rb')))

    sims_rank_2 = []
    for i in range(5):
        sims_rank_2.append(find_ranks_kernels(list(sims_rank[i])))

    sims = []
    sims.append(pkl.load(open('../outs/sim500_wl.pkl', 'rb')))
    sims.append(pkl.load(open('../outs/sim500_pm.pkl', 'rb')))
    sims.append(pkl.load(open('../outs/sim500_pa_attr.pkl', 'rb')))
    sims.append(pkl.load(open('../outs/sim500_sm_attr.pkl', 'rb')))
    sims.append(pkl.load(open('../outs/sim500_gh_attr.pkl', 'rb')))

    gd = []
    for idx in large_idx:
        gd.append(gd_dict[idx[0]])

    gk_names = ['Weisfeiler-Lehman', 'Pyramid Match', 'Shortest Path', 'Propagation Attributes', 'Subgraph Matching', 'Graph Hopper']

    amount = 500

    ndcgs = dict()

    for k in [10, 5, 2]:
        ndcgs[k] = list()
        for sim in sims_rank_2:
            nn = list()
            for i in range(amount):
                nn.append(ndcg_score([gd_rank[i]], [sim[i]], k=k))

            ndcgs[k].append(nn)

    mean_ndcgs = dict()
    for k in [10, 5, 2]:
        mean_ndcgs[k] = list()
        for i in range(7):
            mean_ndcgs[k].append(np.mean(ndcgs[k][i]))

    print("NDCG:", mean_ndcgs)


    hps, rbos = score_stats(gd, sims)

if __name__ == "__main__":
    main()