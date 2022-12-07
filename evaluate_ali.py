"""
evaluate model on Ali
"""
from dataset import *
from metric import *

from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from loguru import logger
from tqdm import tqdm


def embedding_by_batch(embedder, data, batch_size):
    embeddings = []
    for i in tqdm(range(0, len(data), batch_size)):
        data_batch = data[i: min(i+batch_size, len(data))]
        embedding = embedder.encode(data_batch, convert_to_tensor=True)
        embeddings.append(embedding)
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings


if __name__ == "__main__":
    logger.add("out_eval_ali.log")

    embedder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

    # load ali data
    data_ali = DataAli("./data_ali/")
    sub = "med"
    dst = 'dev'
    logger.info("sub: {}, dst: {}".format(sub, dst))

    # load corpus
    corpus_idmap, corpus = data_ali.get_sub_corpus(sub)
    logger.info("corpus len: {}".format(len(corpus)))
    # corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    corpus = corpus[:100]
    corpus_embeddings = embedding_by_batch(embedder, corpus, 8)
    torch.save(corpus_embeddings, "./corpus_embeddings.pt")
    # corpus_embeddings = torch.load('./corpus_embeddings.pt')
    logger.info("finished corpus embedding of size: {}".format(corpus_embeddings.shape))

    # load queries
    queries, hits_gt, queries_id2idx, queries_idx2id = data_ali.get_sub_dst(sub, dst)
    logger.info("queries len: {}".format(len(queries)))
    queries = queries[:100]
    query_embeddings = embedder.encode(queries, convert_to_tensor=True)
    torch.save(query_embeddings, "./query_embeddings.pt")
    logger.info("finished queries embedding of size: {}".format(query_embeddings.shape))

    # predict
    top_k = 100
    hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=top_k)
    hits_pred = np.zeros((len(hits), top_k))
    hits_true = np.zeros((len(hits), 1))
    for i in range(len(queries)):
        for j in range(hits_pred.shape[1]):
            hits_pred[i][j] = hits[i][j]['corpus_id']
        hits_true[i] = corpus_idmap[hits_gt[queries_idx2id[i]]]
    hits_pred = torch.from_numpy(hits_pred)
    hits_true = torch.from_numpy(hits_true)
    logger.info("top k: {}".format(top_k))

    # evaluate
    search_metric = SearchMetric()
    search_metric(hits_pred, hits_true)
    logger.info("metric: {}".format(search_metric.get()))
