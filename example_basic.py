"""
This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""
from sentence_transformers import SentenceTransformer, util
import torch

embedder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# Corpus with example sentences
corpus = ['兄弟营 电视剧 海顿,翁虹,赵恒煊,杨峰,范哲琛,叶浏,孙湘光,李夏佩,杜厚佳,贺睿,蒋楚依,杜金京,辛修岗,武颖恣,胡腾,鲍柏宇,宋恩逸,陈诚,韩一思,文江,陈子轩,李虓,付志明,樊梓田,申峥 海顿 赵石磊,田妞,老狼,梅山海,沙龙,牛四斤,崔主任,叶百合,于华清,武平,赵青莲,白露,大弹弓,赵剑刚,小疙瘩,韩有利,李泽木,豆腐章,李医生,王青天,山鸡,二娃子,汪彪,周顺,藤原 2018 浪花淘尽之以眼还眼 杜鹃花 猛虎营 兄弟团',
          '恐龙世界 霸王龙 恐龙动画片 恐龙乐园 恐龙帝国 恐龙世界 恐龙总动员',
          '儿童动画雪花彩泥粘土diy手工制作玩具视频教程大全 水果月饼',
          '刘墉下南京 张禄说梦',
          '益智动画 各种五颜六色的运动球教学习颜色',
          '别知己 果果吉特巴六',
          ]
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
print(corpus_embeddings.shape)  # [6, 384]

# Query sentences:
queries = ['谍战电视剧战争',
           '恐龙动画片儿',
           '彩泥月饼的制作方法',
           ]


# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(5, len(corpus))

# 方法1
# for query in queries:
#     query_embedding = embedder.encode(query, convert_to_tensor=True)
#
#     # We use cosine-similarity and torch.topk to find the highest 5 scores
#     cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
#     top_results = torch.topk(cos_scores, k=top_k)
#
#     print("\n\n======================\n\n")
#     print("Query:", query)
#     print("\nTop 5 most similar sentences in corpus:")
#
#     for score, idx in zip(top_results[0], top_results[1]):
#         print(corpus[idx], "(Score: {:.4f})".format(score))

# 方法二
query_embedding = embedder.encode(queries, convert_to_tensor=True)
hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
for i in range(len(queries)):
    print("\n\n======================\n\n")
    print("Query:", queries[i])
    print("\nTop 5 most similar sentences in corpus:")
    for hit in hits[i]:
        print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
