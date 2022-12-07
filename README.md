# IR-final

## 12.1

关键假设：query没有关键词



#### 需求

背景：平时使用lofter、豆瓣等软件阅读文章时，发现其中的标签内搜索或组内搜索功能不足，表现为只能搜索题目中出现的关键词。但用户在使用软件过程中，会有更多的需求。

需求与目标：

1. 需求：用户只能想起文章内容中的关键词，但该关键词未出现在题目中。目标：实现**全文**搜索功能。
2. 需求：用户无法想到任何关键词，只能对**文章情节**做简单描述。目标：实现支持文章情节的搜索功能。
3. 需求：用户在阅读**长篇小说**时，对情节搜索功能的需求更加强烈，但长篇小说不同章节之间是有前后联系的。因此在搜索的过程中需要考虑文章前后的连贯性。举例说明：比如第一页写道”我去了家里的厨房做菜“，第二页写道”我做了一道西红柿炒鸡蛋“，第九页写“我去了餐馆”，第十页写“我吃了西红柿炒鸡蛋”。然后检索“在家吃西红柿炒鸡蛋”。应该返回1、2页，但如果不考虑连续性，很可能第10页匹配得更好。目标：实现支持长篇小说情节连续性的搜索。



#### 方法

faiss：

[(88条消息) 信息检索时，如果没有关键词命中，用faiss做语义召回是个不错的方法_zh515858237的博客-CSDN博客](https://blog.csdn.net/zh515858237/article/details/111038805)

query预处理：

https://zhuanlan.zhihu.com/p/112719984



* semantic search

https://www.searchenginejournal.com/semantic-search-how-it-works-who-its-for/438960/

> Semantic search attempts to apply user intent and the meaning (or semantics) of words and phrases to find the right content.
>
> It goes beyond keyword matching by using information that might not be present immediately in the text (the keywords themselves) but is closely tied to what the searcher wants.

wiki

> https://en.wikipedia.org/wiki/Semantic_search

向量检索：

https://zhuanlan.zhihu.com/p/161467314

SVD

https://www.sohu.com/a/245954412_100118110

Google:

https://www.tabpear.com/articles/Tabpear-seo-what-is-semantic-search

SimCSE+Ernie:

https://blog.csdn.net/Kaiyuan_sjtu/article/details/122183888

Transformer for search(SBERT):

https://blog.csdn.net/weixin_44826203/article/details/119868241

ELMo:

https://blog.csdn.net/u011984148/article/details/102481283

SBERT介绍：

https://zhuanlan.zhihu.com/p/383138444



#### 数据集

* 小说

文章-情节的数据集哪里来？

小说大部分带有介绍，对介绍进行一些处理能否作为“情节”。（感觉不太行）

* 文章

可以爬虫收集。





#### 其它

人名、情节等搜小说的网站：

[拨云搜索 - 专注于小说搜索引擎 (boyunso.com)](https://www.boyunso.com/)





#### 总结

检索模型的结构：

1. 需要一个提取语义的模型。例如SBERT, ELMo, SimCSE等。能够将query/doc表示成语义向量。
2. 训练策略。其一，**加载预训练语言模型**，不训练。其二，用IR数据集做有监督的微调。其三，用语料库做**无监督训练**（没有相关性标注，参考SimCSE）。
3. 用cosine等计算相似度，作为query-document之间的relevance。
4. 数据集。方法一：常规IR做有监督训练，文章数据集或者小说数据集做展示。方法二：文章数据集做无监督训练和展示。



工作：

1. 加载一个语义模型，做encoding。

    D

2. 优化计算相似度。

    W

3. IR数据集。

    D

4. 有监督训练。评价。

    D

5. 爬虫。爬取无监督语料。

    W

6. 无监督微调。

    D

7. 展示（poster或ppt）。





## 12.7

* paperswithcode

semantic search:

1. https://paperswithcode.com/paper/sgpt-gpt-sentence-embeddings-for-semantic

    google，bing等用的是bert-like的transformer架构，本文是gpt-like。

    （读后感：好复杂。。。）

2. https://paperswithcode.com/paper/learning-deep-structured-semantic-models-for

    DNN做non-linear mapping为128维向量，然后算cos similarity。论文模型图很清楚。和我们要做的不太一样，但是可以参考。模型名叫DSSM。



semantic retrieval:

https://paperswithcode.com/task/semantic-retrieval



* 选择Embedding Model

1. bert

    https://zhuanlan.zhihu.com/p/183952345

    实验显示bert比DSSM好。没有开源的code和datasets。

2. **sbert**

    https://zhuanlan.zhihu.com/p/383138444

    bert的原生缺陷是得到的vector不适合做similarity相关的任务，因此提出了sbert，更适合做检索，并且计算效率更高。

    paper: https://arxiv.org/abs/1908.10084

    code: https://github.com/UKPLab/sentence-transformers

    请深度学习此code，尤其examples/semantic_search。

结论：sbert



* 有监督数据集

1. 文献检索数据集（语料和小说差别太大）（并且不是document，只是短短的title）

    https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/neural_search

2. **阿里云**的，已经通过申请（三种语料和小说也不怎么沾边）

    https://tianchi.aliyun.com/dataset/135962?t=1670403444919】

3. msmarco（英语的）

    https://microsoft.github.io/msmarco/Datasets

（中文的信息检索数据集比想象的难找。。救救我）

（准确说是passage retrieval或者document retrieval，语料如果是新闻的就好了）

暂定：阿里云的



* 其它

1. SimCSE检索系统：

    [动手搭建一套语义检索系统-学术文献检索 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/452529430)

    https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/neural_search

2. zhengyima师兄的finetune bert on msmarco

    https://github.com/zhengyima/pretrain4ir_tutorial



* 实验

corpus太大，暂时跑不动。