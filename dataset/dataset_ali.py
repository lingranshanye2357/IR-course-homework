import os


class DataAli:

    def __init__(self, data_dir="./data_ali/"):
        self.data_dir = data_dir
        self.subs = ['ecom', 'med', 'video']
        self.sub_dirs = {sub: os.path.join(self.data_dir, sub) for sub in self.subs}
        self.dsts = ['train', 'dev', 'test']

        self.corpus = {x: None for x in self.subs}
        self.data = {x: {} for x in self.subs}
        for x in self.data.keys():
            self.data[x] = {dst: {} for dst in self.dsts}

    def load_sub_corpus(self, sub):
        sub_dir = self.sub_dirs[sub]
        # load corpus
        corpus_file = os.path.join(sub_dir, "corpus.tsv")
        corpus_idmap = {}
        corpus = []
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if line == "":
                    break
                doc_id, doc = line.split("\t")
                # corpus_id.append(doc_id)
                corpus_idmap[doc_id] = len(corpus)
                corpus.append(doc)
        self.corpus[sub] = {"corpus_id": corpus_idmap, "corpus": corpus}

    def load_sub_dst(self, sub, dst):
        sub_dir = self.sub_dirs[sub]
        # load train, test, dev
        query_file = os.path.join(sub_dir, "{}_query.tsv".format(dst))
        # queries_id = []
        queries_id2idx = {}
        queries_idx2id = {}
        queries = []
        with open(query_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if line == "":
                    break
                query_id, query = line.split("\t")
                # queries_id.append(query_id)
                queries_id2idx[query_id] = len(queries)
                queries_idx2id[len(queries)] = query_id
                queries.append(query)
        self.data[sub][dst]['queries_id2idx'] = queries_id2idx
        self.data[sub][dst]['queries_idx2id'] = queries_idx2id
        self.data[sub][dst]["queries"] = queries

        hit_file = os.path.join(sub_dir, "{}.tsv".format(dst))
        hits = {}
        with open(hit_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if line == "":
                    break
                query_id, doc_id = line.split('\t')
                hits[query_id] = doc_id
        self.data[sub][dst]["hits"] = hits

    def load_data(self):
        for sub in self.subs:
            self.load_sub_corpus(sub)
            for dst in self.dsts:
                self.load_sub_dst(sub, dst)

    def get_sub_corpus(self, sub='ecom'):
        if sub not in self.subs:
            raise KeyError("invalid key: \'{}\'".format(sub))
        if self.corpus[sub] is None:
            self.load_sub_corpus(sub)
        corpus = self.corpus[sub]
        return corpus['corpus_id'], corpus['corpus']

    def get_sub_dst(self, sub='ecom', dst='train'):
        if sub not in self.subs:
            raise KeyError("invalid key: \'{}\'".format(sub))
        if dst not in self.dsts:
            raise KeyError("invalid key: \'{}\'".format(dst))
        if self.data[sub][dst] == {}:
            self.load_sub_dst(sub, dst)
        data = self.data[sub][dst]
        return data['queries'], data['hits'],data['queries_id2idx'], data['queries_idx2id']
