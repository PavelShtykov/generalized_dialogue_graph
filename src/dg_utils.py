from cProfile import label
import imp
from urllib.request import proxy_bypass
from xml.dom.minicompat import EmptyNodeList
from tqdm import trange
from random import choice
import warnings
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import Callable, List, Sequence, Tuple, Union
from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer
import pandas as pd
from collections import defaultdict, Counter
import re
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import networkx as nx
from IPython.display import Image
from sklearn.feature_extraction.text import TfidfVectorizer
import textwrap


class Preproccesing():
    def __init__(self, company_words: List[str] = [], use_stop_words=True, use_lemmatization=True):
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.use_stop_words = use_stop_words
        self.use_lem = use_lemmatization
        self.word_counter = Counter()
        self.company_words = set([w.lower() for w in company_words])

    def preprocessing_string(self, s: str) -> str:
        s = re.sub(r'http\S+', '', s)
        text = list(filter(str.isalpha, word_tokenize(s.lower())))
        if self.use_lem:
            text = list(self.lemmatizer.lemmatize(word) for word in text)
        text = list("companyname" if w in self.company_words else w for w in text)
        if self.use_stop_words:
            text = list(word for word in text if word not in self.stop_words)

        for word in text:
            self.word_counter[word] += 1

        return ' '.join(text)

    def preprocessing_list_strings(self, l: List[str]) -> List[str]:
        res_list = [self.preprocessing_string(s) for s in l]

        return res_list

    def preprocessing_corpus(self, corp: List[List[str]]) -> List[List[str]]:
        res = corp.copy()
        for i in tqdm(range(len(corp)), desc="Preproccesing ", leave=False):
            res[i] = self.preprocessing_list_strings(corp[i])
        
        return res
        

class Embbeding():
    def __init__(self, embder_name: str = "all-mpnet-base-v2", normalize=True):
        self.normalize = normalize
        self.embder = SentenceTransformer(embder_name)

    def embed_string(self, utter: str) -> np.ndarray:
        return self.embder.encode(utter)

    def embed_list_of_strings(self, dialog: List[str]) -> np.ndarray:
        ret = self.embder.encode(dialog,
                                normalize_embeddings=self.normalize,
                                convert_to_numpy=True)
        
        return ret
    
    def embed_corpus(self, corpus: List[List[str]]) -> List[np.ndarray]:
        ret = []
        for dialog in tqdm(corpus, desc="Embedding: ", leave=False):
            ret.append(self.embed_list_of_strings(dialog))
        
        return ret


class TurboDataset():
    def __init__(
        self, 
        dialogues: List[List[str]],
        preprocessor: Preproccesing,
        embedder: Embbeding,
        pad_start_end_utter: bool = True
    ):
        self.preprocessor = preprocessor
        self.embedder = embedder
        self.dialogues = dialogues
        self.dials_lens = [len(d) for d in self.dialogues]
        self.prepr_dials = self.preprocessor.preprocessing_corpus(self.dialogues)
        self.dials_embds = self.embedder.embed_corpus(self.prepr_dials) 

        self.utter_to_idx = {}
        base_idx = 1 if pad_start_end_utter else 0
        for utter in self.get_all_prepr_dials():
            if utter not in self.utter_to_idx.keys():
                self.utter_to_idx[utter] = base_idx
                base_idx += 1
        self.idx_to_utter = {v: k for k, v in self.utter_to_idx.items()}
    
    def get_all_dials(self) -> List[str]:
        return sum(self.dialogues, [])

    def get_all_prepr_dials(self) -> List[str]:
        return sum(self.prepr_dials, [])
    
    def get_all_embds(self) -> np.ndarray:
        return np.concatenate(self.dials_embds, axis=0)


class BaseStatisticEvaluator():
    def get_statistic(self, p_u_giv_v: np.ndarray) -> List[str]:
        pass


class SelfLabelStat(BaseStatisticEvaluator):
    def __init__(
        self,
        dataset: TurboDataset,
        pad_start_end_utter: bool = True,
        top_k_utter_to_find = 5,
        lenth_bound: Tuple[int] = (2, 5),
        ):
        self.lenth_bound = lenth_bound
        self.dataset = dataset
        self.pad_start_end = pad_start_end_utter
        self.top_k = top_k_utter_to_find
    
    def get_statistic(self, p_u_giv_v: np.ndarray) -> List[str]:
        ret = []
        s = slice(1, -1) if self.pad_start_end else ...
        for u_probs in p_u_giv_v[s]:
            u_probs = u_probs[s]
            kth = min((self.top_k, u_probs.shape[0]-1))
            idxes = np.argpartition(-u_probs, kth)[:kth]
            # idxes = idxes[u_probs[idxes] > self.eps]
            idxes += 1 if self.pad_start_end else 0

            pretenders = [self.dataset.idx_to_utter[i] for i in idxes \
                            if self.lenth_bound[0] <= len(self.dataset.idx_to_utter[i].split(" ")) <= self.lenth_bound[1]]
            ret.append(choice(pretenders))

        if self.pad_start_end:
            ret.insert(0, "BEGIN")
            ret.append("END")

        return ret


class TfIdfStat(BaseStatisticEvaluator):
    def __init__(
        self,
        dataset: TurboDataset,
        top_k: int = 50,
        pad_start_end_utter: bool = True,
        eps = 1e-3,
        max_df=1,
        min_df=1
        ):
        self.top_k = top_k
        self.dataset = dataset
        self.pad_start_end = pad_start_end_utter
        self.tfidf = TfidfVectorizer(min_df=min_df, max_df=max_df)
        self.eps = eps
        # self.tfidf.fit(self.dataset.utter_to_idx.keys())
    
    def get_statistic(self, p_u_giv_v: np.ndarray) -> List[str]:
        corpus = []
        s = slice(1, -1) if self.pad_start_end else ...
        # print(p_u_giv_v.shape)
        for u_probs in p_u_giv_v[s]:
            u_probs = u_probs[s]
            kth = min((self.top_k, u_probs.shape[0]-1))
            idxes = np.argpartition(-u_probs, kth)[:kth]
            idxes = idxes[u_probs[idxes] > self.eps]
            idxes += 1 if self.pad_start_end else 0

            doc = " ".join([self.dataset.idx_to_utter[i] for i in idxes])
            corpus.append(doc)
        
        embs = self.tfidf.fit_transform(corpus).toarray()
        feat_list = np.array(self.tfidf.get_feature_names())
        ret = []
        for emb in embs:
            idxes = np.argpartition(-emb, 4)[:4]
            ret.append(", ".join(feat_list[idxes]))
        
        if self.pad_start_end:
            ret.insert(0, "BEGIN")
            ret.append("END")
        
        return ret


class TurboModel():
    def __init__(
        self,
        dataset: TurboDataset,
        v_giv_u_predictor: Callable,
        is_soft_predicter: bool,
        num_v: int,
        pad_start_end_utter: bool = True
        ):
        # self.node_labeler = node_labeler
        self.num_v = num_v
        self.num_v += 2 if pad_start_end_utter else 0
        if is_soft_predicter:
            self.v_giv_u_fun = v_giv_u_predictor
        else:
            self.v_giv_u_fun = lambda x: self._get_soft_clustering(v_giv_u_predictor(x))
        self.dataset = dataset
        self.pad_start_end = pad_start_end_utter

        self.num_u = len(self.dataset.utter_to_idx)
        self.num_u += 2 if pad_start_end_utter else 0
        self.p_u = np.zeros(self.num_u, dtype=np.float32)
        # self.adj_in_U = np.zeros((self.num_u, self.num_u), dtype=np.float32)
        self.adj_in_U = csr_matrix((self.num_u, self.num_u), dtype=np.float32)
        self.p_v_giv_u = np.zeros((self.num_u, self.num_v), dtype=np.float32)
        if self.pad_start_end:
            self.p_v_giv_u[0, 0] = 1.0
            self.p_v_giv_u[-1, -1] = 1.0

        self.adj_in_V = None
        self.p_u_giv_v = None

    def _get_soft_clustering(self, clustering: np.ndarray) -> np.ndarray:
        base_num_v = self.num_v
        base_num_v -= 2 if self.pad_start_end else 0
        soft_clustering = np.zeros((clustering.shape[0], base_num_v), dtype=clustering.dtype)
        soft_clustering[np.arange(soft_clustering.shape[0]), clustering] = 1
        
        return soft_clustering

    def _get_p_u_giv_v(self, p_v_giv_u: np.ndarray, p_u: np.ndarray) -> np.ndarray:
        numerator = p_v_giv_u.T * p_u

        return numerator / numerator.sum(axis=1, keepdims=True)

    def build_graph(self) -> Tuple[np.ndarray, np.ndarray]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for dial, emb_dial in tqdm(zip(self.dataset.prepr_dials, self.dataset.dials_embds), desc="Graph building: ", leave=False):
                if self.pad_start_end:
                    self.p_u[0] += 1
                    self.p_u[-1] += 1
                    self.adj_in_U[0, self.dataset.utter_to_idx[dial[0]]] += 1
                    self.adj_in_U[self.dataset.utter_to_idx[dial[-1]], -1] += 1

                from_idxes = [self.dataset.utter_to_idx[u] for u in dial[:-1]]
                to_idxes = [self.dataset.utter_to_idx[u] for u in dial[1:]]
                all_idxes = from_idxes + [self.dataset.utter_to_idx[dial[-1]]]
                
                np.add.at(self.p_u, all_idxes, 1)
                
                for idx in zip(from_idxes, to_idxes):
                    self.adj_in_U[idx] += 1
                
                if self.pad_start_end:
                    self.p_v_giv_u[all_idxes, 1:-1] = self.v_giv_u_fun(emb_dial)
                else:
                    self.p_v_giv_u[all_idxes, :] = self.v_giv_u_fun(emb_dial)

            self.p_u = self.p_u / self.p_u.sum()

            adj_in_U_denum = self.adj_in_U.sum(axis=1).reshape(-1, 1)
            adj_in_U_denum[adj_in_U_denum==0] = 1
            self.adj_in_U =  self.adj_in_U  / adj_in_U_denum

            self.p_u_giv_v = self._get_p_u_giv_v(self.p_v_giv_u, self.p_u)

            self.adj_in_V = self.p_u_giv_v.dot(self.adj_in_U).dot(self.p_v_giv_u)

            self.adj_in_V = np.array(self.adj_in_V)

            return self.adj_in_V, self.p_u_giv_v
        
    def draw_graph(self,
                    node_labeler: BaseStatisticEvaluator,
                    path_to_save: str = "./dialog_graph.svg",
                    min_prob=0.1) -> Image:
        G = nx.from_numpy_matrix(self.adj_in_V, create_using=nx.DiGraph)

        labels = node_labeler.get_statistic(self.p_u_giv_v)
        labels = [textwrap.fill(s, 13) for s in labels]
        nx.set_node_attributes(G, {k: {'label': labels[k],
                                        "shape": "box",
                                        "fontsize": 12}
                                        for k in range(self.num_v)})
        if self.pad_start_end:
            nx.set_node_attributes(G, {i: {'fillcolor': "gray",
                                            "style": "filled",
                                            "shape": "ellipse"}
                                            for i in [0, len(labels)-1]})
        nx.set_edge_attributes(G, {(e[0], e[1]): {
                                                    # 'label': f"{e[2]['weight']:.2f}",
                                                    "fontsize": 11, 
                                                    "penwidth": 7*e[2]['weight'] + 0.2,
                                                    } for e in G.edges(data=True)})

        # filter out all edges above threshold and grab id's
        long_edges = list(filter(lambda e: e[2] < min_prob, (e for e in G.edges.data('weight'))))
        le_ids = list(e[:2] for e in long_edges)

        # remove filtered edges from graph G
        G.remove_edges_from(le_ids)

        D = nx.drawing.nx_agraph.to_agraph(G)

        # Modify node fillcolor and edge color.
        # D.node_attr.update(color='blue', style='filled', fillcolor='yellow')
        D.edge_attr.update(arrowsize=1)
        pos = D.layout('dot')
        D.draw(path_to_save)

        return Image(D.draw(format='png'))


def twitter_dialogs_extraction(base_df: pd.DataFrame, companies: List[str]) -> pd.DataFrame:
    res = defaultdict(list)
    companies = set(companies)
    rev_base = base_df
    idx = base_df.index[-1]
    while idx >= base_df.index[0]:
        if  rev_base.at[idx, "in_response_to_tweet_id"] == -1 and \
            rev_base.at[idx, "inbound"]:

            bug_flag = False
            curr_authors = {rev_base.at[idx, "author_id"]}
            curr_dialog = [rev_base.at[idx, "text"]]

            idx -= 1
            while True:
                if  rev_base.at[idx, "in_response_to_tweet_id"] == -1 or \
                    rev_base.at[idx, "inbound"] == rev_base.at[idx+1, "inbound"] or \
                    rev_base.at[idx, "in_response_to_tweet_id"] != rev_base.at[idx+1, "tweet_id"] or \
                    rev_base.at[idx, "created_at"] < rev_base.at[idx+1, "created_at"]:
                    
                    bug_flag = True
                    break

                curr_dialog.append(rev_base.at[idx, "text"])
                curr_authors.add(rev_base.at[idx, "author_id"])

                if rev_base.at[idx, "response_tweet_id"] == -1:
                    break
                idx -= 1
                
            names_intersec = companies & curr_authors
            if  not bug_flag and \
                len(curr_authors) == 2 and \
                len(names_intersec) == 1:
                
                res[list(names_intersec)[0]].append(curr_dialog)

        idx -= 1
    
    for k, v in res.items():
        curr_df = pd.DataFrame()
        curr_df["dialog"] = v
        curr_df["lens"] = [len(l) for l in v]
        res[k] = curr_df

    return res