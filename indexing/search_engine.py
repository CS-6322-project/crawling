import json
import os
import re
import math
import random
import numpy as np
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Import clustering functions from modular files
from association_cluster import build_association_cluster_expansion
from metric_clustering import build_metric_cluster_expansion
from scaler_clustering import build_scalar_cluster_expansion

nltk.download('stopwords', quiet=True)
DATA_FOLDER = "project_output_test/5k_result"

class SearchEngine:
    def __init__(self, data_folder=DATA_FOLDER):
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.data_folder = data_folder

        self.load_indexes()
        self.load_url_map()

        self.query_history = {}
        self.relevance_feedback = {}

        # Clustering data (lazy-loaded)
        self.term_co_occurrence = None
        self.metric_clusters = None
        self.scalar_clusters = None

        print("Search engine initialized and ready.")

    def load_indexes(self):
        with open(os.path.join(self.data_folder, 'inverted_index.json'), 'r') as f:
            self.inverted_index = json.load(f)

        with open(os.path.join(self.data_folder, 'tf_idf_scores.json'), 'r') as f:
            self.tf_idf = json.load(f)

        with open(os.path.join(self.data_folder, 'pagerank_scores.json'), 'r') as f:
            self.pagerank = json.load(f)

        with open(os.path.join(self.data_folder, 'hits_scores.json'), 'r') as f:
            hits = json.load(f)
            self.hubs = hits.get("hubs", {})
            self.authorities = hits.get("authorities", {})

    def load_url_map(self):
        self.url_map = {}
        try:
            with open('url_ids.jsonl', 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    self.url_map[str(data['url_id'])] = data['url']
        except FileNotFoundError:
            print("URL map file not found!")

    def preprocess_query(self, text):
        return [self.ps.stem(w) for w in re.findall(r'\w+', text.lower()) if w not in self.stop_words]

    def get_matching_docs(self, query_tokens):
        docs = set()
        for token in query_tokens:
            docs.update(self.inverted_index.get(token, []))
        return docs

    def calculate_query_vector(self, query_tokens):
        query_tf = Counter(query_tokens)
        query_vector = {}
        N = len(self.tf_idf)

        for token, freq in query_tf.items():
            tf = 1 + math.log(freq)
            df = len(set(self.inverted_index.get(token, [])))
            idf = math.log(N / (1 + df)) if df else 0
            query_vector[token] = tf * idf
        return query_vector

    def calculate_cosine_similarity(self, query_vec, doc_id):
        dot, q_mag, d_mag = 0.0, 0.0, 0.0
        doc_vec = self.tf_idf.get(doc_id, {})
        for term, q_w in query_vec.items():
            d_w = doc_vec.get(term, 0.0)
            dot += q_w * d_w
        q_mag = math.sqrt(sum(w ** 2 for w in query_vec.values()))
        d_mag = math.sqrt(sum(w ** 2 for w in doc_vec.values()))
        return dot / (q_mag * d_mag) if q_mag and d_mag else 0.0

    def expand_query(self, method, query, relevant=None, non_relevant=None):
        if method == "rocchio":
            return self.rocchio_expansion(query, relevant or [], non_relevant or [])
        elif method == "association":
            return build_association_cluster_expansion(query, self.tf_idf, self.inverted_index, self.ps, self.stop_words)
        elif method == "metric":
            return build_metric_cluster_expansion(query, self.tf_idf, self.ps, self.stop_words)
        elif method == "scalar":
            return build_scalar_cluster_expansion(query, self.tf_idf, self.ps, self.stop_words)
        else:
            return self.preprocess_query(query)

    def rocchio_expansion(self, query_text, rel_docs, nonrel_docs, alpha=1, beta=0.75, gamma=0.15):
        orig = self.calculate_query_vector(self.preprocess_query(query_text))
        new_query = {t: w * alpha for t, w in orig.items()}
        for doc in rel_docs:
            for t, w in self.tf_idf.get(doc, {}).items():
                new_query[t] = new_query.get(t, 0) + beta * w / len(rel_docs)
        for doc in nonrel_docs:
            for t, w in self.tf_idf.get(doc, {}).items():
                new_query[t] = new_query.get(t, 0) - gamma * w / len(nonrel_docs)
        return [t for t, w in sorted(new_query.items(), key=lambda x: x[1], reverse=True) if w > 0][:10]

    def rank_results(self, docs, query_vec, method="combined"):
        scores = {}
        for doc in docs:
            tfidf_score = self.calculate_cosine_similarity(query_vec, doc)
            pr = self.pagerank.get(self.url_map.get(doc, ""), 0)
            hit = self.authorities.get(self.url_map.get(doc, ""), 0)

            if method == "tfidf":
                scores[doc] = tfidf_score
            elif method == "pagerank":
                scores[doc] = pr
            elif method == "hits":
                scores[doc] = hit
            else:  # combined
                scores[doc] = tfidf_score * 0.6 + pr * 0.3 + hit * 0.1

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def search(self, query, rank_method="combined", expand_method=None, rel=None, nonrel=None, max_results=10):
        if expand_method:
            tokens = self.expand_query(expand_method, query, rel, nonrel)
        else:
            tokens = self.preprocess_query(query)
        docs = self.get_matching_docs(tokens)
        qvec = self.calculate_query_vector(tokens)
        ranked = self.rank_results(docs, qvec, rank_method)
        return [{"doc_id": doc, "url": self.url_map.get(doc, "N/A"), "score": round(score, 4)} for doc, score in ranked[:max_results]]
