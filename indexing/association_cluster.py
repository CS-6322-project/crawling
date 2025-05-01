import json
import os
import timeit
import numpy as np
import os

DATA_FOLDER = "project_output_test/5k_result/"

print("Loading indexes and ranking data...")

# Load inverted index (X1)
with open(os.path.join(DATA_FOLDER, 'inverted_index.json'), 'r') as f:
    inverted_index = json.load(f)

# Load TF-IDF scores (X2)
with open(os.path.join(DATA_FOLDER, 'tf_idf_scores.json'), 'r') as f:
    tf_idf = json.load(f)
    
    
# Load doc_term_freq scores (X2)
with open(os.path.join(DATA_FOLDER, 'doc_term_freq.json'), 'r') as f:
    doc_term_freq = json.load(f)

print("Indexes loaded successfully for association cluster.")

def build_association_cluster_expansion(query, data_folder, max_docs=500, top_k=5):

    start = timeit.default_timer()

    query = "mona"


    ###CHECK THIS, IF I USE ALL DOCS, IT TAKES A LOT OF TIME###
    ### I KEEP THIS To 500 TO MINIMIZE THE TIME BUT IT SHOULD BE ALL DOCS###
    doc_list = inverted_index[query][0:500]
    doc_count = len(doc_list)


    all_tokens = []
    for doc_id in doc_list:
        all_tokens.extend(list(doc_term_freq[doc_id].keys()))
        
    unique_tokens = set(all_tokens)
    association_matrix = {token: [0] * doc_count for token in unique_tokens}


    stop = timeit.default_timer()

    print('Time: ', stop - start)  

    cnt = 0
    for doc_id in doc_list:
        tokens = doc_term_freq[doc_id].keys()
        for token in tokens:
            association_matrix[token][cnt] += doc_term_freq[doc_id][token]
        cnt += 1
        
    stop = timeit.default_timer()

    print('Time: ', stop - start)  


    association_cluster_result = {}
    for term in unique_tokens:
        if term != query:
            c_uv =  sum([x * y for x, y in zip(association_matrix[term], association_matrix[query])])
            c_uu =  sum([x * y for x, y in zip(association_matrix[query], association_matrix[query])])
            c_vv =  sum([x * y for x, y in zip(association_matrix[term], association_matrix[term])])
            association_cluster_result[term] = c_uv / (c_uu + c_uv + c_vv)


    print("Association Cluster Result:")
    top_results = sorted(association_cluster_result.items(), key=lambda x: x[1], reverse=True)[:top_k]
    for term, score in sorted(association_cluster_result.items(), key=lambda x: x[1], reverse=True)[0:5]:
        print(f"Query: {query}, Term: {term}, Score: {score}")
        
    stop = timeit.default_timer()

    print('Time: ', stop - start)  
    print('--------------------------------')
    return {
        "query": query,
        "results": [{"term": term, "score": round(score, 5)} for term, score in top_results]
    }