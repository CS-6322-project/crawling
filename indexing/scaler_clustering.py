import json
import os
import timeit
import numpy as np
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

print("Indexes loaded successfully for scaler clustering.")

def build_scalar_cluster_expansion(query="mona", data_folder=DATA_FOLDER, max_docs=25, top_k=5):
    start = timeit.default_timer()

    query = "mona"


    ###CHECK THIS, IF I USE ALL DOCS, IT TAKES A LOT OF TIME###
    ### I KEEP THIS To 500 TO MINIMIZE THE TIME BUT IT SHOULD BE ALL DOCS###
    doc_list = inverted_index[query][0:25]
    doc_count = len(doc_list)


    all_tokens = []
    for doc_id in doc_list:
        all_tokens.extend(list(doc_term_freq[doc_id].keys()))
        
    unique_tokens = list(set(all_tokens))
    association_matrix = {token: [0] * doc_count for token in unique_tokens}


    stop = timeit.default_timer()

    print('Time: ', stop - start)  

    token_mapping = {unique_tokens[i]:i for i in range(len(unique_tokens))}
    idx_mapping = {i:unique_tokens[i] for i in range(len(unique_tokens))}
    CM = np.zeros((len(unique_tokens), doc_count), dtype=int)


    cnt = 0
    for doc_id in doc_list:
        tokens = doc_term_freq[doc_id].keys()
        for token in tokens:
            CM[token_mapping[token]][cnt] += doc_term_freq[doc_id][token]
        cnt += 1
        
    SM = np.matmul(CM, CM.T)

        
    stop = timeit.default_timer()

    print('Time: ', stop - start)  


    query = "gogh"
    scaler_cluster_result = {}
    for term in unique_tokens:
        if term != query:
            s_uv = np.dot(SM[token_mapping[term]], SM[token_mapping[query]])
            s_uv = s_uv / (np.linalg.norm(SM[token_mapping[term]]) * np.linalg.norm(SM[token_mapping[query]]))
            scaler_cluster_result[term] = s_uv
            
    scaler_cluster_result = sorted(scaler_cluster_result.items(), key=lambda x: x[1], reverse=True)
    sorted_results = sorted(scaler_cluster_result.items(), key=lambda x: x[1], reverse=True)

    print("Scaler Cluster Result:")
    for term in scaler_cluster_result[0:5]:
        print(term[0], term[1])
        
    stop = timeit.default_timer()

    print('Time: ', stop - start) 
    return [{"query": query, "term": term, "score": score} for term, score in sorted_results[:top_k]]
 