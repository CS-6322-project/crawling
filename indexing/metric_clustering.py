import json
import os
import timeit
import numpy as np
DATA_FOLDER = "project_output_test/5k_result/"




# Load inverted index (X1)
with open(os.path.join(DATA_FOLDER, 'inverted_index.json'), 'r') as f:
    inverted_index = json.load(f)

# Load TF-IDF scores (X2)
with open(os.path.join(DATA_FOLDER, 'tf_idf_scores.json'), 'r') as f:
    tf_idf = json.load(f)
    
    
# Load doc_term_freq scores (X2)
with open(os.path.join(DATA_FOLDER, 'doc_term_freq.json'), 'r') as f:
    doc_term_freq = json.load(f)
    
    
with open(os.path.join(DATA_FOLDER, 'doc_term_pos.json'), 'r') as f:
    doc_term_pos = json.load(f)

print("Indexes loaded successfully for metric cluster.")

stop = timeit.default_timer()

def build_metric_cluster_expansion(query="mona", data_folder=DATA_FOLDER, max_docs=25, top_k=5):

    start = timeit.default_timer()

    print("Loading indexes and ranking data...")

    query = "mona"


    ###CHECK THIS, IF I USE ALL DOCS, IT TAKES A LOT OF TIME###
    ### I KEEP THIS To 500 TO MINIMIZE THE TIME BUT IT SHOULD BE ALL DOCS###
    doc_list = inverted_index[query][0:25]
    doc_count = len(doc_list)

    all_tokens = []
    for doc_id in doc_list:
        all_tokens.extend(list(doc_term_freq[doc_id].keys()))
        
    unique_tokens = set(all_tokens)

    stop = timeit.default_timer()

    print('Time: ', stop - start)  

    metric_cluster_result = {}
    score = 0
    for term in unique_tokens:
        for doc_id in doc_list:     
            if query in doc_term_pos[doc_id]:       
                for i in doc_term_pos[doc_id][query]:
                    if term in doc_term_pos[doc_id] and term != query:
                        for j in doc_term_pos[doc_id][term]:
                            if term not in metric_cluster_result:
                                metric_cluster_result[term] = 0
                            metric_cluster_result[term] += 1 / abs(i - j)
                            
                            
    stop = timeit.default_timer()
    sorted_result = sorted(metric_cluster_result.items(), key=lambda x: x[1], reverse=True)
    print('Time: ', stop - start)  


    # DIDN't NORMALIZE BECAUSE we only stems...

    sorted_metric_cluster_result = sorted(metric_cluster_result.items(), key=lambda x: x[1], reverse=True)
    print("Metric Cluster Result:")
    for term, score in sorted_metric_cluster_result[0:5]:
        print(f"Term: {term}, Score: {score}")

    stop = timeit.default_timer()

    print('Time: ', stop - start) 
    return [{"query": query, "term": term, "score": score} for term, score in sorted_result[:top_k]]
 