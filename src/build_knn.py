# build knn both for conversation and recommendation
'''
key_store: representation of the history utterance;[utterance_number,1,hidden_size]
rec_store: representation of the recommendation position (attention: before the classification layer) [utterance_number,1,hidden_size]
gen_store: representation of the generation utterance;[utterance_number,1,hidden_size]

注意选择的position
'''