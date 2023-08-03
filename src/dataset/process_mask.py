import json
import re
import pandas as pd
import html
from tqdm import tqdm
movie_pattern = re.compile(r'@\d+')


def process_utt(utt, movieid2name, replace_movieId, remove_movie=False):
    def convert(match):
        movieid = match.group(0)[1:]
        if movieid in movieid2name:
            if remove_movie:
                return '[MASK]'
            movie_name = movieid2name[movieid]
            # movie_name = f'<soi>{movie_name}<eoi>'
            return movie_name
        else:
            return match.group(0)

    if replace_movieId:
        utt = re.sub(movie_pattern, convert, utt)
    utt = ' '.join(utt.split())
    utt = html.unescape(utt)

    return utt




def process(data_file, out_file, movie_set):
    total_test = 0
    with open(data_file, 'r', encoding='utf-8') as fin, open(out_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            total_test += 1
            dialog = json.loads(line)
            if len(dialog['messages']) == 0:
                continue

            movieid2name = dialog['movieMentions']
            name2movieid = {}
            if len(movieid2name) !=0:
                for k,v in movieid2name.items():
                    if v is None:
                        continue
                    name2movieid[v.replace(' ','')] = '@'+str(k)
            user_id, resp_id = dialog['initiatorWorkerId'], dialog['respondentWorkerId']
            context, resp = [], ''
            context_words = []
            entity_list = []
            messages = dialog['messages']
            turn_i = 0
            while turn_i < len(messages):
                worker_id = messages[turn_i]['senderWorkerId']
                utt_turn = []
                entity_turn = []
                movie_turn = []
                movie_token_turn = []
                mask_utt_turn = []
                word_turn = []

                turn_j = turn_i
                while turn_j < len(messages) and messages[turn_j]['senderWorkerId'] == worker_id:
                    
                    utt = messages[turn_j]['text']
                    utt_turn.append(utt)

                    mask_utt = process_utt(messages[turn_j]['text'], movieid2name, replace_movieId=True, remove_movie=True)
                    mask_utt_turn.append(mask_utt)

                    entity_ids = [entity2id[entity] for entity in messages[turn_j]['entity'] if entity in entity2id]
                    entity_turn.extend(entity_ids)

                    

                    movie_token = [name2movieid[movie.replace(" ","")] for movie in messages[turn_j]['movie_name'] if movie.replace(" ","") in name2movieid]
                    movie_token_turn.extend(movie_token)
                    

                    # 用这种方式去对，一定能保证有movie_token与movie_ids对齐
                    movie_ids = [movie_ids_index[movies_with_mentions.index(int(m_v[1:]))]for m_v in movie_token]
                    movie_turn.extend(movie_ids)   

                    assert len(movie_token_turn) == len(movie_turn)


                    word_ids = [word2id[word] for word in messages[turn_j]['word_name'] if word in word2id]
                    word_turn.extend(word_ids)
                    turn_j += 1

                utt = ' '.join(utt_turn)
                mask_utt = ' '.join(mask_utt_turn)

                if worker_id == user_id:
                    context.append(utt)
                    context_words.extend(word_ids)
                    entity_list.append(entity_turn + movie_turn)
                else:
                    resp = utt

                    context_entity_list = [entity for entity_l in entity_list for entity in entity_l]
                    context_entity_list_extend = []
                    # entity_links = [id2entity[id] for id in context_entity_list if id in id2entity]
                    # for entity in entity_links:
                    #     if entity in node2entity:
                    #         for e in node2entity[entity]['entity']:
                    #             if e in entity2id:
                    #                 context_entity_list_extend.append(entity2id[e])
                    context_entity_list_extend += context_entity_list
                    context_entity_list_extend = list(set(context_entity_list_extend))

                    if len(context) == 0:
                        context.append('')

                    rec_movie_index = []
                    for m_t in movie_turn:
                        if m_t in movie_ids_index:
                            rec_movie_index.append(movie_ids_index.index(m_t))
                        else:
                            raise
                    turn = {
                        'context': context,
                        'resp': mask_utt,
                        'rec_movie_index': rec_movie_index,
                        'rec_movie_token':movie_token_turn,
                        'entity': context_entity_list_extend,
                        'word': context_words
                    }



                    fout.write(json.dumps(turn, ensure_ascii=False) + '\n')

                    context.append(resp)
                    entity_list.append(movie_turn + entity_turn)
                    movie_set |= set(movie_turn)

                turn_i = turn_j

if __name__ == '__main__':
    with open('entity2id.json', 'r', encoding='utf-8') as f:
        entity2id = json.load(f)
    with open('concept2id.json','r' ,encoding='utf-8') as f:
        word2id = json.load(f)
    with open('movie_ids.json','r',encoding='utf-8') as f:
        movie_ids_index = json.load(f)
    id2entity = {v: k for k, v in entity2id.items()}




    movies_with_mentions = pd.read_csv('../../movies_with_mentions.csv')
    movies_with_mentions = movies_with_mentions.set_index('movieId').T.to_dict('list')
    movies_with_mentions = list(movies_with_mentions.keys())
    movie_set = set()



    

    process('valid_data_dbpedia.jsonl', 'valid_data_processed.jsonl', movie_set)
    process('test_data_dbpedia.jsonl', 'test_data_processed.jsonl', movie_set)
    process('train_data_dbpedia.jsonl', 'train_data_processed.jsonl', movie_set)
    print(f'#movie: {len(movie_set)}')