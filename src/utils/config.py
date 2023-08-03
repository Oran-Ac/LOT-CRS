from collections import defaultdict
import yaml
import json
def redial_config(path,data_type): 
    def _entity_kg_process(opt, SELF_LOOP_ID=185):
        edge_list = []  # [(entity, entity, relation)]
        for entity in range(opt['n_entity']):
            if str(entity) not in opt['entity_kg']:
                continue
            edge_list.append((entity, entity, SELF_LOOP_ID))  # add self loop
            for tail_and_relation in opt['entity_kg'][str(entity)]:
                if entity != tail_and_relation[1] and tail_and_relation[0] != SELF_LOOP_ID:
                    edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
                    edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))

        relation_cnt, relation2id, edges, entities = defaultdict(int), dict(), set(), set()
        for h, t, r in edge_list:
            relation_cnt[r] += 1
        for h, t, r in edge_list:
            if relation_cnt[r] > 1000:
                if r not in relation2id:
                    relation2id[r] = len(relation2id)
                edges.add((h, t, relation2id[r]))
                entities.add(opt['id2entity'][h])
                entities.add(opt['id2entity'][t])
        return {
            'edge': list(edges),
            'n_relation': len(relation2id),
            'entity': list(entities)
        }
    def _word_kg_process(opt):
        edges = set()  # {(entity, entity)}
        entities = set()
        with open(opt['word_kg'],'r') as f:
            for line in f:
                kg = line.strip().split('\t')
                entities.add(kg[1].split('/')[0])
                entities.add(kg[2].split('/')[0])
                e0 = opt['word2id'][kg[1].split('/')[0]]
                e1 = opt['word2id'][kg[2].split('/')[0]]
                edges.add((e0, e1))
                edges.add((e1, e0))
        # edge_set = [[co[0] for co in list(edges)], [co[1] for co in list(edges)]]
        return {
            'edge': list(edges),
            'entity': list(entities)
        }
    config_dict = dict()
    with open(path, 'r', encoding='utf-8') as f:
        config_dict.update(yaml.safe_load(f.read()))
    with open(config_dict[data_type]['movie_ids_path'],'r') as f:
        movie_ids = json.load(f)
    config_dict['movie_ids'] = movie_ids
    
    with open(config_dict[data_type]['entity2id_path'],'r') as f:
        entity2id = json.load(f)
    with open(config_dict[data_type]['entity_kg_path'],'r') as f:
        entity_kg = json.load(f)
    with open(config_dict[data_type]['token2id_path'],'r') as f:
        token2id = json.load(f)
    with open(config_dict[data_type]['word2id_path'],'r') as f:
        word2id = json.load(f)
    config_dict['graph'] = {}
    config_dict['graph']['word_kg'] = config_dict[data_type]['concept_kg_path']
    config_dict['graph']['entity2id'] = entity2id
    config_dict['graph']['token2id'] = token2id
    config_dict['graph']['word2id'] = word2id
    config_dict['graph']['entity_kg'] = entity_kg
    config_dict['graph']['id2entity'] = {idx: entity for entity, idx in entity2id.items()}
    config_dict['graph']['n_entity'] = max(entity2id.values()) + 1
    config_dict['graph']['n_word'] = max(word2id.values()) + 1
    
    entity_kg_dict = _entity_kg_process(config_dict['graph'])
    word_kg_dict = _word_kg_process(config_dict['graph'])
    config_dict['graph']['entity_kg'] = entity_kg_dict
    config_dict['graph']['word_kg'] = word_kg_dict
    
    return config_dict