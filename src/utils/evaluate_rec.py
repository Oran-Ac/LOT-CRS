import math
from collections import defaultdict
import torch
from .args import Movie_Name_Path
import pandas as pd
movieCsv = pd.read_csv(Movie_Name_Path['redial'],usecols=['movieId','nbMentions'])



class RecEvaluator:
    def __init__(self, k_list=None, device=torch.device('cpu'),longtail_bar=4):
        if k_list is None:
            k_list = [1, 10, 50]
        self.k_list = k_list
        self.device = device

        self.item_set = defaultdict(set)
        self.tail_item_set = defaultdict(set)
        self.longTail_test = 0
        self.longTail_total = 0

        self.metric = {}
        self.reset_metric()
        self.movie2num = dict()
        for i,row in movieCsv.iterrows():
            self.movie2num[i] = row['nbMentions']
            if row['nbMentions'] <= 4:
                self.longTail_total += 1
        self.longtail_bar = longtail_bar

    def evaluate(self, logits, labels):
        for logit, label in zip(logits, labels):
            for k in self.k_list:
                self.metric[f'recall@{k}'] += self.compute_recall(logit, label, k)

                for item in logit[:k]:
                    self.item_set[f"coverage@{k}"].add(item)
                    if item >= 6925:
                        raise
                    if self.movie2num[item] <= self.longtail_bar:
                        self.tail_item_set[f"tail_coverage@{k}"].add(item)

                # if self.movie2num[label] <= self.longtail_bar: #只计算label为长尾时，其的正确率
                #     self.metric[f"Correcttail@{k}"]+= self.compute_recall(logit, label, k)
                #     self.longTail_test += 1
                
            self.metric['count'] += 1

    def compute_recall(self, rank, label, k):
        return int(label in rank[:k])

    def compute_mrr(self, rank, label, k):
        if label in rank[:k]:
            label_rank = rank.index(label)
            return 1 / (label_rank + 1)
        return 0

    def compute_ndcg(self, rank, label, k):
        if label in rank[:k]:
            label_rank = rank.index(label)
            return 1 / math.log2(label_rank + 2)
        return 0

    def reset_metric(self):
        # for metric in ['recall', 'coverage', 'tail_coverage','Correcttail']:
        for metric in ['recall', 'coverage', 'tail_coverage']:
            for k in self.k_list:
                self.metric[f'{metric}@{k}'] = 0
        self.metric['count'] = 0
        self.longTail_test = 0
        self.item_set = defaultdict(set)
        self.tail_item_set = defaultdict(set)

    def report(self):
        report = {}
        for k, v in self.metric.items():
            report[k] = torch.tensor(v, device=self.device)[None]
        for k, v in self.item_set.items():
            report[k] = torch.tensor(len(list(v)), device=self.device)[None]
        for k, v in self.tail_item_set.items():
            report[k] = torch.tensor(len(list(v)), device=self.device)[None]
        return report