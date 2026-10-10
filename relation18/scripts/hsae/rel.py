"""Synthetic biography world helpers (people.csv). Same interface as the FB helper: facts, split, rel_answers, node_union."""
import csv, collections


class RelData:
    def __init__(self, tree, people=None):
        import os; people = people or os.environ.get("HSAE_DATA", "data/generated") + "/people.csv"
        self.tree = tree
        rows = list(csv.DictReader(open(people, encoding="utf-8")))
        self.facts = {r["full_name"]: {rel: r[rel] for rel in tree.leaves} for r in rows}      # person -> {relation: value}
        self.split = {r["full_name"]: "train" for r in rows}
        self.rel_answers = collections.defaultdict(set)                                          # relation -> every value it takes
        for e, fr in self.facts.items():
            for r, a in fr.items(): self.rel_answers[r].add(a.lower().strip())

    def node_union(self, e, node, level):
        """union of e's values under a class (level 'class') or a root (level 'root'), in tree order (deduplicated)"""
        t = self.tree
        rs = [r for r in t.leaves if (t.cls_of[r] == node if level == "class" else t.root_of[t.cls_of[r]] == node)]
        vals = []
        for r in rs:
            if self.facts[e][r] not in vals: vals.append(self.facts[e][r])
        return vals, rs
