"""Data helpers, tree access and the question-surface tables shared by all experiments."""
import json
import pickle
import torch
from torch.utils.data import Dataset

# ---- taxonomy surface forms ----------------------------------------------------------
# how a class / root node is spelled inside a question (in-distribution forms)
CLASS_SURFACE = {"mammal": "mammal", "aquatic": "aquatic animal", "bird": "bird", "invert": "invertebrate"}
ROOT_SURFACE = {"vertebrate": "vertebrate", "invertebrate": "invertebrate"}
# unseen surface forms (none of these strings occurs in any training file) -> "new synonyms"
CLASS_SYN = {"mammal": "mammalian animal", "aquatic": "aquatic creature", "bird": "avian animal",
             "invert": "spineless animal"}
ROOT_SYN = {"vertebrate": "backboned animal", "invertebrate": "animal without a backbone"}


def art(s):
    return "an" if s[:1].lower() in "aeiou" else "a"


# ---- jsonl ----------------------------------------------------------------------------
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_json_list(path):
    data = json.load(open(path, "r", encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON file does not contain a list.")
    return data


# ---- tree -------------------------------------------------------------------------------
class Tree:
    """hier8/tree.json: leaves -> classes -> roots plus the parent maps and child masks."""
    def __init__(self, path):
        t = json.load(open(path))
        self.raw = t
        self.leaves, self.classes, self.roots = t["layer1"], t["layer2"], t["layer3"]
        self.cls_of, self.root_of = t["class_of_leaf"], t["root_of_class"]

    def class_idx_of_leaf(self, leaf):
        return self.classes.index(self.cls_of[leaf])

    def root_idx_of_leaf(self, leaf):
        return self.roots.index(self.root_of[self.cls_of[leaf]])

    def root_idx_of_class(self, cls):
        return self.roots.index(self.root_of[cls])

    def masks(self, device):
        return build_masks(self.raw, device)


def build_masks(tree_raw, device):
    """child masks from a raw tree dict: ([n_class, n_leaf], [n_root, n_class])"""
    m2 = torch.tensor([r["child_mask_over_layer1"] for r in tree_raw["layer2_rel"]], dtype=torch.float32, device=device)
    m3 = torch.tensor([r["child_mask_over_layer2"] for r in tree_raw["layer3_rel"]], dtype=torch.float32, device=device)
    return m2, m3


# ---- activations ------------------------------------------------------------------------
def load_activations(path):
    """Returns (H [N, d] float tensor, rows list) from a collect_activations pickle."""
    rows = pickle.load(open(path, "rb"))
    H = torch.stack([torch.as_tensor(r["h"]) for r in rows]).float()
    return H, rows


class ActDataset(Dataset):
    def __init__(self, path):
        self.data = pickle.load(open(path, "rb"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        return {"h": torch.from_numpy(d["h"]).float(),
                "concept_idx": int(d["concept_idx"]),
                "property_idx": int(d.get("property_idx", -1))}


# ---- answer matching --------------------------------------------------------------------
def match(v, a):
    v, a = str(v).lower().strip(), str(a).lower().strip()
    return bool(v) and bool(a) and (v in a or a in v)


def first_word(t):
    t = t.split("<end_of_text>")[0].strip().split("\n")[0]
    t = t.split(".", 1)[0].split(",", 1)[0].strip()
    return t.split()[0] if t.split() else t
