"""HierSAE + property block + per-slot multi-label value head, all 3 levels.

Extends the sequential HierSAE (12 leaf / 4 class / 2 root concept slots) with, at
EACH level, a property block and a per-slot value head:

  slot layout per level:  [ n_free free | n_concept concept | P property ]
      P = 20 property slots (one gating slot per property key)

  value head (per level): for each property p, a small MLP maps that property
      slot's scalar activation -> logits over property p's value vocabulary V_p.
      => 20 independent value predictors, "per_slot" routed by property index.

Targets are MULTI-LABEL (multi-hot over V_p): a node's value(s) for property p =
the UNION of its descendant leaves' values (leaf: single value). Trained with BCE.
At eval, top-K_layer logits per property must CONTAIN all true values.

  K_layer: layer1=1, layer2=3, layer3=4  (max distinct values any node holds).

property_labels.json (built by build_hier_property_labels.py) supplies vocab + unions.
"""
import json
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F


class SAELevel(nn.Module):
    """[free | concept | property] encoder/decoder. concept slots -> concept logits.

    value head (--value_head):
      "per_slot"  : each property slot's scalar -> MLP -> logits over its vocab.
                    Reads ONLY property slots; concept is NOT an input, so swapping
                    concept does NOT change the value output (not steerable).
      "bilinear"  : concept identity IS an input. concept_slots -> softmax -> soft
                    embedding c_emb; for each property p, Hadamard(W_c c_emb,
                    W_p prop_emb[p]) -> per-property output head -> logits over V_p.
                    Swapping the concept slot changes c_emb -> changes value output
                    (steerable), mirroring the main-line EmbeddingBilinearValueHead.
    """
    def __init__(self, d_model, n_free, n_concept, prop_vocab_sizes,
                 value_head="bilinear", val_hidden=32, d_emb=64, d_hidden=128):
        super().__init__()
        self.n_free = n_free
        self.n_concept = n_concept
        self.P = len(prop_vocab_sizes)
        self.n_slots = n_free + n_concept + self.P
        self.value_head = value_head
        self.prop_vocab_sizes = list(prop_vocab_sizes)
        self.encoder = nn.Linear(d_model, self.n_slots, bias=True)
        self.decoder = nn.Linear(self.n_slots, d_model, bias=True)
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data[:, :d_model].T.clone()

        if value_head == "per_slot":
            self.value_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(1, val_hidden), nn.GELU(), nn.Linear(val_hidden, v))
                for v in prop_vocab_sizes
            ])
        elif value_head == "bilinear":
            # property identity = FIXED per-property embedding (property slots unused).
            # concept identity -> soft embedding (this is what makes it swappable)
            self.concept_emb = nn.Embedding(n_concept, d_emb)
            self.prop_emb = nn.Embedding(self.P, d_emb)
            self.W_c = nn.Linear(d_emb, d_hidden, bias=False)
            self.W_p = nn.Linear(d_emb, d_hidden, bias=False)
            # one output projection per property (different vocab sizes)
            self.out_heads = nn.ModuleList([nn.Linear(d_hidden, v) for v in prop_vocab_sizes])
        elif value_head == "bilinear2":
            # MAIN-LINE-ALIGNED: both WHOLE slot vectors are inputs.
            #   c_w = softmax(concept_slots) -> c_emb   (swappable)
            #   p_w = softmax(property_slots) -> p_emb  (property identity read from slots)
            #   shared = W_c(c_emb) (Hadamard) W_p(p_emb)
            # then a per-property output head projects the shared hidden to each
            # property's value vocab (multi-label). property_emb size = P (one row per
            # property slot), mirroring EmbeddingBilinearValueHead's property_emb.
            self.concept_emb = nn.Embedding(n_concept, d_emb)
            self.property_emb = nn.Embedding(self.P, d_emb)
            self.W_c = nn.Linear(d_emb, d_hidden, bias=False)
            self.W_p = nn.Linear(d_emb, d_hidden, bias=False)
            self.out_heads = nn.ModuleList([nn.Linear(d_hidden, v) for v in prop_vocab_sizes])
        else:
            raise ValueError(value_head)

    def forward(self, h):
        z = self.encoder(h)
        return z, self.decoder(z)

    def concept_logits(self, z):
        return z[:, self.n_free:self.n_free + self.n_concept]

    def property_slots(self, z):
        return z[:, self.n_free + self.n_concept:]          # [B, P]

    def value_logits(self, z):
        """Returns list of length P; entry p is [B, V_p] multi-label logits."""
        if self.value_head == "per_slot":
            ps = self.property_slots(z)
            return [self.value_heads[p](ps[:, p:p + 1]) for p in range(self.P)]

        if self.value_head == "bilinear":
            # concept-driven; property identity = fixed per-property embedding index
            c_w = F.softmax(self.concept_logits(z), dim=-1)   # [B, n_concept]
            c_emb = c_w @ self.concept_emb.weight             # [B, d_emb]
            hc = self.W_c(c_emb)                              # [B, d_hidden]
            pe = self.W_p(self.prop_emb.weight)               # [P, d_hidden]
            return [self.out_heads[p](hc * pe[p].unsqueeze(0)) for p in range(self.P)]

        # bilinear2: MAIN-LINE-ALIGNED — both whole slot vectors are inputs.
        c_w = F.softmax(self.concept_logits(z), dim=-1)       # [B, n_concept]
        p_w = F.softmax(self.property_slots(z), dim=-1)       # [B, P]  <- whole vector
        c_emb = c_w @ self.concept_emb.weight                 # [B, d_emb]
        p_emb = p_w @ self.property_emb.weight                # [B, d_emb]
        hc = self.W_c(c_emb)                                  # [B, d_hidden]
        hp = self.W_p(p_emb)                                  # [B, d_hidden]
        shared = hc * hp                                      # [B, d_hidden] Hadamard
        return [self.out_heads[p](shared) for p in range(self.P)]


class HierValueSAE(nn.Module):
    def __init__(self, d_model, n_free, n1, n2, n3, prop_vocab_sizes, value_head="bilinear"):
        super().__init__()
        self.sae1 = SAELevel(d_model, n_free, n1, prop_vocab_sizes, value_head)
        self.sae2 = SAELevel(d_model, n_free, n2, prop_vocab_sizes, value_head)
        self.sae3 = SAELevel(d_model, n_free, n3, prop_vocab_sizes, value_head)

    def forward(self, h):
        z1, h1 = self.sae1(h)
        z2, h2 = self.sae2(h1)
        z3, h3 = self.sae3(h2)
        return (z1, h1), (z2, h2), (z3, h3)


def load_property_tables(path, dev):
    """Returns (props, vocab_sizes, K, multihot_by_layer) where multihot_by_layer[layer]
    is a tensor [n_nodes, P, max_V] multi-hot (0/1) padded to max vocab; plus a mask of
    valid vocab entries [P, max_V] and per-prop vocab lists for decoding."""
    pl = json.load(open(path))
    props = pl["properties"]
    vocab = pl["value_vocab"]                       # prop -> [values]
    vidx = {p: {v: i for i, v in enumerate(vocab[p])} for p in props}
    vocab_sizes = [len(vocab[p]) for p in props]
    Vmax = max(vocab_sizes)
    K = pl["K"]

    # valid-entry mask [P, Vmax]
    vmask = torch.zeros(len(props), Vmax)
    for pi, p in enumerate(props):
        vmask[pi, :vocab_sizes[pi]] = 1.0
    vmask = vmask.to(dev)

    def layer_tensor(layer_key, node_order):
        L = pl["layers"][layer_key]
        T = torch.zeros(len(node_order), len(props), Vmax)
        for ni, node in enumerate(node_order):
            for pi, p in enumerate(props):
                for v in L[node][p]:
                    T[ni, pi, vidx[p][v]] = 1.0
        return T.to(dev)

    return props, vocab, vocab_sizes, vmask, vidx, K, layer_tensor
