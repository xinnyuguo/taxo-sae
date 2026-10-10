"""Uniform access to the three systems compared in the paper.

    hier    : Hierarchical SAE   — supervised slots at every level
    hybrid  : Baseline 2         — the Hierarchical SAE's SAE1 (frozen) + unsupervised SAE2 / SAE3
    full    : Baseline 1         — unsupervised SAE1 / SAE2 / SAE3

Every system exposes the same interface so the tests can be written once:
    L1, L2, L3            level modules, each  forward(x) -> (z, reconstruction)
    leaf_dirs [d, n_leaf] steering direction of each leaf   (decoder column / mean of k feature columns)
    class_dirs[d, n_cls]  steering direction of each class
    read_leaf(z1), read_class(z2), read_root(z3)  -> node index (argmax)
"""
import torch
from .models.hier_value_model import HierValueSAE, SAELevel, load_property_tables
from .models.unsup_sae import TraditionalSAE


def load_hier(path, property_labels, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    _, _, vocab_sizes, _, _, _, _ = load_property_tables(property_labels, device)
    m = HierValueSAE(ck["d_model"], ck["n_free"], ck["n1"], ck["n2"], ck["n3"],
                     vocab_sizes, value_head=ck.get("value_head", "bilinear2")).to(device)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m, ck


def load_unsup_level(ck, key, device):
    m = TraditionalSAE(ck["d_model"], ck["n_features"]).to(device)
    m.load_state_dict(ck[key]); m.eval()
    return m


def feature_sets(ck, key, k=None):
    """{node index: [feature ids]} from a differential-activation checkpoint; k truncates
    the stored (score-sorted) top-5 list so the k-sweep needs no extra checkpoints."""
    return {int(i): (v["feature_indices"] if k is None else v["feature_indices"][:k])
            for i, v in ck[key].items()}


class System:
    def __init__(self, kind, tree, device, hier_ckpt=None, unsup_ckpt=None, property_labels=None, k=None):
        self.kind, self.tree, self.device = kind, tree, device
        n_leaf, n_cls, n_root = len(tree.leaves), len(tree.classes), len(tree.roots)
        if kind == "hier":
            self.hier, hck = load_hier(hier_ckpt, property_labels, device)
            self.n_free = hck["n_free"]
            self.L1, self.L2, self.L3 = self.hier.sae1, self.hier.sae2, self.hier.sae3
            nf = self.n_free
            self.leaf_dirs = self.L1.decoder.weight[:, nf:nf + n_leaf].detach()
            self.class_dirs = self.L2.decoder.weight[:, nf:nf + n_cls].detach()
            self.read_leaf = lambda z1: self.L1.concept_logits(z1)[0].argmax().item()
            self.read_class = lambda z2: self.L2.concept_logits(z2)[0].argmax().item()
            self.read_root = lambda z3: self.L3.concept_logits(z3)[0].argmax().item()
            self.root_axis = None
            self.leaf_feats = lambda i: [self.n_free + i]
            self.class_feats = lambda c: [self.n_free + c]
        else:
            uck = torch.load(unsup_ckpt, map_location="cpu", weights_only=False)
            self.uck = uck
            CF, RF = feature_sets(uck, "class_features", k), feature_sets(uck, "root_features", k)
            self.L2, self.L3 = load_unsup_level(uck, "sae2", device), load_unsup_level(uck, "sae3", device)
            if kind == "hybrid":
                self.hier, hck = load_hier(hier_ckpt, property_labels, device)
                self.n_free = hck["n_free"]; nf = self.n_free
                self.L1 = self.hier.sae1
                self.leaf_dirs = self.L1.decoder.weight[:, nf:nf + n_leaf].detach()
                self.read_leaf = lambda z1: self.L1.concept_logits(z1)[0].argmax().item()
                self.leaf_feats = lambda i: [self.n_free + i]
            elif kind == "full":
                LF = feature_sets(uck, "leaf_features", k)
                self.L1 = load_unsup_level(uck, "sae1", device)
                W1 = self.L1.decoder.weight.detach()
                self.leaf_dirs = torch.stack([W1[:, LF[i]].mean(1) for i in range(n_leaf)], 1)
                self.read_leaf = lambda z1: torch.tensor([z1[0, LF[i]].mean() for i in range(n_leaf)]).argmax().item()
                self.leaf_feats = lambda i: LF[i]
            else:
                raise ValueError(kind)
            W2 = self.L2.decoder.weight.detach()
            self.class_dirs = torch.stack([W2[:, CF[c]].mean(1) for c in range(n_cls)], 1)
            self.read_class = lambda z2: torch.tensor([z2[0, CF[c]].mean() for c in range(n_cls)]).argmax().item()
            self.read_root = lambda z3: torch.tensor([z3[0, RF[r]].mean() for r in range(n_root)]).argmax().item()
            self.CF, self.RF = CF, RF
            self.class_feats = lambda c: CF[c]

    # ---- convenience ------------------------------------------------------------------
    @torch.no_grad()
    def forward_stack(self, h):
        """h -> (z1, h1) -> (z2, h2) -> (z3, h3)"""
        z1, h1 = self.L1(h); z2, h2 = self.L2(h1); z3, h3 = self.L3(h2)
        return (z1, h1), (z2, h2), (z3, h3)

    @torch.no_grad()
    def read_all(self, h):
        (z1, _), (z2, _), (z3, _) = self.forward_stack(h)
        return self.read_leaf(z1), self.read_class(z2), self.read_root(z3)

    @torch.no_grad()
    def read_all_abstain(self, h, tau):
        """read_all with abstention: per level, the R node scores (Hier: concept slots; baselines: the nodes'
        selected features) are compared and the argmax is returned only if it beats the runner-up by more than
        tau * |top|; otherwise -1 (abstain, scored as wrong)."""
        out = []
        for lv, (z, _) in enumerate(self.forward_stack(h)):
            if self.kind == "hier" or (self.kind == "hybrid" and lv == 0):
                v = (self.L1, self.L2, self.L3)[lv].concept_logits(z)[0]
            else:
                feats = ({i: f for i, f in enumerate(self.leaf_feats(i) for i in range(len(self.tree.leaves)))}
                         if lv == 0 else self.CF if lv == 1 else self.RF)
                v = torch.stack([z[0, feats[i]].mean() for i in range(len(feats))])
            top = v.topk(2).values
            out.append(int(v.argmax()) if top[0] - top[1] > tau * top[0].abs() else -1)
        return tuple(out)


def build_system(args, tree, device):
    """Shared CLI convention: --system {hier,hybrid,full} --hier_model --unsup_model [--k]."""
    return System(args.system, tree, device, hier_ckpt=args.hier_model, unsup_ckpt=args.unsup_model,
                  property_labels=args.property_labels, k=getattr(args, "k", None))


def add_system_args(ap, default_hier="models/hier_sae_goal/hier_value_final.pt", allow_none=False):
    choices = ["hier", "hybrid", "full"] + (["none"] if allow_none else [])
    ap.add_argument("--system", required=True, choices=choices,
                    help="hier = Hierarchical SAE, hybrid = Baseline 2, full = Baseline 1" + (", none = no SAE" if allow_none else ""))
    ap.add_argument("--hier_model", default=default_hier,
                    help="Hierarchical SAE checkpoint (used by hier, and as SAE1 by hybrid)")
    ap.add_argument("--unsup_model", default="", help="unsupervised checkpoint (hybrid / full)")
    ap.add_argument("--k", type=int, default=None,
                    help="baselines: use the top-k differential features per node (default: all 5 stored)")
    ap.add_argument("--tree", default="data/00_taxonomy/tree.json")
    ap.add_argument("--property_labels", default="data/00_taxonomy/property_labels.json")
