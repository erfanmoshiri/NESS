"""Demo: print the related-work category tree to the terminal.
Edit CATEGORIES below to restructure; run `python relwork_tree_demo.py`.
This is just for eyeballing the hierarchy before drawing it in LaTeX.
"""

# Each node: (label, [children]). Leaves carry example methods in the label.
TREE = (
    "Learning under Missing Node Features",
    [
        ("Imputation-based", [
            ("Statistical / Propagation  [KNN, NeighAggre, FP, PCFI]", []),
            ("Learned / Generative  [SAT, SVGA, GAIN, GINN]", []),
        ]),
        ("Imputation-free / Architectural  [GCNMF, PaGCN]", []),
        ("Per-node Parameterization  [MATE, ITR, RITR, Amer]", []),
        ("Structure-leaning / Weak-info  [D2PT, T2-GNN, IDGL, Pro-GNN]", []),
        ("Self-Supervised & Multi-View", [
            ("General graph SSL (technique neighbors)", [
                ("Contrastive / multi-view  [DGI, GMI, MVGRL, GRACE, GCA]", []),
                ("Masked / reconstruction  [GraphMAE, MaskGAE, RARE]", []),
            ]),
            ("SSL for attribute-missing (problem neighbors)  [AmGCL, MATE, MOBA, AIAE, NCSSL]", []),
        ]),
    ],
)


def render(node, prefix="", is_last=True, is_root=True):
    label, children = node
    if is_root:
        print(label)
    else:
        connector = "└── " if is_last else "├── "
        print(prefix + connector + label)
        prefix += "    " if is_last else "│   "
    for i, child in enumerate(children):
        render(child, prefix, is_last=(i == len(children) - 1), is_root=False)


if __name__ == "__main__":
    render(TREE)
    print("\n(NESS = ours; sits under 'SSL for attribute-missing')")
