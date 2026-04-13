import math
import pandas as pd
from collections import Counter
from goatools.obo_parser import GODag

parent_go = "GO:0043231"
DATA = "data/"

gaf_cols = [
"DB","DB_Object_ID","Gene","Qualifier","GO_ID","DB_Reference",
"Evidence","With","Aspect","Name","Synonym","Type","Taxon",
"Date","Assigned_By","Extension","Gene_Product_Form_ID"
]

gaf = pd.read_csv(
    DATA + "goa_human.gaf",
    sep="\t",
    comment="!",
    header=None,
    dtype=str,
    names=gaf_cols,
    na_filter=False
)
# print(gaf)

dag = GODag(DATA + "go-basic.obo")
print(dag.paths_to_top(parent_go))

counts = Counter(gaf["GO_ID"])
# print(counts)

total = sum(counts.values())
# print(total)

ancestors = dag[parent_go in dag["is_a"].split(" ! ")[0]]
# print(ancestors)

prob = {t: c/total for t,c in counts.items()}
# print(prob)

IC = {t: -math.log(p) for t,p in prob.items()}
# print(IC)

parent_annotations = gaf[gaf["GO_ID"]]
# print(parent_annotations)
