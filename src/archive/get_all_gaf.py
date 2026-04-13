import pandas as pd
from goatools.obo_parser import GODag
from goatools.anno.gaf_reader import GafReader

DATA = "data/"

# -------------------------
# Load GO ontology
# -------------------------
go_dag = GODag(DATA + "go-basic.obo")

parent_go = "GO:0043231"

# parent + direct children
go_terms = [parent_go] + [
    child.id for child in go_dag[parent_go].children
]

# -------------------------
# Load GO annotations
# -------------------------
gaf = GafReader(DATA + "goa_human.gaf")
assoc = gaf.get_associations()

# Build GO -> genes map
go_to_genes = {}

for a in assoc:
    go_id = a.GO_ID
    gene = a.DB_Symbol
    
    if go_id not in go_to_genes:
        go_to_genes[go_id] = set()
        
    go_to_genes[go_id].add(gene)

# -------------------------
# Build dataframe
# -------------------------
records = []

for go in go_terms:
    
    name = go_dag[go].name
    
    genes = list(go_to_genes.get(go, []))
    
    records.append({
        "go_id": go,
        "organelle_name": name,
        "genes": genes
    })

df = pd.DataFrame(records)

print(df.truncate(None))

