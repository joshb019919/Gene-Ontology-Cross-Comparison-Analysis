# Gene-Ontology-Cross-Comparison-Analysis
This project serves to compare three different methods of analyzing gene ontology (GO) term hierarchies.  The methods involve deep learning with the BioBERT transformer, statistically using Poincaré discs, and with three LLMs.  We have limited our focus specifically to cellular structure components, ignoring underlying cellular and overall biological functions.  We also only consider the 37 terms of the intracellular membrane-bound organelle and it's 36 direct children, for simplicity.

Basically speaking, when hierarchical structure is held constant, we want to know what determines semantic closeness among GO terms by comparing multiple semantic similarity regimes.

In utilizing the terminology hierarchy of gene ontology in medical research, scientists must make determinations based on the idea of the terms being aligned in the form of a directional acyclic graph (DAG) with multiple inheritance.  It can be thought of as a tree structure on paper, even though it's actually a DAG.  Terms near the "top" of a tree or "center" of a graph are more general, like the concept of the larger, outermost parts of a human cell.  If they're "further down" or "further out" from top or center, they're more specific, specialized terms.  Imagine that the cell is a more outer or higher component, the organelle are contained within the cell, and each organelle could have any number of individual componenets.  The components making up the organelle are, in turn, made up of subcomponents, and so on.

Terms on the same level of the hierarchy should be able to be assumed as having equally specific structure (mitochondrion and nucleus are both organelles and direct children of "organelle," which itself is a child of "cellular component".  Terms near one another on the graph (direct parents, children, and siblings) should be able to be assumed as being more biologically similar than terms further away from one another.  However, in practice this is not true, and especially in the middle and toward the bottom of the graph, terms on the same level may not represent the same level of specificity at all.  Terms near one another, graphcially, may not even be part of the same biological meaning.

## 
  
## Transformer: BioBERT

## Hierarchy: Poincaré discs

## LLMs: GPT open-weight and Ollama
The three LLMs used are the free, open-weight models, GPT-OSS:20B, GPT-OSS:120B, and Ollama.  
