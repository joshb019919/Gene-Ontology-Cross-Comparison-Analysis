import ollama
import asyncio
import json

# response = ollama.chat(
#     model="gpt-oss:20b",
#     messages=[{"role": "system", "content": 
# """
# You are an expert biomedical ontologist specializing in Gene Ontology (GO) cellular component terminology.

# Your task is to evaluate semantic specificity of GO-CC terms based strictly on biological meaning and annotation frequency, not DAG depth.

# Specificity definition:
# A term is more specific if it refers to a structurally, functionally, or spatially constrained biological entity.

# Rules:
# - Do not use graph level information.
# - Do not rely on term ID numbering.
# - Base judgments only on biological interpretation.
# - If uncertain, state uncertainty explicitly.

# Output format (JSON only):
# {
#   "more_specific_term": "...",
#   "less_specific_term": "...",
#   "confidence": 0-1,
#   "rationale": "concise explanation"
# }
# """},
#         {"role": "user", "content":
# """
# Example 1

# Input:
# mitochondrial inner membrane
# mitochondrial outer membrane
# cytoplasm

# Reasoning:
# The first two share the same organelle and similar specificity depth within mitochondrial substructures. Cytoplasm is broader and outside the organelle.

# Answer:
# Most similar pair: mitochondrial inner membrane & mitochondrial outer membrane

# Example 2

# Input:
# nucleus
# nucleolus
# plasma membrane

# Reasoning:
# Nucleolus and nucleus are organelles, plus nucleolus is contained within nucleus and closely related structurally. Plasma membrane is not an organelle and is unrelated spatially.

# Answer:
# Most similar pair: nucleus & nucleolus

# Example 3brane-bound organelles with analogous spatial and functional constraints, whereas the 

# Input:
# ER body
# Golgi apparatus
# vacuole

# Reasoning: 
# ER body and Golgi apparatus are both major organelles and similar in the function of DNA -> RNA -> protein.  Vacuoles are membrane-surrounded empty spaces which store ions in water for fluid balance, waste disposal, pH regulation, cell defense, and other uses.

# Answer:
# Most similar pair: ER body and Golgi apparatus

# Example 4

# Input:
# exoneme
# microneme
# plastid

# Reasoning: 
# Exonemes and micronemes are specialized secretory organelles designed to help invasive parasites dissolve membranes and egress into cells.  Plastids exist only in plants and algae to perform photosynthesis and other essential functions.

# Answer:
# Most similar pair: exoneme and microneme

# Example 5

# Input:
# anammoxosome
# nucleomorph
# nucleus

# Reasoning:
# Anammoxosomes and nucleomorphs are specialized or atrophied organelles and not present in humans.  The nucleus is a major human organelle.

# Answer:
# Most similar pair: anammoxosome and nucleomorph

# Question Prompt:
# Which two are more similar among the GO-CC triple (mitochondria, golgi body, plasma membrane)?
# """}
#     ]
# )

# print(response['message']['content'])

# Results of GPT-OSS:20b
"""
more_specific_term
mitochondria & golgi body

less_specific_term
plasma membrane

confidence
0.9,

rationale
Both mitochondria and Golgi body are intracellular, membrane-bound organelles with analogous 
spatial and functional constraints, whereas the plasma membrane is a cell surface boundary 
not comparable in structural or functional type.
"""

MODEL = "gpt-oss:20b"

SYSTEM_PROMPT = """
You are an expert biomedical ontologist specializing in Gene Ontology (GO) cellular component terminology.

Your task is to evaluate a CSV file of hyperbolic radius and angle points on a polar graph with truncated 
gene ontology names and answer questions regarding the points and their meanings in biological semantic
function and ontological hierarchy.

The CSV file of hyperbolic radius and angle points is:
r,theta,org_trunc
3.6030262658390537,-0.8606754760165877,intracellular mem
10.276373483168868,2.7326985155776704,ocelloid
9.357261718500713,2.6998419464814734,acidocalcisome
9.442128944379109,2.736903741185305,anammoxosome
8.791766147029014,2.587025817049477,attachment organe
6.204295609751624,-0.938403734671309,cis-Golgi network
3.314796536292332,-0.9003385747194074,endoplasmic retic
6.080795760405903,-1.567303352110034,endoplasmic retic
9.337457099139389,2.6696757665743984,endosperm protein
9.225299786318434,2.6048050260436,ER body
9.18641661526306,2.6852727962348726,exoneme
9.005278296580784,2.572589463136852,eyespot apparat
3.3854422630888705,-1.258941651030783,Golgi apparatus
9.440167045368854,2.740675978761161,hydrogenosome
10.53564772468889,2.749440215537313,intracellular ves
9.202745439893098,2.664812611479571,karyomere
9.358295089126528,2.7072336972323416,methane-oxidizing
8.911570785804694,2.7261277281321545,microbo
9.008621020743856,2.682461862018086,microneme
9.759188357257912,2.719942079951516,mitochondrial der
3.148026533705722,-0.2291251183763434,mitochondrion
9.37637844649191,2.7138369890762566,mitosome
9.237873621227005,2.683169107349264,mononeme
10.790193805131135,2.701212830828593,nucleomorph
2.5975203134554112,0.4289862104213759,nucleus
10.323443515870348,2.743509214994806,omegasome
9.49859937003134,2.6862451974875543,pirellulosome
7.727188112505983,2.710188751054769,plastid
9.553335300049556,2.8387499473157547,platelet dense tu
9.266471125796036,2.718662704754517,polyhydroxyalkano
10.764770154614002,2.8038163007904013,subsynaptic retic
8.089836660704012,2.7387035672287734,vacuole

Rules:
- If uncertain, state uncertainty explicitly.

Output format (JSON only).
"""

# Each prompt is sent as a separate async call so Ollama can process them concurrently.
PROMPTS = {
    "clustering":        "Group each point within a probable cluster and output the likeliest value for k and which point and organelle belongs to which cluster in the graph.",
    "radius_significance": "What is the significance of the clusters based on radius?",
    "theta_significance":  "What is the significance of the clusters based on theta?",
    "combined_significance": "What is the significance, if any, of the clusters based on both radius and theta?",
    "parasite_targets":  "Given the significance of each cluster, what are the likeliest organelles to target for genes regarding research into fighting parasite exfiltration into the larger cell and body?",
    "secretion":         "Which organelles involve secretion functions?",
    "nucleus_shared":    "Which organelles have shared functionality or biological specificity with the nucleus?",
}


async def ask(client: ollama.AsyncClient, label: str, user_text: str) -> tuple[str, str]:
    """Send a single prompt and stream the response token-by-token."""
    print(f"\n=== {label} ===", flush=True)
    content_parts = []
    async for chunk in await client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_text},
        ],
        stream=True,
    ):
        token = chunk["message"]["content"]
        print(token, end="", flush=True)
        content_parts.append(token)
    print()  # newline after streaming finishes
    return label, "".join(content_parts)


async def main():
    client = ollama.AsyncClient()
    # Fire all prompts concurrently; each streams its own output as tokens arrive.
    tasks = [ask(client, label, text) for label, text in PROMPTS.items()]
    results = await asyncio.gather(*tasks)

    # Pretty-print the accumulated JSON from each prompt
    print("\n\n=== Parsed JSON Results ===")
    for label, content in results:
        print(f"\n--- {label} ---")
        try:
            parsed = json.loads(content)
            print(json.dumps(parsed, indent=2, ensure_ascii=False, sort_keys=True))
        except json.JSONDecodeError:
            print(content)


asyncio.run(main())
