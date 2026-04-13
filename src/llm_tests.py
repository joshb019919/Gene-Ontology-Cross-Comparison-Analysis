import json
import os
import re
from pathlib import Path
from ollama import ChatResponse, chat
from pydot import Any

MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

MESSAGES = [{
        "role": "system", "content": """

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
    """},{
        "role": "user", "content": """

    "clustering": "Group each point within a probable cluster and output the likeliest value for k and which point and organelle belongs to which cluster in the graph.",
    "radius_significance": "What is the significance of the clusters based on radius?",
    "theta_significance": "What is the significance of the clusters based on theta?",
    "combined_significance": "What is the significance, if any, of the clusters based on both radius and theta?",
    "parasite_targets": "Given the significance of each cluster, what are the likeliest organelles to target for genes regarding research into fighting parasite exfiltration into the larger cell and body?",
"""}]


def next_response_path() -> Path:
    """Return next output path as results/json/response_<n>.json."""
    output_dir = Path("results/json/")
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(r"^response_(\d+)\.json$")
    latest = 0
    for file_path in output_dir.glob("response_*.json"):
        match = pattern.match(file_path.name)
        if match:
            latest = max(latest, int(match.group(1)))

    return output_dir / f"response_{latest + 1}.json"


def benchmark_log_path() -> Path:
    """Return benchmark log path in results/json."""
    output_dir = Path("results") / "json"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "benchmark_log.jsonl"


def ns_to_seconds(ns, default=0.0):
    return default if ns is None else ns / 1e9


def save_runtimes(response: ChatResponse) -> None:
    durations = {
        "model": response.model,
        "total_duration": ns_to_seconds(response.total_duration),
        "load_duration": ns_to_seconds(response.load_duration),
        "eval_duration": ns_to_seconds(response.eval_duration),
        "prompt_eval_duration": ns_to_seconds(response.prompt_eval_duration)
    }
    with benchmark_log_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(durations, indent=2) + "\n")


def save_response(response: ChatResponse) -> None:
    output_path = next_response_path()
    content = response.message.content
    content_text = content if isinstance(content, str) else ""

    with output_path.open("w", encoding="utf-8") as f:
        parsed = json.loads(content_text)
        json.dump(parsed, f, indent=2, ensure_ascii=False)

    print(f"Saved output to {output_path}")


def main(payload: dict[str, Any]) -> None:
    response = chat(**payload)
    print(response.message.content)

    # Create next response file for content
    save_response(response)

    # Store Ollama benchmark runtimes
    save_runtimes(response)


if __name__ == "__main__":
    payloads = [
        {"model": "gpt-oss:20b", "messages": MESSAGES, "stream": False, "think": "low", "format": "json", "options": {"seed": 42, "temperature": 0, "num_threads": 8,}},
        {"model": "llama3.2:latest", "messages": MESSAGES, "stream": False, "format": "json", "options": {"seed": 42, "temperature": 0, "num_threads": 8,}},
        {"model": "llama3:latest", "messages": MESSAGES, "stream": False, "format": "json", "options": {"seed": 42, "temperature": 0, "num_threads": 8,}}
    ]

    for payload in payloads:
        main(payload)
