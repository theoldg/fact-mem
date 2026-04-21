import json
from dataclasses import asdict
from pathlib import Path
import torch
from tqdm import tqdm
from pythia import PythiaModel

# List of facts (prefix, suffix)
# Suffixes typically start with a space to align with token boundaries in Pythia's BPE.
FACTS = [
    # First batch
    ("Fact: Toki Pona is a", " language"),
    ("The capital of France is", " Paris"),
    ("Michael Jordan plays the sport of", " basketball"),
    ("The chemical symbol for water is", " H2O"),
    ("The Great Wall is located in", " China"),
    ("The Earth orbits the", " Sun"),
    ("The author of Hamlet is", " Shakespeare"),
    ("The largest ocean on Earth is the", " Pacific"),
    ("DNA stands for deoxyribonucleic", " acid"),
    ("The first human to step on the moon was", " Neil Armstrong"),
    ("The square root of 16 is", " 4"),

    # Geography
    ("The highest mountain in the world is Mount", " Everest"),
    ("The longest river in South America is the", " Amazon"),
    ("The continent directly south of Europe is", " Africa"),
    
    # Science & Space
    ("The powerhouse of the cell is the", " mitochondria"),
    ("The lightest element on the periodic table is", " hydrogen"),
    ("The red planet in our solar system is", " Mars"),
    ("Albert Einstein developed the theory of", " relativity"),
    ("The chemical symbol for gold is", " Au"),
    ("The hardest naturally occurring substance on Earth is", " diamond"),
    
    # History
    ("The Declaration of Independence was signed in the year", " 1776"),
    ("The first president of the United States was", " George Washington"),
    ("The ancient pyramids are located in", " Egypt"),
    
    # Arts & Literature
    ("The Mona Lisa was painted by", " Leonardo da Vinci"),
    ("The dystopian novel 1984 was written by", " George Orwell"),
    ("The wizarding school in Harry Potter is called", " Hogwarts"),
    
    # Technology & Math
    ("The creator of the Linux kernel is", " Linus Torvalds"),
    ("The base-2 numeral system is also known as", " binary"),
    ("A polygon with eight sides is called an", " octagon"),
    
    # General Knowledge & Pop Culture
    ("Mario's brother in the Nintendo franchise is", " Luigi"),
    ("The Beatles were originally a rock band from", " Liverpool"),
    ("The primary ingredient in guacamole is", " avocado"),
    ("The largest land animal is the", " elephant"),
    ("An adult human normally has 32", " teeth"),
]

steps = list(range(40000, 96000, 1000))
variant = "6.9b"

# Create output directory
output_dir = Path("results_fact_stats")
output_dir.mkdir(exist_ok=True)

for step in tqdm(steps, desc="Processing Pythia checkpoints"):
    output_path = output_dir / f"stats_step_{step}.json"
    
    # Skip if file already exists
    if output_path.exists():
        continue
        
    try:
        # Load the model for this specific step
        model = PythiaModel.from_variant_and_revision(variant=variant, revision=step)
        
        step_results = []
        for prefix, suffix in FACTS:
            try:
                stats = model.text_completion_stats(prefix, suffix)
                res = asdict(stats)
                res["prefix"] = prefix
                res["suffix"] = suffix
                step_results.append(res)
            except ValueError as e:
                tqdm.write(f"  Error for fact '{prefix}'/'{suffix}' at step {step}: {e}")
                
        # Save results for this specific step immediately
        with open(output_path, "w") as f:
            json.dump(step_results, f, indent=2)
            
        # Free memory to avoid OOM when processing multiple checkpoints
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        tqdm.write(f"  Failed to load model at step {step}: {e}")

print(f"\nDone! Results saved in {output_dir}/")
