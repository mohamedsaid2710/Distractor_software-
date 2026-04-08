import matplotlib.pyplot as plt
import numpy as np
import re
import fasttext
import fasttext.util

# === CONFIGURATION ===
INPUT_FILE = 'txt.txt'
SEMANTIC_THRESHOLD = 0.35  # Must match your params_de.txt
LANG = 'de'

# 1. Load FastText model
print(">>> Loading FastText model (this may take a minute)...")
fasttext.util.download_model(LANG, if_exists='ignore')
ft = fasttext.load_model(f'cc.{LANG}.300.bin')

# 2. Parse target-distractor pairs from txt.txt
print(">>> Parsing target-distractor pairs...")
pairs = []

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(';')
        if len(parts) < 4:
            continue
        
        words = parts[2].split()
        dists = parts[3].split()
        
        for i in range(min(len(words), len(dists))):
            target = re.sub(r'[^\w]', '', words[i]).lower()
            distractor = re.sub(r'[^\w]', '', dists[i]).lower()
            
            # Skip placeholders and very short tokens
            if distractor.startswith('x') and ('-' in dists[i] or len(distractor) <= 1):
                continue
            if len(target) < 2 or len(distractor) < 2:
                continue
            
            pairs.append((target, distractor))

print(f"    Found {len(pairs)} target-distractor pairs.")

# 3. Compute cosine similarities
print(">>> Computing cosine similarities...")
similarities = []
for target, distractor in pairs:
    vec_t = ft.get_word_vector(target)
    vec_d = ft.get_word_vector(distractor)
    
    norm_t = np.linalg.norm(vec_t)
    norm_d = np.linalg.norm(vec_d)
    
    if norm_t > 0 and norm_d > 0:
        cos_sim = np.dot(vec_t, vec_d) / (norm_t * norm_d)
        similarities.append(cos_sim)

similarities = np.array(similarities)
mean_sim = np.mean(similarities)
median_sim = np.median(similarities)

print(f"    Mean similarity:   {mean_sim:.2f}")
print(f"    Median similarity: {median_sim:.2f}")
print(f"    Max similarity:    {np.max(similarities):.2f}")
print(f"    Min similarity:    {np.min(similarities):.2f}")
print(f"    Pairs above threshold ({SEMANTIC_THRESHOLD}): {np.sum(similarities > SEMANTIC_THRESHOLD)}/{len(similarities)}")

# 4. Plot histogram
fig, ax = plt.subplots(figsize=(12, 7))

# Histogram
n, bins, patches = ax.hist(
    similarities, 
    bins=40, 
    range=(min(-0.1, similarities.min()), max(1.0, similarities.max())),
    color='#4CAF50', 
    edgecolor='#2E7D32', 
    linewidth=0.5,
    alpha=0.9
)

# Mean line
ax.axvline(mean_sim, color='red', linestyle='--', linewidth=2, 
           label=f'Mean Similarity ({mean_sim:.2f})')

# Threshold line
ax.axvline(SEMANTIC_THRESHOLD, color='orange', linestyle=':', linewidth=2.5, 
           label=f'Semantic Filter Threshold ({SEMANTIC_THRESHOLD})')

# Labels and styling
ax.set_xlabel('FastText Cosine Similarity (1.0 = Synonyms, 0.0 = Unrelated)', fontsize=12)
ax.set_ylabel('Frequency (Number of Word Pairs)', fontsize=12)
ax.set_title('Distribution of Semantic Similarities (Targets vs. Distractors)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.set_xlim(-0.15, 1.05)

# Grid
ax.grid(axis='y', alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('semantic_similarity_histogram.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n>>> Histogram saved to: semantic_similarity_histogram.png")
