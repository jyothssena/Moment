# Experiment Tracking — Moment Compatibility Pipeline

## Tool
MLflow 3.10.1 — run locally, logged to `mlruns/`

---

## Run: threshold_0.82_topk_20

### Parameters

| Parameter | Value |
|---|---|
| similarity_threshold | 0.82 |
| faiss_top_k | 20 |
| embedding_model | hkunlp/instructor-xl |
| prompt_version | v1 |
| total_moments | 450 |
| total_users | 50 |
| candidate_pairs | 393 |

### Metrics

| Metric | Value |
|---|---|
| total_results | 393 |
| resonance_count | 218 |
| mirror_count | 148 |
| contradiction_count | 27 |
| no_match_count | 0 |
| resonance_pct | 55.47% |
| mirror_pct | 37.66% |
| contradiction_pct | 6.87% |
| avg_confidence | 0.8788 |
| avg_confidence_resonance | 0.8985 |
| avg_confidence_mirror | 0.8660 |
| avg_confidence_contradiction | 0.7904 |

---

## Results

### Match Type Distribution

![Match Type Distribution](images/match_type_distribution.png)

393 pairs were evaluated. Resonance is the dominant match type at 55.47%, 
followed by mirror at 37.66%, and contradiction at 6.87%. No pairs were 
classified as no_match, which reflects the aggressive pre-filtering at the 
FAISS retrieval layer — only pairs with cosine similarity above 0.82 reached 
the agent, meaning near-certain incompatible pairs were eliminated before 
the agent ran.

---

### Confidence Score Distribution by Match Type

![Confidence Distribution](images/confidence_distribution.png)

Resonance and mirror pairs are heavily concentrated in the 0.88–0.95 range, 
indicating the agent is highly certain when classifying aligned or opposing 
reader pairs. Contradiction pairs show a wider, lower spread — several fall 
below 0.80 — reflecting genuine agent uncertainty on structurally complex pairs.

---

### Confidence Score Range by Match Type

![Confidence Boxplot](images/confidence_boxplot.png)

The box plot confirms the pattern:

- Resonance: median ~0.90, tight range, two low outliers at 0.78 and 0.82
- Mirror: median ~0.85, wider range, whisker reaching down to 0.78
- Contradiction: median ~0.80, widest spread, one outlier as low as 0.72

This is the key bias detection finding — the agent systematically assigns 
lower confidence to contradiction pairs. This is expected behavior: 
contradiction matches require the agent to reason about productive tension 
between frameworks, which is a harder judgment than detecting alignment 
(resonance) or opposition (mirror).

---

## Findings and Model Selection

**Why threshold 0.82 was selected:**
An initial run with threshold 0.70 produced 1,123 candidate pairs — too many 
weak matches and prohibitively expensive to evaluate. Raising to 0.82 reduced 
the pool to 393 pairs with meaningfully higher average confidence (0.8788), 
while retaining a diverse match type distribution. Manual spot-checks confirmed 
the 0.82 pairs felt substantively connected.

**Key finding — no_match absence:**
The pipeline produced zero no_match verdicts. This is a direct consequence of 
the similarity pre-filter: by the time a pair reaches the agent, it has already 
cleared a high embedding similarity bar. Future work should evaluate whether 
relaxing the threshold to 0.75 surfaces genuine no_match cases, or whether 
a dedicated no_match calibration pass on borderline pairs is warranted.

**Key finding — contradiction confidence gap:**
avg_confidence_contradiction (0.7904) is notably lower than resonance (0.8985) 
and mirror (0.8660). This suggests contradiction is the hardest classification 
for the agent — a finding worth addressing in future prompt iterations by 
providing clearer criteria for what makes a contradiction match generative 
rather than simply incompatible.

![WhatsApp Image 2026-03-22 at 2 38 47 PM](https://github.com/user-attachments/assets/ef24f46e-0a95-4a97-a0bb-c2b323d75f36)

![WhatsApp Image 2026-03-22 at 2 39 06 PM](https://github.com/user-attachments/assets/594a49db-8d27-413f-b6e5-9c108add08a2)

<img width="800" height="500" alt="match_type_distribution" src="https://github.com/user-attachments/assets/c2fc1992-b366-4111-ac6b-8c3b5e0f1bf9" />

<img width="900" height="500" alt="confidence_distribution" src="https://github.com/user-attachments/assets/b5177485-e6f1-4a33-be69-a0b543ff70b4" />

<img width="800" height="500" alt="confidence_boxplot" src="https://github.com/user-attachments/assets/dadb5a70-33f5-4146-8992-1f2220332050" />

---

## Next Experiment

Ran different models sing different techniques and saved the results.
