# LoRA-Hash

**Unveiling Model Identity from Heavy-Tailed Weight Distributions** — ICME 2026.

LoRA-Hash is an **unsupervised perceptual hash** for LoRA (low-rank adaptation)
checkpoints. Given a `.safetensors` adapter it produces a short binary
fingerprint that is stable under benign reparameterisation and mild
perturbation, yet different for genuinely different models. No training, no
reference model and no trigger-word prompt are required — which is what makes it
usable for model plagiarism detection and identity verification.

```python
from lora_hash import LoraPerceptualHasher

hasher = LoraPerceptualHasher(device="cuda")     # 256-bit fingerprint
h_ref = hasher.generate_hash("benign.safetensors")
h_sus = hasher.generate_hash("suspect.safetensors")

print(hasher.calculate_hamming_distance(h_ref, h_sus))   # e.g. 0.0352
```

---

## Method

A LoRA layer stores two factors `A` (`lora_down`) and `B` (`lora_up`) whose
product is the weight update `ΔW = B·A`. LoRA-Hash turns the whole adapter into a
fingerprint in four steps:

1. **Weight reconstruction.** For every layer, materialise `ΔW = B·A` from the
   low-rank factors. Working on `ΔW` (rather than on the raw factors) makes the
   hash invariant to the rank decomposition itself.
2. **Per-layer Z-score standardisation.** Each `ΔW` is standardised to zero mean
   and unit variance. LoRA updates are extremely **heavy-tailed**: the vast
   majority of entries sit in a "dead zone" around zero while a tiny fraction of
   extreme entries carries essentially all of the semantic content. Standardising
   per layer removes the influence of the `alpha` scaling factor and stops one
   large-magnitude layer from dominating the fingerprint.
   *(Removing this step is the `lora-nonorm` ablation — see below.)*
3. **Segmented extremum extraction.** The concatenated standardised vector is cut
   into `hash_length + 1` contiguous segments. Because mean and standard
   deviation of a heavy-tailed distribution are dominated by the bulk, they are
   noise-sensitive; the extrema are not. Each segment is therefore summarised by
   its **maximum** *and* its **minimum**.
4. **dHash encoding.** Each channel is binarised by comparing neighbouring
   segment statistics (`1` if a statistic exceeds its successor, else `0`),
   which is robust to global amplitude changes. Concatenating the two channels
   yields a `2 × hash_length` = **256-bit** fingerprint.

### Why the extrema?

The repository includes the two motivation experiments behind step 3:

* `scripts/energy_analysis.py` reports how much of the update's L1 energy lives in
  the extreme tail versus in the bulk.
* `scripts/attacks/destructive_ablation.py` produces LoRAs with the top-*x*%,
  bottom-*x*% or a random *x*% of weights zeroed. Measuring LPIPS for each shows
  that zeroing a fraction of a percent of the **largest** weights collapses the
  model's behaviour, while zeroing a much larger fraction of the **smallest**
  weights is visually imperceptible.

---

## Installation

```bash
git clone https://github.com/xaddwell/LoRA-Hash
cd LoRA-Hash
pip install -r requirements.txt
```

The core hasher needs only `torch`, `safetensors` and `numpy`; `diffusers`,
`lpips` and `scikit-image` are required only for the perceptual ground-truth
pipeline. A CUDA GPU is recommended but everything also runs on CPU.

Verify the install:

```bash
python -c "from lora_hash import LoraPerceptualHasher; print(LoraPerceptualHasher().hash_length)"
python tests/test_hasher.py    # dataset-free smoke tests
```

---

## Repository layout

```
lora_hash/
    hasher.py                 LoRA-Hash itself (the method)
    common.py                 ΔW reconstruction, dHash, Hamming distance
    baselines/
        ablations.py          mean / std / max / min / |max| channel variants
        simhash.py            SimHash (random-hyperplane LSH)
        nts.py                normality-test-statistic hashing
        hos.py                higher-order-moment hashing
        liu_cvpr2025.py       supervised projection network (needs a checkpoint)
        crypto.py             SHA-256 / MD5 file digests
scripts/
    hash_variants.py          hash every reference LoRA + its variants  -> JSON
    measure_lpips_ssim.py     perceptual ground truth (SSIM / LPIPS)    -> JSON
    attack_breakdown.py       precision / recall / F1 against ground truth
    energy_analysis.py        heavy-tail energy statistics
    time_cost.py              latency and peak GPU memory per fingerprint
    attacks/                  how the tampered LoRA variants are generated
configs/
    lora_metadata.py          trigger words and prompt templates per reference LoRA
tests/
    test_hasher.py            dataset-free smoke tests
```

---

## Reproducing the evaluation

The pipeline needs a directory of **reference** LoRAs and, for each of them, a
set of **tampered copies** produced by `scripts/attacks/`. Expected layout:

```
<lora-root>/
    benign_SD1.5/      reference adapters, one per model
    noise/             <name>_st_0_05.safetensors      ...
    prune/             <name>_pruned_30pct.safetensors ...
    flip/              <name>_flipped.safetensors
    ablation/          <name>_ABLA_top_0.5pct.safetensors
```

**1. Generate the tampered variants** (reuse the reference set for the others):

```bash
python scripts/attacks/noise_injection.py   --input data/benign_SD1.5 --output data/noise --batch
python scripts/attacks/magnitude_pruning.py --input data/benign_SD1.5 --output data/prune --batch
python scripts/attacks/sign_flip.py         --input data/benign_SD1.5 --output-dir data/flip
python scripts/attacks/rank_reduction.py    --input data/benign_SD1.5 --output data/rank \
        --kohya-script /path/to/kohya_ss/library/resize_lora.py
```

**2. Build the perceptual ground truth.** LPIPS between images sampled from the
reference and the tampered adapter, with identical prompts and seeds:

```bash
for name in $(ls data/benign_SD1.5 | sed 's/.safetensors//'); do
    python scripts/measure_lpips_ssim.py --lora-name "$name" \
        --lora-files-root data --benign-subdir benign_SD1.5 \
        --output-dir results/lpips_ssim_similarity
done
```

**3. Hash every pair and record Hamming distances:**

```bash
python scripts/hash_variants.py --lora-root data --benign-dir data/benign_SD1.5 \
    --hasher lora --output-dir results/hanming_distance/lorahash
```

**4. Score the detector:**

```bash
python scripts/attack_breakdown.py \
    --lpips-dir results/lpips_ssim_similarity \
    --dist-dir results/hanming_distance/lorahash \
    --threshold 0.1 --lpips 0.30 --sweep
```

A tampered adapter is treated as a **copy** when `LPIPS ≤ 0.30` (it still behaves
like the original) and is **flagged** when its Hamming distance is below
`--threshold`.

### Reference results

Reproduced with this release on the SD 1.5 subset (27 reference LoRAs, 286
evaluated pairs, decision threshold 0.1, LPIPS boundary 0.30):

| Variant | Precision | Recall | F1 | False-alarm rate |
|---|---|---|---|---|
| **LoRA-Hash** (Z-score) | **91.09 %** | 83.64 % | **0.8720** | **27.3 %** |
| LoRA-Hash w/o Z-score (`lora-nonorm`) | 85.84 % | 88.18 % | 0.8700 | 48.5 % |

Sweeping the threshold, LoRA-Hash reaches **F1 = 0.8978 at threshold 0.138**,
against 0.8902 at 0.236 without standardisation. Standardising per layer is
therefore what makes a *low* decision threshold usable: it roughly halves the
false-alarm rate at comparable recall. Numbers from the full benchmark are
reported in the paper.

### Cost

```bash
python scripts/time_cost.py --hasher lora --lora-path data/benign_SD1.5/<name>.safetensors
```

---

## Baselines and ablations

Every hasher exposes the same three-call interface, so the loop above works
unchanged with `--hasher sim|nts|hos|cvpr|mean|max|min|abs|std|sha256|md5`.

| `--hasher` | Method |
|---|---|
| `lora` | **LoRA-Hash** (this work) |
| `lora-nonorm` | ablation: no per-layer Z-score standardisation |
| `mean`, `std` | ablation: bulk statistic instead of extrema |
| `max`, `min`, `abs` | ablation: a single channel instead of max+min |
| `sim` | SimHash random-hyperplane LSH |
| `nts` | Shapiro-Wilk normality statistic per segment |
| `hos` | skewness + kurtosis per segment |
| `cvpr` | supervised projection network (Liu et al., CVPR 2025) |
| `sha256`, `md5` | cryptographic file digests |

> **Note on `cvpr`:** the trained checkpoint
> (`cvpr_baseline_trained.pth`, ~850 MB) is **not** distributed here. Without it
> the network is randomly initialised and its numbers are meaningless. Either
> train it yourself or drop the checkpoint in the working directory.

---

## Scope and limitations

* Checkpoints must use the kohya/`diffusers` LoRA key convention
  (`*.lora_down.weight` / `*.lora_up.weight`). Adapters stored with PEFT's
  `lora_A`/`lora_B` naming are not currently recognised.
* LoRA-Hash is a *perceptual* hash: it is designed to be robust to benign
  reparameterisation and to tolerate mild modification, and it is therefore
  deliberately not collision-resistant in the cryptographic sense. Two unrelated
  models can, on rare occasions, land within the threshold.
* The fingerprint is derived from weights only; it says nothing about the
  provenance of the *data* a model was trained on.

## Citation

```bibtex
@inproceedings{lorahash2026,
  title     = {LoRA-Hash: Unveiling Model Identity from Heavy-Tailed Weight Distributions},
  author    = {He, Xing and Chen, Jiahao and Li, Junhao and Feng, Zhou and Ji, Shouling},
  booktitle = {IEEE International Conference on Multimedia and Expo (ICME)},
  year      = {2026}
}
```

## License

Released for research use. Please contact the authors before redistributing for
commercial purposes.
