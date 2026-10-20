# CLIP Image Classification for Waste

Code repository for the thesis work of Lionello Penoni and Pedro Lerma. It evaluates zero-shot and few-shot image classification on the Garbage Classification v2 dataset using CLIP ViT-B/32 (`openai/clip-vit-base-patch32`).

## Repository Structure

| File | Purpose |
|---|---|
| `CLIP_ZeroShot.ipynb` | Zero-shot classification: plain labels, prompt templates, averaged prompt embeddings, analysis of the "trash" class |
| `CLIP_BestK_Trash_Prompting.ipynb` | Candidate prompt phrases for the "trash" class, scored by cosine similarity |
| `CLIP_LinearProbe.ipynb` | Few-shot linear probe with a k-shot sweep |
| `CLIP_MLmodels.ipynb` | Classical ML classifiers on CLIP image embeddings, with and without PCA |
| `CLIP_Adapter.ipynb` | Neural-network adapter on CLIP embeddings |
| `utils.py` | Shared helpers: evaluation, confusion matrix, embedding extraction, k-shot sampling |
| `few-shot_results/` | Stored results of the k-shot sweep (`few_shot_results.pkl`, `trained_probes.pkl`) |

## Setup

Python 3.12 is recommended.

```bash
git clone https://github.com/Nello-pn/trashCLIP.git
cd trashCLIP
pip install -r requirements.txt
```

The dataset (`sumn2u/garbage-classification-v2`) and the CLIP weights are downloaded automatically on first run. A GPU is optional; `utils.get_device()` selects CUDA, MPS (Apple Silicon) or CPU.

## Usage

- Each notebook is self-contained and can be run independently, top to bottom.
- Data split: stratified 80/20 train/test with seed 42.
- In `CLIP_LinearProbe.ipynb`, the long k-shot sweep is skipped by default and the stored results in `few-shot_results/` are loaded instead. To recompute it, set `RUN_EXPENSIVE_CELL = True`.