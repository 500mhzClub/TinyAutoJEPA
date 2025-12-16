# TinyJEPA: A Minimal World Model 

**TinyJEPA** is a practical implementation of a **Joint Embedding Predictive Architecture (JEPA)**.

Unlike standard AI that predicts the next *word*, this project builds a World Model that learns the physics of a driving environment (`CarRacing-v2`) purely from visual observation. It uses **VICReg** regularization to prevent representation collapse and **Latent Dynamics** to imagine future states.

The end goal is a system that can "dream" driving trajectories and distinguish between **plausible futures** (low energy) and **impossible futures** (high energy).

---

## Architecture & Hardware Strategy

This project is optimized for a dual-machine workflow to separate CPU-bound simulation from GPU-bound training.

| Role | Machine Spec | Responsibility |
| :--- | :--- | :--- |
| **The Factory** | **AMD Ryzen 5950X** + RX 6950 XT | **Data Generation.** Utilizes 32 threads to run 16+ parallel Box2D physics simulations significantly faster than a single core. |
| **The Brain** | AMD Ryzen 3600X + **RX 9060 XT** | **Training.** Utilizes RDNA 4's Matrix Cores and native `bfloat16` support for efficient Transformer/MLP training. |

---

## Prerequisites

### Machine A (The Factory)
* Python 3.10+
* `pip install gymnasium[box2d] numpy opencv-python`

### Machine B (The Brain)
* Python 3.10+ (Linux Recommended for ROCm)
* **ROCm 6.2+** (Required for RDNA 4 support)
* `pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2`
* `pip install gymnasium[box2d] matplotlib scikit-learn`

---

## The 4-Day Plan

### Day 1: The "Multiverse" (Data Collection)
**Objective:** Create a "Memory Bank" of physics.
Before the AI can predict physics, it must observe cause and effect. We collect **Triplets**: `(State_t, Action_t, State_t+1)`.

* **Script:** `collect_data.py`
* **Task:** Run 16 parallel agents on the 5950X.
* **Target:** 50,000 - 100,000 frames.
* **Output:** Compressed `.npz` or `.pt` files containing trajectory chunks.
* **Key Concept:** **Experience Replay** — Storing past interactions to learn offline.

### Day 2: The "Eye" (VICReg Encoder)
**Objective:** Learn to "see" without labels.
We train a **ResNet-18** to compress 64x64 images into 512-dim embeddings. To prevent the model from "cheating" (outputting all zeros), we use **VICReg Loss**.

* **Script:** `train_encoder.py`
* **Loss Function:**
    1.  **Variance:** Forces embeddings to differ across the batch (prevents collapse).
    2.  **Invariance:** Forces two augmented views of the same image to match.
    3.  **Covariance:** Forces embedding dimensions to be independent (maximizes information).
* **Hardware Tip:** Use `torch.bfloat16` on the RX 9060 XT for 2x speedup.

### Day 3: The "Brain" (Latent Predictor)
**Objective:** Learn the physics of the latent space.
We freeze the Encoder and train a Predictor Network (MLP) to model the flow of time.
$$P(z_t, action_t) \approx z_{t+1}$$

* **Script:** `train_predictor.py`
* **Architecture:** 3-Layer MLP or small Transformer Block.
* **Task:** Minimize MSE between the *Predicted Future Embedding* and the *Actual Future Embedding*.
* **Optional:** Train a `Decoder` to visualize the "dreamt" embeddings as blurry images.

### Day 4: The "Judge" (Energy & Inference)
**Objective:** Detect impossible futures.
We use the trained JEPA to evaluate "Energy" (Compatibility).
* **Real Future:** Low Distance (Energy) between Prediction and Proposal.
* **Fake Future:** High Distance (Energy).

* **Script:** `demo_energy.py`
* **The Demo:** A live window showing the car driving, with a real-time graph plotting the Energy of the current state vs. a random/teleporting state.

---

## Repository Structure

```text
/tiny_jepa
├── /data                # Raw .npz or .pt files (GitIgnored)
├── /models              # Saved .pth weights
├── buffer.py            # Data collection logic & Dataset class
├── networks.py          # ResNet Encoder, MLP Predictor, Decoder
├── vicreg.py            # The VICReg Loss Function implementation
├── collect_data.py      # (Day 1) Multiprocessing generation script
├── train_encoder.py     # (Day 2) Self-Supervised training
├── train_predictor.py   # (Day 3) Latent Dynamics training
└── demo_energy.py       # (Day 4) The final visualization

```

## Theory & ReferencesThis project is built on concepts from:

* **VICReg: Variance-Invariance-Covariance Regularization** (Bardes, Ponce, LeCun, 2021)
* [Paper](https://arxiv.org/abs/2105.04906) | [Meta AI Blog](https://www.google.com/search?q=https://ai.meta.com/blog/vicreg-variance-invariance-covariance-regularization-for-self-supervised-learning/)


* **JEPA: Joint Embedding Predictive Architecture** (LeCun, 2022)
* [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf)


* **World Models** (Ha, Schmidhuber, 2018)
* [Interactive Paper](https://worldmodels.github.io/)



---

## Quick Start1. **Generate Data (Machine A):**
```bash
python collect_data.py
# Copy /data folder to Machine B

```


2. **Train Encoder (Machine B):**
```bash
python train_encoder.py --epochs 50 --batch_size 256

```


3. **Train Predictor (Machine B):**
```bash
python train_predictor.py --epochs 50

```


4. **Run Demo (Machine B):**
```bash
python demo_energy.py

```
