# TinyAutoJEPA: Self-Supervised World Model Pilot

TinyAutoJEPA is an autonomous driving system based on Joint Embedding Predictive Architecture (JEPA). It focuses on learning the dynamics of an environment through self-supervised observation rather than direct imitation.

Unlike standard Imitation Learning, which clones human actions directly, this system builds a "World Model"—an internal representation of physics and causality. It learns to distinguish surface properties (e.g., grass vs. road) and predicts future states based on current actions before being tasked with driving.

## Hardware Configuration

This project is configured for a high-performance AMD workstation running Linux.

| Component | Specification | Function |
| :--- | :--- | :--- |
| **CPU** | **AMD Ryzen 9 5950X** (16C/32T) | Handles parallel data loading and image preprocessing. Utilizes multi-threading to ensure the GPU is constantly supplied with data, preventing bottlenecks during resizing operations. |
| **GPU 1** | **AMD Radeon RX 6950 XT** (16GB) | Primary compute unit for training. Leverages **ROCm** for hardware acceleration and Mixed Precision (`torch.amp`) training. The 16GB VRAM buffer accommodates large batch sizes (256+). |
| **GPU 2** | **9060 XT** | Secondary compute or display output unit. |
| **RAM** | **64 GB DDR4** | Facilitates OS-level caching of the dataset (approx. 1M frames), minimizing disk I/O latency during repeated training epochs. |
| **OS** | **Pop!_OS (Linux)** | Selected for native ROCm kernel support and efficient thread scheduling. |

### Estimated Training Times

| Component | Epochs | Duration per Epoch (Approx) | Description |
| :--- | :--- | :--- | :--- |
| **Encoder (Initial)** | 50 (Scratch) | ~52 mins | Base feature learning on initial random dataset. |
| **Encoder (Update)** | 20 (Fine-Tune) | ~50 mins | Adaptation to mixed dataset (Road + Grass). Computationally intensive due to dataset size. |
| **Predictor** | 30 | ~20 mins | Fast training. Optimization of latent vector dynamics. |
| **Decoder** | 20 | ~25 mins | Moderate intensity. Optimization of image reconstruction for visualization. |

## Installation & Setup

### 1. Environment Configuration
Ensure AMD ROCm drivers are correctly installed on the host system.

```bash
# Create Virtual Environment
python -m venv venv
source venv/bin/activate

# Install PyTorch for ROCm (Verify version match at pytorch.org)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/rocm6.0](https://download.pytorch.org/whl/rocm6.0)

# Install Project Dependencies
pip install gymnasium[box2d] opencv-python numpy tqdm matplotlib

```

### 2. Gymnasium Versioning

This project utilizes `CarRacing-v3`. Ensure the library is up to date to prevent physics engine deprecation errors.

```bash
pip install --upgrade gymnasium swig

```

## Training Pipeline

The training strategy employs a balanced dataset consisting of 50% random exploration data (off-road recovery) and 50% expert driving data (lane keeping).

### Step 1: Data Generation

Generates expert trajectory data using an algorithmic solver.

```bash
python collect_race_data.py
# Output: ./data_race/*.npz (~20k frames)

```

### Step 2: Encoder Training

Fine-tunes the visual encoder. This step resumes from previous checkpoints to retain basic feature recognition while adapting to specific road features found in the new dataset.

```bash
python train_encoder.py

```

### Step 3: Predictor Training

Trains the latent dynamics model. This process starts from scratch to ensure the model learns a cohesive physics representation of both asphalt and grass surfaces without bias from previous iterations.

```bash
python train_predictor.py

```

### Step 4: Decoder Training

Trains the reconstruction model. This component is required only for human verification of the latent state representation.

```bash
python train_decoder.py

```

### Step 5: Visualization

Generates a batch of video files comparing the model's predicted future trajectories against the ground truth.

```bash
python visualize_batch.py
# Output: dream_batch_*.avi

```

## Component Architecture

### 1. The Encoder (`TinyEncoder`)

**Layman Explanation:**
The Encoder functions as the system's vision. It compresses raw, high-dimensional image data (pixels) into a compact summary vector. This vector allows the system to process the scene's content (e.g., "curve ahead") without processing every individual pixel.

**Technical Detail:**
The architecture is a modified **ResNet18** backbone. Max-pooling layers have been removed to preserve spatial granularity essential for the 64x64 input resolution. Training utilizes **VICReg** (Variance-Invariance-Covariance Regularization), a self-supervised loss function that prevents mode collapse and ensures high-quality feature representations without requiring negative sample pairs.

### 2. The Predictor (`Predictor`)

**Layman Explanation:**
The Predictor functions as the system's internal simulator. It takes the current state summary and a proposed action (e.g., "steer left") to calculate the likely outcome. It learns physics rules, such as the relationship between surface friction and acceleration.

**Technical Detail:**
This is a 3-layer **Multi-Layer Perceptron (MLP)**.

* **Input:** Concatenation of the Latent State () and Action vector ().
* **Output:** Predicted Next Latent State ().
* **Objective:** Minimize the Mean Squared Error (MSE) between the predicted next state vector and the actual encoded next state.

### 3. The Decoder (`TinyDecoder`)

**Layman Explanation:**
The Decoder translates the internal numerical summaries back into visible images. This allows developers to visually inspect the system's "imagination" and verify accuracy.

**Technical Detail:**
The architecture is a **Transposed Convolutional Network** (Inverse ResNet). It projects the 512-dimensional latent vector into a 4x4 spatial feature map and progressively upsamples it to the original 64x64 resolution using learned convolutional filters.

## Future Implementation (Model Predictive Control)

The current system is a passive observer. The next phase involves implementing Model Predictive Control (MPC) to enable active driving.

**MPC Workflow:**

1. **Simulation:** The model generates multiple random action sequences.
2. **Prediction:** The Predictor estimates the future state for each sequence.
3. **Evaluation:** A cost function scores the predicted states (e.g., higher score for high speed and low distance from the track center).
4. **Execution:** The action sequence with the optimal score is executed.
