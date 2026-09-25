# ORTHOWORLDS: Multi-View Orthographic World Models

**ORTHOWORLDS** is a framework for **multi-view orthographic world modeling**. 

Conventional monocular game simulators typically only simulate world environments from a single first-person perspective. In contrast, **ORTHOWORLDS** simultaneously generates 4 geometrically coupled camera views (**Top-Down, Side, Rear, and 3D / First-Person**) with multi-player text prompt steering and 3D cross-view consistency.

---

## Key Research Contributions

1. **Beyond Monocular World Models**: 
   Single-camera world models suffer from partial observability, blind spots, and hallucination drift. ORTHOWORLDS couples orthographic projections (providing global spatial coordinate truth) with perspective views (providing local visual rendering).
2. **Orthographic Conditional Flow Matching**:
   Instead of fragile joint sampling that suffers from multi-modal divergence (e.g., the top view turns left while the 3D view turns right), ORTHOWORLDS formulates world generation as an **Orthographic Markov Random Field (MRF)**. By Brook's Lemma and conditional score decomposition, training with random view masking trains a single network to evaluate all full conditionals.

   $$P(\text{Top}_{t+1} \mid \text{Side}_{t+1}, \text{Rear}_{t+1}, \text{3D}_{t+1}, \mathbf{a}_t, x_t)$$

   $$P(\text{Side}_{t+1} \mid \text{Top}_{t+1}, \text{Rear}_{t+1}, \text{3D}_{t+1}, \mathbf{a}_t, x_t)$$

   $$P(\text{Rear}_{t+1} \mid \text{Top}_{t+1}, \text{Side}_{t+1}, \text{3D}_{t+1}, \mathbf{a}_t, x_t)$$

   $$P(\text{3D}_{t+1} \mid \text{Top}_{t+1}, \text{Side}_{t+1}, \text{Rear}_{t+1}, \mathbf{a}_t, x_t)$$
   
3. **Cross-View Geometric Consensus**:
   Allows inference via either:
   * **Anchor-Chain Generation**: Sample the global top-down anchor first, then conditionally sample side, rear, and 3D views with zero geometric ambiguity.
   * **Cyclic Gibbs Refinement**: Iteratively refine noisy views to guaranteed fixed-point physical consensus.

---

## 🎮 Supported Environments

ORTHOWORLDS provides synchronized 4-camera wrappers across rich physical and game environments:

| Environment | Description | Modality |
| :--- | :--- | :--- |
| **Drone Dogfight** 🚁 | Aerial combat simulation with 3D flight dynamics | Multi-Agent / 3D |
| **Mario Escape** 🍄 | Side-scrolling platformer with orthographic depth | Discrete / Physics |
| **MultiCar Racing** 🏎️ | Competitive top-down and cockpit racing simulation | Continuous / Multi-Player |
| **Excavator** 🚜 | Heavy machinery kinematics with articulate boom control | Multi-Joint Robotics |
| **Wrecking Ball** 🏗️ | Demolition physics with coupled cable-pendulum dynamics | Rigid-Body Physics |
| **Bipedal Walker** 🤖 | 4-joint biped locomotion with terrain obstacles | Continuous Control |
| **Lunar Lander** 🚀 | Rocket descent with fuel dynamics and gimbal thrusters | Space Physics |

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/dexterdley/ORTHOWORLDS.git
cd ORTHOWORLDS

pip install -r requirements.txt
pip install -U diffusers transformers accelerate peft torch torchvision opencv-python gradio
```

---

## 🏋️ Training

### Fine-Tuning DiT with LoRA (Recommended)

Trains orthographic cross-view flow matching with text prompt conditioning:

```bash
python ortho/train_ortho_wan.py \
    --model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --batch_size 2 \
    --lora_rank 16 \
    --epochs 10 \
    --steps_per_epoch 100 \
    --eval_every 1 \
    --eval_steps 15 \
    --lr 1e-4
```