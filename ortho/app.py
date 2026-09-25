import os
import sys
import gradio as gr
import spaces

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Map environment names to their corresponding .gif files in the same folder
GIF_MAP = {
    "Drone Dogfight 🚁": "./assets/Drone_Dogfight_grid.gif",
    "Mario Escape 🍄": "./assets/Mario_Escape_grid.gif",
    "Wrecking Ball 🏗️": "./assets/Wrecking_Ball_grid.gif",
    "Excavator 🚜": "./assets/Excavator_grid.gif",
    "Bipedal Walker 🤖": "./assets/Bipedal_Walker_grid.gif",
    "Lunar Lander 🚀": "./assets/Lunar_Lander_grid.gif",
}


@spaces.GPU
def load_gif(env_name):
    filename = GIF_MAP.get(env_name, "./assets/Drone_Dogfight_grid.gif")
    filepath = os.path.join(CURRENT_DIR, filename)
    if os.path.exists(filepath):
        return filepath
    return None



def load_description_text():
    desc_path = os.path.join(CURRENT_DIR, "game_envs", "game_envs_descriptions.md")
    if os.path.exists(desc_path):
        with open(desc_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Environment description file not found."


# ==============================================================================
# GRADIO UI & CUSTOM STYLING
# ==============================================================================
custom_css = """
:root {
    color-scheme: dark !important;
    --body-background-fill: #0b0f19 !important;
    --body-text-color: #ffffff !important;
    --body-text-color-subdued: #f1f5f9 !important;
    --background-fill-primary: #0b0f19 !important;
    --background-fill-secondary: #151b28 !important;
    --neutral-950: #0b0f19 !important;
    --neutral-900: #151b28 !important;
    --neutral-800: #1e2638 !important;
    --neutral-700: #2d3748 !important;
    --neutral-600: #f8fafc !important;
    --neutral-500: #ffffff !important;
    --block-background-fill: #151b28 !important;
    --block-label-text-color: #ffffff !important;
    --checkbox-label-text-color: #ffffff !important;
    --radio-label-text-color: #ffffff !important;
}

/* ENTIRE SCREEN DARK MODE BACKGROUND */
html, body, #root, .gradio-container, .gradio-app, .main, .app, 
header, footer, nav, section, main, iframe, .tabs, .tabitem, .tab-nav,
.panel, .form, .block, .block-container, div[data-testid="block"] {
    background-color: #0b0f19 !important;
    color: #ffffff !important;
}

body, .gradio-container, .dark,
p, span, div, h1, h2, h3, h4, h5, h6, li, label, strong, em, b, i,
.prose, .prose *, .markdown, .markdown *,
.tabs button, .tabitem, .label, .group_title,
span.text, span.label-text, font {
    color: #ffffff !important;
}

.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto;
    background-color: #0b0f19 !important;
}

/* CHECKBOX DARK BACKGROUND & WHITE TEXT */
label.block, label.checkbox, div[data-testid="checkbox"], .gr-checkbox, label:has(input[type="checkbox"]) {
    background-color: #151b28 !important;
    border: 1px solid #2d3748 !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    margin-bottom: 8px !important;
    color: #ffffff !important;
}

label.block span, label.checkbox span, div[data-testid="checkbox"] span, label:has(input[type="checkbox"]) span {
    color: #ffffff !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
}

input[type="checkbox"] {
    accent-color: #38bdf8 !important;
    width: 18px !important;
    height: 18px !important;
    cursor: pointer !important;
}

.hero-card {
    background: linear-gradient(135deg, #1e2638 0%, #0d121d 100%);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 16px;
    padding: 36px;
    margin-bottom: 28px;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.5);
    text-align: center;
}

.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 12px;
}

.hero-subtitle {
    font-size: 1.2rem;
    color: #cbd5e1 !important;
    margin-bottom: 22px;
    line-height: 1.6;
}

.badge-container {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
}

.badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.88rem;
    font-weight: 600;
    background: rgba(56, 189, 248, 0.2);
    color: #38bdf8 !important;
    border: 1px solid rgba(56, 189, 248, 0.4);
}

.section-card {
    background: #151b28 !important;
    border: 1px solid #2d3748 !important;
    border-radius: 16px;
    padding: 30px;
    margin-bottom: 28px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
}

/* Strictly 3x2 Grid (3 Columns x 2 Rows) for Radio options */
#env-radio fieldset, #env-radio .radio-group, #env-radio div.wrap, #env-radio > div, #env-radio form {
    display: grid !important;
    grid-template-columns: repeat(3, 1fr) !important;
    gap: 12px !important;
}
#env-radio label {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 12px 16px !important;
    background: #1e2638 !important;
    border: 1px solid #2d3748 !important;
    border-radius: 10px !important;
    margin: 0 !important;
    cursor: pointer !important;
    text-align: center !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    color: #ffffff !important;
}
#env-radio label.selected, #env-radio label:has(input:checked) {
    background: rgba(56, 189, 248, 0.25) !important;
    border-color: #38bdf8 !important;
    color: #38bdf8 !important;
}
#env-radio label.selected *, #env-radio label:has(input:checked) * {
    color: #38bdf8 !important;
}
"""

with gr.Blocks(
    css=custom_css, 
    theme=gr.themes.Default(), 
    head="<style>html, body, #root, .gradio-container { background-color: #0b0f19 !important; color: #ffffff !important; }</style>",
    js="() => { document.body.classList.add('dark'); document.documentElement.classList.add('dark'); }", 
    title="Orthographic World Models"
) as demo:
    
    with gr.Tabs():
        
        # --- TAB 1: MAIN PAGE ---
        with gr.TabItem("🏠 Main"):
            
            # 1. HERO HEADER
            with gr.Column(elem_classes=["hero-card"]):
                gr.Markdown(
                    """
                    <div class="hero-title">Orthographic World Models</div>
                    <div class="hero-subtitle">Diffusion Generative Neural Game Engines with Multi-player, Multi-View & Orthographic Representations</div>
                    <div class="badge-container">
                        <span class="badge">Diffusion Generative World Models</span>
                        <span class="badge">Multi-View Physics Simulator</span>
                        <span class="badge">Reinforcement Learning</span>
                    </div>
                    """
                )

            # 2. INTRODUCTION & MOTIVATION
            with gr.Column(elem_classes=["section-card"]):
                gr.Markdown(
                    r"""
                    # 📖 Introduction & Motivation
                    
                    ### 🌐 What is an Orthographic World Model?
                    
                    **Orthographic World Models (OWMs)** are generative neural game engines and world models trained on synchronized **multi-perspective and orthographic representations** of physical game environments.
                    
                    Unlike conventional single-perspective world models (e.g. FPV-only or static camera models), OWMs decompose complex 3D environments into synchronized orthogonal projections (**Top-Down Plan View, Orthographic Front View, Orthographic Side View**) alongside first-person perspective (**FPV**) streams.
                    
                    ---
                    
                    ### 💡 Motivation: Why Are Multi-Views Needed?
                    
                    #### 1. 🙈 Disambiguating Visual Occlusions & Frustum Loss
                    In single-perspective first-person or third-person views, critical game state objects frequently drift out-of-frame or become occluded by obstacles. 
                    - **Top-Down Plan View** ($X-Y$) maintains an unclipped global topological map of the entire arena.
                    - **Orthographic Elevation Views** ($X-Z$ & $Y-Z$) preserve height distributions, jumping arcs, and vertical hazards.
                    
                    #### 2. 📐 Metric Scale & Linear Spatial Kinematics
                    Perspective projection applies non-linear perspective division ($x' = f \cdot X / Z$), distorting object sizes, velocities, and distance perception as objects move relative to the camera focal length.
                    - Orthographic projections preserve **exact linear scale everywhere in the frame**.
                    - Physical velocities ($\Delta x / \Delta t$) and bounding box collision volumes remain **invariant to distance**, providing far cleaner spatial gradients for neural policy networks and latent generative diffusion models.
                    
                    #### 3. 🤝 Multi-Agent & Cross-View Consistency
                    In multi-agent environments (e.g., *Drone Dogfight*, *MultiCar Racing*), individual agents operate with local egocentric FPV views. Orthographic multi-views provide a canonical global coordinate space that allows world models to maintain unified cross-view physics consistency and prevent hallucinations during long-rollout neural simulation.
                    """
                )

            # 3. SYNCHRONIZED 4-VIEW ROLLOUT SHOWCASE
            with gr.Column(elem_classes=["section-card"]):
                gr.Markdown(
                    """
                    # 🎥 Synchronized 4-View Environment Rollouts
                    Select an environment below to view its synchronized **Multi-View Rollout Animation** (Top-Down Plan | 3D FPV | Orthographic Rear | Orthographic Right Side).
                    """
                )
                
                env_select = gr.Radio(
                    choices=list(GIF_MAP.keys()),
                    value="Drone Dogfight 🚁",
                    label="Select Environment",
                    elem_id="env-radio",
                )
                
                gr.Markdown("---")
                
                gif_player = gr.Image(
                    value=load_gif("Drone Dogfight 🚁"),
                    label="Orthographic Multi-View Environments",
                    type="filepath",
                    interactive=False,
                )

                env_select.change(
                    fn=load_gif,
                    inputs=[env_select],
                    outputs=[gif_player],
                )

            # 4. MULTI-AGENT & MULTI-VIEW NEURAL SIMULATION
            with gr.Column(elem_classes=["section-card"]):
                gr.Markdown(
                    """
                    # 🏎️ Multi-Agent & Multi-View Game Engine
                    
                    Orthographic World Models naturally scale to **multi-agent competitive and cooperative scenarios**, where multiple autonomous entities interact simultaneously in shared 3D space.
                    
                    ### Key Multi-Agent Capabilities:
                    ✅ **Joint Egocentric & Allocentric Perception**: Combines egocentric FPV perspectives with global allocentric orthographic channels so agents can reason about opponent positions both locally and globally.
                    
                    ✅ **Cross-Agent Spatial Physics Grounding**: Maintains accurate collision boundaries, overtaking trajectories, and slipstream physics across multi-car racing and aerial drone combat.
                    
                    ✅ **Synchronized Generative Rollouts**: Enables diffusion world models to roll out multi-agent state predictions across all camera channels concurrently without inter-view divergence.
                    
                    ✅ **Minimap & Localisation Support**: Provides real-time top-down plan views that serve as active in-game minimaps for tactical planning, localisation & multi-agent navigation.
                    """
                )
                
                multi_agent_gif_path = os.path.join(CURRENT_DIR, "./assets/multicar_dual_agent_run.gif")
                if os.path.exists(multi_agent_gif_path):
                    gr.Image(
                        value=multi_agent_gif_path,
                        label="MultiCarRacing — Multi-View Rollout (CarRacing Re-imagined)",
                        type="filepath",
                        interactive=False,
                    )
            
            # 5. DIFFUSION GAME ENGINES & MULTI-VIEW WORLD MODELS
            with gr.Column(elem_classes=["section-card"]):
                gr.Markdown(
                    r"""
                    # 🌍 Diffusion Game Engines & Multi-View World Models
                    
                    Traditional generative world models typically rely on a single viewpoint, which often leads to issues with occlusion, poor spatial reasoning, and "hallucinated" physics. **Orthographic Multi-View World Models** are proposed to overcome these limitations by forcing the underlying AI to learn and maintain the true 3D geometry of its environment. 

                    By leveraging **Latent Diffusion Models (LDMs)** to function as interactive Generative Game Engines, this architecture performs **joint multi-view diffusion** across four distinct perspectives.
                    
                    ### Key Benefits & Objectives:
                                        
                    1. **Enforcing 3D Geometric Consistency**: Utilizing cross-view spatial attention mechanisms ensures the different camera angles never contradict each other. This forces the diffusion model to learn robust, real-world physics and accurate 3D structures, rather than superficial 2D pixel approximations.
                    
                    2. **Stable Long-Horizon Simulation**: Conditioned on historical action sequences ($a_{t-k:t}$) and previous frame memories ($x_{t-1}$), the multi-view geometric constraints act as an anchor. This drastically reduces the autoregressive drift and visual degradation usually seen in generative engines over time.
                    
                    3. **A Unified Neural Physics Engine**: The ultimate purpose is to bypass traditional rendering and physics pipelines entirely. This approach directly synthesizes complex spatial collisions, terrain modifications, and dynamic interactions purely from learned neural representations, requiring absolutely no hardcoded rules.
                    """
                )
                
                diffusion_gif_path = os.path.join(CURRENT_DIR, "./assets/diffusion_training_progression.gif")
                if os.path.exists(diffusion_gif_path):
                    gr.Image(
                        value=diffusion_gif_path,
                        type="filepath",
                        interactive=False,
                    )

            # 6. SCALING LAWS & ENVIRONMENT DIVERSITY
            with gr.Column(elem_classes=["section-card"]):
                gr.Markdown(
                    r"""
                    # 📈 Scaling Law Experiments & Game Environment Diversity [TBD]
                    
                    A fundamental question in generative world modeling is how multi-view neural game engines scale as the diversity of training environments increases.
                    
                    ### Key Empirical Observations:
                    - **Log-Linear Fidelity Gains**: Visual reconstruction quality (PSNR in dB) improves steadily as the model is trained across a wider spectrum of environment mechanics, approaching a diminishing returns frontier around 50 games.
                    - **Emergent Zero-Shot Generalization**: Zero-shot transfer to unseen physics tasks exhibits an emergent S-curve—scaling from 12.5% on single-environment baseline training to >90% when trained on multi-environment benchmarks.
                    - **Task-Agnostic 3D Spatial Priors**: Broad multi-view exposure forces the latent diffusion backbone to encode fundamental 3D geometric constraints rather than overfitting to environment-specific visual artifacts.
                    """
                )
                
                scaling_img_path = os.path.join(CURRENT_DIR, "./assets/scaling_law_envs.png")
                if os.path.exists(scaling_img_path):
                    gr.Image(
                        value=scaling_img_path,
                        type="filepath",
                        interactive=False,
                    )

        # --- TAB 2: ENVIRONMENT ZOO TAXONOMY ---
        with gr.TabItem("📚 Environment Zoo & Benchmark Taxonomy"):
            with gr.Column(elem_classes=["section-card"]):
                gr.Markdown(
                    """
                    # 📚 Environment Zoo & Benchmark Taxonomy
                    Below is the full catalog of continuous and discrete benchmark environments designed for multi-view evaluation.
                    """
                )
                gr.Markdown(load_description_text())

        # --- TAB 3: PROJECT ROADMAP ---
        with gr.TabItem("🗺️ Project Roadmap"):
            with gr.Column(elem_classes=["section-card"]):
                gr.Markdown(
                    """
                    # 🗺️ Orthographic World Models — Development Roadmap
                    Interactive progress tracking across core milestones from Box2D environment creation to PPO agent training and Latent Diffusion World Models.
                    """
                )
                
                cb1 = gr.Checkbox(label="1. Create Box2D Game Environments (Extruded 3D Projections)", value=True)
                cb2 = gr.Checkbox(label="2. Check: Game State Space, Action Space & Reward Function Engineering", value=True)
                cb3 = gr.Checkbox(label="3. Train PPO Expert Policy Reinforcement Learning Agents", value=True)
                cb4 = gr.Checkbox(label="4. Collect Multi-View Dataset Rollouts with Trained PPO Agents", value=True)
                cb5 = gr.Checkbox(label="5. Train Diffusion World Models (Joint Denoising Engine)", value=True)
                
                roadmap_summary = gr.Markdown()

                def update_roadmap(*states):
                    total = len(states)
                    done_count = sum(1 for s in states if s)
                    pct = int((done_count / total) * 100)
                    
                    status_html = f"### 📊 Project Progress: `{done_count}/{total} Milestones Completed ({pct}%)`\n\n"
                    
                    titles = [
                        "Create Box2D Game Environments (Extruded 3D Projections)",
                        "Check: Game State Space, Action Space & Reward Function Engineering",
                        "Train PPO Expert Policy Reinforcement Learning Agents",
                        "Collect Multi-View Dataset Rollouts with Trained PPO Agents",
                        "Train Diffusion World Models (Joint Denoising Engine)",
                    ]
                    
                    for idx, (title, st) in enumerate(zip(titles, states), 1):
                        icon = "✅" if st else "⏳"
                        status_str = "**Completed**" if st else "*In Progress / Pending*"
                        status_html += f"- {icon} **Step {idx}: {title}** — {status_str}\n"
                    
                    return status_html

                all_cbs = [cb1, cb2, cb3, cb4, cb5]
                
                for cb in all_cbs:
                    cb.change(fn=update_roadmap, inputs=all_cbs, outputs=[roadmap_summary])
                
                # Initial render
                demo.load(fn=update_roadmap, inputs=all_cbs, outputs=[roadmap_summary])

if __name__ == "__main__":
    demo.launch()
