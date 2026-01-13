"""
RETRO-MORPHO v2: The Self-Assembling World Model
================================================
A merger of:
1. Retro-Causal Predictive Coding (Minimize Surprise)
2. Adaptive Morphogenesis (Grow Topology on Panic)

Features:
- Starts from Void (0 connections).
- "Panic" Mechanism: If Surprise > Average, it grows new wires.
- "Zen" Mechanism: If Surprise < Average, it prunes for efficiency.
- Breaks the 'Homeostasis Trap' by dynamically adjusting rates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np
import time
import tkinter as tk
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- CONFIGURATION ---
IMG_SIZE = 256            # Keep small for rapid growth dynamics
LATENT_DIM = 256         # Size of the "Concept" vector
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================================================================
# 1. THE MORPHING NEURON LAYER
# ===========================================================================
class MorphoLinear(nn.Module):
    """
    A Linear layer that starts dead (all zeros) and grows/prunes connections
    based on 'Panic' (Surprise relative to history).
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weights exist but are masked
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # The Mask: 0 = Dead, 1 = Alive
        self.register_buffer('mask', torch.zeros(out_features, in_features))
        
        # Init weights with potential (latent capability)
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        self.weight.data *= 0.1 # Keep them suppressed initially

    def forward(self, x, override=False):
        # If override=True, we use ALL weights (Hypothetical Probe)
        # If override=False, we only use GROWN weights (Reality)
        m = torch.ones_like(self.mask) if override else self.mask
        return F.linear(x, self.weight * m, self.bias)

    def morphogenesis_step(self, active_grad, current_loss, avg_loss):
        """
        TRUE NEUROPLASTICITY:
        - Panic Level > 1.1: The world is scary. GROW capacity.
        - Panic Level < 0.8: The world is boring. PRUNE efficiency.
        """
        # 1. Calculate Panic Level
        safe_avg = max(avg_loss, 0.0001)
        panic_level = current_loss / safe_avg
        
        # 2. Set Dynamic Rates based on Panic
        grow_rate = 0.001  # Default slow growth
        prune_rate = 0.05  # Default pruning
        
        if panic_level > 1.1:
            # PANIC MODE: Adrenaline Rush
            # Grow proportionally to confusion (capped at 10% per step)
            grow_rate = 0.05 * min(panic_level, 2.0)
            prune_rate = 0.0 # Stop pruning while learning!
            
        elif panic_level < 0.8:
            # ZEN MODE: Deep Sleep
            # Prune aggressively to save energy
            grow_rate = 0.0
            prune_rate = 0.005

        # --- EXECUTE PHYSICAL CHANGES ---
        with torch.no_grad():
            # A. DEMAND (Hypothetical Probe)
            demand = torch.abs(active_grad) * (1 - self.mask)
            # Failsafe for dead gradients
            if demand.max() == 0: demand = torch.rand_like(demand) * (1 - self.mask)
            
            # B. GROW
            n_dead = (self.mask == 0).sum().item()
            n_grow = int(n_dead * grow_rate)
            
            if n_grow > 0:
                _, idx = torch.topk(demand.flatten(), n_grow)
                self.mask.view(-1)[idx] = 1.0
                # Wake up with a "Spark" (small random weights)
                self.weight.view(-1)[idx] += torch.randn(n_grow).to(DEVICE) * 0.05

            # C. PRUNE
            n_alive = self.mask.sum().item()
            if n_alive > 100 and prune_rate > 0:
                strength = torch.abs(self.weight * self.mask)
                # Don't prune what is already dead
                strength[self.mask == 0] = float('inf')
                
                n_prune = int(n_alive * prune_rate)
                if n_prune > 0:
                    thresh = torch.topk(strength.flatten(), n_prune, largest=False).values.max()
                    self.mask[strength <= thresh] = 0

    def connection_count(self):
        return self.mask.sum().item()

# ===========================================================================
# 2. THE SELF-ASSEMBLING BRAIN
# ===========================================================================
class LivingWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # SENSES (Encoder): Standard CNN
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), LATENT_DIM),
            nn.Tanh()
        )
        
        # BRAIN (Predictor): THE MORPHING LAYER
        self.brain = MorphoLinear(LATENT_DIM, LATENT_DIM)
        
        # IMAGINATION (Decoder): Standard CNN
        self.dec_fc = nn.Linear(LATENT_DIM, 128 * (IMG_SIZE // 8) * (IMG_SIZE // 8))
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh()
        )

    def forward(self, x, override_brain=False):
        # 1. Encode Reality
        z = self.enc(x)
        
        # 2. Predict Future (Using sparse or hypothetical brain)
        z_next_pred = self.brain(z, override=override_brain)
        
        # 3. Decode Imagination
        x_next_pred = self.dec(
            self.dec_fc(z_next_pred).view(-1, 128, IMG_SIZE // 8, IMG_SIZE // 8)
        )
        
        return x_next_pred, z, z_next_pred

# ===========================================================================
# 3. THE LIFECYCLE ENGINE
# ===========================================================================
class MorphoEngine:
    def __init__(self):
        self.model = LivingWorldModel().to(DEVICE)
        self.opt = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.history_loss = []
        self.history_conns = []
        self.frame_curr = None # Memory of t-1
        
    def step(self, frame_bgr):
        # Preprocess
        img = cv2.resize(frame_bgr, (IMG_SIZE, IMG_SIZE))
        x = torch.from_numpy(img).float().permute(2,0,1) / 127.5 - 1.0
        x = x.unsqueeze(0).to(DEVICE)
        
        loss_val = 0
        conns = 0
        img_pred = np.zeros_like(img)
        img_surprise = np.zeros_like(img)
        
        if self.frame_curr is not None:
            # === PHASE 1: THE HYPOTHETICAL PROBE (Growth) ===
            self.model.zero_grad()
            
            # Predict future using ALL potential connections (Override=True)
            _, z_curr_real, z_next_hypo = self.model(self.frame_curr, override_brain=True)
            
            # Target is the ACTUAL latent of the current frame
            with torch.no_grad():
                z_target_real = self.model.enc(x)
            
            # Probe Loss
            probe_loss = F.mse_loss(z_next_hypo, z_target_real)
            probe_loss.backward()
            
            # --- ADAPTIVE GROWTH LOGIC ---
            # Use PREVIOUS frame's panic level to decide growth for THIS frame
            current_panic = self.history_loss[-1] if len(self.history_loss) > 0 else 0.01
            avg_panic = np.mean(self.history_loss) if len(self.history_loss) > 0 else 0.01
            
            self.model.brain.morphogenesis_step(self.model.brain.weight.grad, current_panic, avg_panic)
            
            # === PHASE 2: REALITY CHECK (Training) ===
            self.model.zero_grad()
            
            # Normal forward (Override=False)
            x_pred, z_curr, z_next = self.model(self.frame_curr, override_brain=False)
            
            # Loss: Visual Surprise + Conceptual Surprise
            l_vis = F.mse_loss(x_pred, x)
            l_lat = F.mse_loss(z_next, z_target_real.detach())
            
            total_loss = l_vis + l_lat
            total_loss.backward()
            
            # Mask gradients for dead synapses
            with torch.no_grad():
                self.model.brain.weight.grad *= self.model.brain.mask
            
            self.opt.step()
            
            # Stats
            loss_val = total_loss.item()
            conns = self.model.brain.connection_count()
            self.history_loss.append(loss_val)
            self.history_conns.append(conns)
            if len(self.history_loss) > 100: 
                self.history_loss.pop(0)
                self.history_conns.pop(0)
            
            # Viz
            img_pred = self.tensor_to_cv(x_pred)
            diff = torch.abs(x_pred - x)
            img_surprise = cv2.applyColorMap(self.tensor_to_cv(diff, 5.0), cv2.COLORMAP_JET)

        self.frame_curr = x
        return loss_val, conns, img_pred, img_surprise

    def tensor_to_cv(self, t, scale=1.0):
        # Detach and convert to numpy for display
        arr = t[0].permute(1,2,0).detach().cpu().numpy()
        arr = (arr * scale + 1) * 127.5
        return arr.clip(0,255).astype(np.uint8)

# ===========================================================================
# 4. GUI VISUALIZATION
# ===========================================================================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("RETRO-MORPHO: Artificial Life")
        self.root.configure(bg="#000")
        
        self.engine = MorphoEngine()
        self.cap = cv2.VideoCapture(0)
        
        self._init_ui()
        self.running = True
        self.loop()
        
    def _init_ui(self):
        # Header
        self.lbl_stats = tk.Label(self.root, text="INITIALIZING VOID...", bg="#000", fg="#0f0", font=("Consolas", 12))
        self.lbl_stats.pack(pady=5)
        
        # Video Panel
        self.panel = tk.Label(self.root, bg="#000")
        self.panel.pack()
        
        # Graphs
        self.fig = Figure(figsize=(10, 3), dpi=100, facecolor="#000")
        self.ax_loss = self.fig.add_subplot(121)
        self.ax_conn = self.fig.add_subplot(122)
        
        self.ax_loss.set_facecolor("#111")
        self.ax_conn.set_facecolor("#111")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def loop(self):
        if not self.running: return
        
        ret, frame = self.cap.read()
        if ret:
            loss, conns, pred, surp = self.engine.step(frame)
            
            # Update Text
            density = (conns / (LATENT_DIM**2)) * 100
            
            # Determine Brain State for UI
            avg = np.mean(self.engine.history_loss) if self.engine.history_loss else 0.01
            panic = loss / max(avg, 0.0001)
            state = "GROWING (PANIC)" if panic > 1.1 else ("PRUNING (ZEN)" if panic < 0.8 else "STABLE")
            color = "#f00" if panic > 1.1 else ("#00f" if panic < 0.8 else "#0f0")
            
            self.lbl_stats.config(text=f"SURPRISE: {loss:.4f} | SYNAPSES: {conns} ({density:.1f}%) | STATE: {state}", fg=color)
            
            # Composite Image: Input | Prediction | Surprise
            img_in = cv2.resize(frame, (200, 200))
            pred = cv2.resize(pred, (200, 200))
            surp = cv2.resize(surp, (200, 200))
            
            combined = np.hstack([img_in, pred, surp])
            im_pil = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
            im_tk = ImageTk.PhotoImage(image=im_pil)
            self.panel.config(image=im_tk)
            self.panel.image = im_tk
            
            # Update Graphs
            if len(self.engine.history_loss) > 10:
                self.ax_loss.clear()
                self.ax_conn.clear()
                
                self.ax_loss.plot(self.engine.history_loss, 'r')
                self.ax_loss.set_title("Surprise (Loss)", color='white')
                self.ax_loss.tick_params(colors='gray')
                
                self.ax_conn.plot(self.engine.history_conns, 'g')
                self.ax_conn.set_title("Brain Complexity (Connections)", color='white')
                self.ax_conn.tick_params(colors='gray')
                
                self.canvas.draw_idle()

        self.root.after(10, self.loop)

    def close(self):
        self.running = False
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()