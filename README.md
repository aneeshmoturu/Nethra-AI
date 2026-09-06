# Nethra AI: Vision-Centric Dynamic Trajectory Planning

> Real-time, zero-LiDAR trajectory evasion and kinematic stability engine designed for unstructured driving environments.

---

## 1. Executive Summary
Traditional Advanced Driver Assistance Systems (ADAS) depend on structured lane infrastructure and expensive LiDAR sensor arrays. Under non-standard traffic dynamics (e.g., unmarked roads, erratic vehicle paths, and stray obstacles), legacy systems trigger abrupt Emergency Autonomous Braking (AEB), elevating rear-end collision risks.

**Nethra AI** resolves this failure mode via a camera-only, edge-optimized spatial decision pipeline. By coupling object tracking with continuous monocular depth estimation, Time-to-Collision (TTC) derivation, and kinematic rollover verification, Nethra dynamically synthesizes safe **Swept-Path Bezier Trajectories** to evade obstacles while maintaining uninterrupted traffic flow.

---

## 2. System Architecture
[ Monocular RGB Camera ]
                              │
                              ▼
               [ YOLOv8 Perception + ByteTrack ]
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
   [ Spatial ROI Filter ]          [ Class-Based Depth Geometry ]
 (Ego-Path Identification)             (Pinhole Calibration)
              │                               │
              └───────────────┬───────────────┘
                              │
                              ▼
                   [ TTC & Dynamics Matrix ]
                  (Relative Velocity Vectors)
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
     TTC Safe / Space Open           TTC Critical / Blocked
              │                               │
              ▼                               ▼
  [ Kinematic Evasion Engine ]        [ Safe Yield Protocol ]
  - Lateral-G Rollover Check          (Controlled Deceleration)
  - Quintic Bezier Trajectory
  - BEV Perspective Transform
                              │
                              ▼
             [ Decoupled Telemetry Dashboard ]
           (Streamlit + Direct Thread Buffer)

           ## 3. Mathematical Formulations

### A. Pinhole Monocular Depth Approximation
Distance to detected entities is computed using real-world class width models projected through calibrated optical focal length:

$$D = \frac{W_{\text{real}} \cdot f}{w_{\text{bbox}}}$$

Where:
- $W_{\text{real}}$: Calibrated physical width of the detected target class (meters).
- $f$: Camera focal length parameter (pixels).
- $w_{\text{bbox}}$: Extracted bounding box horizontal dimension (pixels).

### B. Time-to-Collision (TTC) Engine
Using ByteTrack continuous track IDs, range history is smoothed over consecutive frames to calculate closing velocity $v_{\text{rel}}$:

$$v_{\text{rel}} = \frac{D_{t - \Delta t} - D_{t}}{\Delta t}$$

$$\text{TTC} = \frac{D_{t}}{v_{\text{rel}}}$$

* **TTC > 2.5s**: Nominal path cruise state.
* **1.5s $\le$ TTC $\le$ 2.5s**: Trajectory generation and dynamic evasion.
* **TTC < 1.5s**: Dynamic evasion window closed; engagement of controlled yield braking.

### C. Kinematic Stability & Rollover Constraint
Before rendering any evasion curve, lateral acceleration demand is evaluated against roll limits to ensure chassis stability:

$$a_{\text{lat}} = \frac{v_{\text{ego}}^2}{R} \le \mu \cdot g$$

Where $v_{\text{ego}}$ is vehicle speed, $R$ is minimum instantaneous curve radius, $\mu$ is tire-road friction coefficient, and $g = 9.81 \text{ m/s}^2$. If $a_{\text{lat}}$ exceeds safe lateral thresholds, steering overrides are locked out in favor of inline deceleration.

---

## 4. Key Features
* **Decoupled Asynchronous Pipeline:** Multithreaded architecture separates computer vision inference and coordinate math from the UI render loop.
* **Bird’s-Eye View (BEV) Projection:** Perspective matrix maps ground-plane contact coordinates to a top-down radar grid.
* **Swept-Path Hull Computation:** Models dynamic vehicle track width along Bezier coordinates rather than rendering single-pixel vectors.
* **Scenario Switcher:** Integrated simulation modes for Highway, Urban Density, and Low-Visibility scenarios.

---

## 5. Installation & Execution

### Prerequisites
- Python 3.10+
- Webcam or pre-recorded scenario media files

### Setup
```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/Nethra-AI.git](https://github.com/YOUR_USERNAME/Nethra-AI.git)
cd Nethra-AI

# Install dependencies
pip install -r requirements.txt

# Launch the edge telemetry interface
streamlit run app.py
### Step 4: Initialize Git and Push to GitHub

Run these commands in your VS Code terminal (inside the `nethra_ai` folder):

```bash
# Initialize local git repository
git init

# Add all files to staging
git add .

# Create the initial commit
git commit -m "feat: complete nethra core engine, kinematic constraints, and bev dashboard"