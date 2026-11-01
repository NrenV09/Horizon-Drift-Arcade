# Horizon Pulse: Golden Hour Festival 🏎️🌅

A high-performance, 3D open-world arcade driving experience built with Three.js, Cannon-es, procedural terrain heightfields, dynamic lighting cycles (Day / Sunset / Night), drifting skill chains, AI traffic rivals, audio engine synthesis, and calibrated ACESFilmic post-processing.

Bundled as a fully offline, self-contained single-page web app and ready for **1-click automatic deployment to GitHub Pages**.

---

## 🚀 Automatic Deployment on GitHub

This repository comes pre-configured with a production-ready **GitHub Actions CI/CD workflow** (`.github/workflows/deploy.yml`).

### 1. Push Code to GitHub
Push this codebase to your GitHub repository:
```bash
git init
git add .
git commit -m "Initial commit: Horizon Pulse 3D Driving Simulator"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

### 2. Enable GitHub Pages in Repository Settings
1. On GitHub, navigate to your repository.
2. Click **Settings** (top tab) ➔ **Pages** (left sidebar under *Code and automation*).
3. Under **Build and deployment**:
   - Change **Source** to **GitHub Actions**.
4. That's it! Every push to `main` or `master` will automatically trigger the deployment workflow.

### 3. Automatic Deployment Workflow Overview
Whenever you push changes:
1. **Automated Unit Tests**: Runs the 9 unit tests in `src/game/rules.test.ts` (verifying highway elevations, 4-wheel ground pose physics, telemetry conversions, drift skill chains, analog steering, and ACESFilmic exposure levels).
2. **Production Build**: Compiles TypeScript and inlines all assets into a standalone bundle in `dist/index.html` with relative asset paths (`base: './'`).
3. **Live Deployment**: Uploads the artifact and publishes it to:
   ```
   https://<your-username>.github.io/<your-repo-name>/
   ```
4. You can also manually trigger a deployment at any time from the **Actions** tab ➔ **Deploy to GitHub Pages** ➔ **Run workflow**.

---

## 🎮 Game Controls

### Keyboard Controls
| Action | Key(s) | Description |
| :--- | :--- | :--- |
| **Accelerate** | `W` or `↑` | Drive forward with progressive engine pitch |
| **Steer Left** | `A` or `←` | Steer wheels left with smooth auto-centering |
| **Steer Right** | `D` or `→` | Steer wheels right with smooth auto-centering |
| **Brake / Reverse** | `S` or `↓` | Apply brakes or reverse when stopped |
| **Handbrake / Drift** | `Space` | Break rear traction and initiate drifting chain |
| **Nitro Boost** | `Shift` or `N` | Turbo impulse thrust with cyan exhaust flame & camera pull |
| **Camera View** | `C` | Cycle Chase (3rd Person), Cockpit (Hood), and Far Cinematic |
| **Time of Day** | `T` | Cycle between Day, Golden Sunset, and Midnight |
| **Reset Vehicle** | `R` | Instantly upright vehicle back onto highway centerline |
| **Toggle Audio** | `M` | Mute / Unmute dynamic procedural Web Audio synthesizer |

### Mobile & Touch Controls
- **Virtual Analog Steering Pad**: Ergonomic bottom-left touch thumbstick with proportional steering sensitivity.
- **Drive / Brake Pedals**: Dedicated touch throttle and brake buttons on bottom-right.
- **Quick Action Buttons**: Nitro boost, Handbrake drift, Reset, Time of Day, and Camera view buttons placed within easy thumb reach.

---

## 🛠️ Local Development

### Requirements
- Node.js 18+ (Node 22 recommended)
- npm or bun

### Commands
```bash
# Install dependencies
npm install

# Run local development server (port 3000)
npm run dev

# Run unit test suite
npm test

# Build production distribution
npm run build

# Preview production build locally
npm run preview
```

---

## 📁 Repository Structure

```
├── .github/
│   └── workflows/
│       └── deploy.yml          # Automated CI/CD workflow for GitHub Pages
├── FINAL/
│   ├── final1.html             # Pre-built standalone single-file playable game
│   └── index.html              # Offline mirror bundle
├── src/
│   ├── game/
│   │   ├── rules.ts            # Pure game rules, highway geometry, & math logic
│   │   └── rules.test.ts       # 9 unit tests verifying mechanics & physics
│   ├── App.tsx                 # Main React HUD overlay & canvas wrapper
│   └── main.tsx                # Client entry point
├── dist/
│   └── index.html              # Production build artifact deployed to GitHub Pages
├── final1.html                 # Direct root offline single-file executable
├── index.html                  # Root template
├── package.json                # Project dependencies and npm scripts
├── tsconfig.json               # TypeScript compiler options
└── vite.config.ts              # Vite config with relative base & singlefile bundling
```

---

## ⚡ 100% Offline Standalone Play
You can also run this game completely offline without any internet connection or web server:
- Simply double-click `final1.html` or `FINAL/final1.html` in any modern web browser (Chrome, Safari, Edge, Firefox).
- All 3D meshes, procedural shaders, audio synthesizers, and physics routines are 100% self-contained in a single file with zero external CDN dependencies.
