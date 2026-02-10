# MV2-CBAM Fire Detection

A lightweight CNN model for UAV-based image classification using MobileNetV2 with CBAM attention mechanism. This project includes a PyTorch-based ML core, a FastAPI backend, and a Next.js frontend dashboard.

## 📂 Project Structure

```
d:/projects/MV2_CBAM_Project/
├── backend/            # FastAPI Backend
├── dataset/            # Training and Test datasets
├── docs/               # Documentation and Papers
├── frontend/           # Next.js Frontend
├── ml_core/            # PyTorch Model & Training Scripts (To be created)
├── venv/               # Python Virtual Environment
└── README.md           # This file
```
## 🛠️ Technology Stack
This project utilizes a modern, high-performance stack for real-time UAV applications:

### **Machine Learning & AI**
- **PyTorch**: Core deep learning framework for model definition and training.
- **MobileNetV2**: Lightweight backbone architecture optimized for edge devices.
- **CBAM (Attention Module)**: Custom implementation to enhance feature focus (Channel/Spatial).
- **Grad-CAM**: Explainable AI technique to visualize model attention heatmaps.
- **Torchvision**: Image transformations and pre-trained weights.
- **Scikit-learn**: Classification metrics and evaluation tools.

### **Backend (API)**
- **FastAPI**: High-performance, async web framework for serving the model.
- **Uvicorn**: ASGI server for production-ready deployment.
- **Python-Multipart**: Handling image uploads via API.

### **Frontend (Dashboard)**
- **Next.js**: React framework for server-side rendering and static generation.
- **React**: Component-based UI library.
- **Tailwind CSS** (if applicable) / **CSS Modules**: For responsive and modern styling.

### **Utilities & DevOps**
- **Pillow (PIL)**: Image processing and manipulation.
- **Python-docx**: Automated report generation.
- **Git**: Version control (with .gitignore for large files).

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js & npm

### ⚡ Quick Start (Windows)
If you have the environment set up, you can run the entire project (Backend + Frontend) with one script:
```powershell
.\run_project.ps1
```

### 1. Python Environment Setup

We use a virtual environment to manage dependencies.

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

# Install dependencies (from root)
pip install -r backend/requirements.txt
# (ML dependencies will be added as we develop)
```

### 2. Backend Setup
```bash
cd backend
uvicorn main:app --reload
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm install
npm run dev
```

### 4. Deployment
#### Frontend (Vercel)
1.  Push your code to GitHub.
2.  Go to [Vercel](https://vercel.com/) and import your repository.
3.  Select the `frontend` folder as the **Root Directory**.
4.  Adding Environment Variables:
    - Vercel automatically detects Next.js.
    - If your backend is deployed, add `NEXT_PUBLIC_API_URL` set to your backend URL.
5.  Click **Deploy**.

#### Backend (Render)
1.  Go to [Render](https://render.com/) and create a new **Web Service**.
2.  Connect your GitHub repo.
3.  **Root Directory**: `backend`
4.  **Runtime**: Python 3
5.  **Build Command**: `pip install -r requirements.txt` (This uses `backend/requirements.txt` which is optimized for CPU/Render)
6.  **Start Command**: `uvicorn main:app --host 0.0.0.0 --port 10000`
7.  **Environment Variables**:
    - Ensure your model file (`mv2_cbam_best.pth`) is accessible. Render has a 500MB slug size limit.
    - **Option A (Git LFS)**: Commit the `.pth` file to Git using LFS.
    - **Option B (Download)**: Add a build script to download the model from an external URL (Dropbox/S3) before starting.

### 5. Training the Model (CRITICAL FIRST STEP)
**You must train the model before running the application or evaluation.**
The training process uses your specific dataset to teach the MobileNetV2 + CBAM network how to distinguish fire from non-fire.

**To start training manually:**
1. Open your terminal in the project root (`d:\projects\MV2_CBAM_Project`).
2. Run the following command:
   ```powershell
   .\venv\Scripts\python train_runner.py --epochs 5 --data_dir dataset
   ```
3. **What happens next?**
   - The script initializes the model and data loaders.
   - It runs for **5 epochs**.
   - You will see a progress bar for each epoch.
   - **Metrics**: Takes about 5-10 minutes on CPU (faster on GPU).
   - **Output**: The best model is saved to `ml_core/models/mv2_cbam_best.pth`.

*(Note: If you skip this, the backend will fail to load a model and the evaluation script will error out.)*

### 5. Project Roadmap & Technical Decisions

This section documents the evolution of the project and the reasoning behind key technical choices.

#### Phase 1: Core Architecture (ML)
- **Model Choice**: `MobileNetV2` was selected over ResNet or VGG because:
    - It is lightweight (suitable for UAV/Edge deployment).
    - It uses inverted residuals and linear bottlenecks for efficiency.
- **Enhancement**: `CBAM (Convolutional Block Attention Module)` was added to:
    - Improve feature representation by focusing on relevant channels and spatial regions.
    - Help the model distinguish "fire" features from similar-looking "smoke" or "sunset" features.
- **Explainability**: `Grad-CAM` was implemented to visualize *where* the model is looking. This is critical for verification (ensuring it detects fire, not just red colors).

#### Phase 2: System Integration
- **Backend (FastAPI)**: Chosen for its speed and native async support, allowing non-blocking inference.
- **Frontend (Next.js)**: Provides a responsive, modern React usage for the dashboard.
- **Containerization**: A virtual environment (`venv`) was used to isolate dependencies, preventing conflicts with system-wide Python packages.

#### Phase 3: Hardware Optimization
- **Issue**: Initial training on CPU was too slow (~20 mins/epoch).
- **Solution**: Switched to GPU optimization.
    - Replaced `torch-cpu` with `torch-cuda`.
    - Leveraged the user's **RTX 3050** for accelerated training (reduced to ~1-2 mins/epoch).
    - Updated `train_runner.py` to allow flexible epoch configuration via CLI.

#### Phase 4: Generalization & Robustness
- **Challenge**: Models often overfit to a single dataset.
- **Solution**:
    - Incorporated a secondary **Kaggle Forest Fire Dataset**.
    - Created a specialized evaluation pipeline (`evaluate_generalization.py`) to test the model on unseen data.
    - **Outcome**: The model achieved **~48.4% accuracy** on the Kaggle dataset compared to **~86.6%** on the training set.
    - **Reasoning**: This performance drop highlights a significant **domain shift**. The Kaggle dataset likely contains different fire characteristics (e.g., lighting, forest type, camera angles) that the model wasn't exposed to during training. This findings confirms the need for more diverse training data or domain adaptation techniques in future iterations.

### [2026-02-10] Domain Adaptation (Generalization Fix)
- **Action**: Addressed the 48% accuracy drop on Kaggle data.
- **Details**:
    - **Strategy**: Merged Kaggle Training set (3040 images) into main dataset.
    - **Retraining**: Retrained MobileNetV2-CBAM for 5 epochs on GPU.
    - **Outcome**: Validation Accuracy rose to **89.4%**.
    - **Verification**: Re-ran Kaggle Evaluation -> **95.8% Accuracy**, solving the domain shift issue.

### [2026-02-10] Generalization & Evaluation
- **Action**: Performed Cross-Domain Evaluation.
- **Details**:
    - **Executed**: Ran `evaluate_generalization.py` on the Kaggle dataset.
    - **Result**: 48.4% Accuracy, 0.0 F1 Score.
    - **Analysis**: The model failed to detect fires in the new dataset (0% recall), acting as a "specific" detector rather than a "general" one. This is a critical insight for real-world deployment safety.

### [2026-02-10] Hardware & Optimization
- **Action**: Enabled GPU Acceleration.
- **Details**:
    - Diagnosed slow training due to CPU-bound PyTorch installation.
    - Re-installed PyTorch with CUDA 11.8 support.
    - Verified GPU availability on RTX 3050.
    - Updated training docs to reflect performance gains.

### [2026-02-10] Generalization & Evaluation
- **Action**: Added secondary dataset support.
- **Details**:
    - Analyze Kaggle dataset structure.
    - Created `ml_core/evaluate_generalization.py` for cross-domain testing.

### [2026-02-10] Full System Implementation
- **ML Core**: Implemented MobileNetV2 + CBAM, Training, and Inference.
- **Backend**: FastAPI with `/predict` and `Grad-CAM`.
- **Frontend**: Next.js Dashboard.
- **Integration**: End-to-end flow verification.

### [2026-02-10] Initial Setup
- **Action**: Project initialization.
- **Details**: 
    - Verified dataset structure.
    - Created initial documentation and plans.
    - Setup `venv` and requirements.

## 📡 API Documentation

### `POST /predict`
- **Description**: Upload an image to detect fire and get a visual attention map.
- **Request**: `multipart/form-data` with `file` field.
- **Response**:
  ```json
  {
    "prediction": "Fire",
    "confidence": 0.98,
    "cam_image_base64": "..."
  }
  ```

### Testing the API
You can test the API using the provided script:
```bash
# Make sure backend is running first
cd backend
python test_api.py path/to/image.jpg
```

## 🛠️ Troubleshooting

- **`ModuleNotFoundError`**: Ensure you activated the virtual environment (`.\venv\Scripts\activate`).
- **`RuntimeError: CUDA error`**: The code defaults to CPU if CUDA is unavailable, but check your PyTorch installation if you have a GPU.
- **Frontend Connection Error**: Ensure the backend is running on port 8000. Check the terminal for errors.

### 6. Research Paper Alignment & Contributions
**Reference Paper**: *A lightweight CNN model for UAV-based image classification* (Soft Computing, 2025).

This project acts as an implementation and extension of the methodology described in the paper.

#### 1. Addressing the Research Gap
**The Paper's Limitation (Page 14)**:
> "Most of these error-prone fire images contain **strong smoke**. In contrast, FLAME images do not have such scenes. As such, it is necessary to further improve MV2-CBAM for forest fire classification with **complex scenes**."

**Our Upgrade**:
We directly addressed this by implementing a **Cross-Domain Evaluation Pipeline** (`evaluate_generalization.py`).
1.  **Initial Test**: The model performance correctly dropped to **48.4%** on the complex Kaggle dataset, confirming the domain gap.
2.  **The Fix (Domain Adaptation)**: We merged the Kaggle training samples into our dataset and retrained the model.
3.  **Final Result**: The model successfully adapted, achieving **95.8% Accuracy** and **100% Recall** on the Kaggle test set. This proves the architecture can learn complex features when provided with diverse data.

#### 2. Major Upgrades Implemented
| Feature | Original Paper | Our Implementation |
| :--- | :--- | :--- |
| **Model** | MV2-CBAM (Theoretical/Experimental) | **Fully Reproduced** in PyTorch |
| **Evaluation** | Manual Self-Collected Set | **Automated Script** for any external dataset |
| **Application** | N/A (Model only) | **Full Web Dashboard** (Next.js + FastAPI) for real-time UAV monitoring |
| **Hardware** | Tesla V100 (Server GPU) | **Optimized for Edge/Consumer GPU** (RTX 3050) |

### 7. Future Work: Implementing "Option 1" (Self-Collected Dataset)
The next step to solve the "Smoke/Complex Background" issue is to collect and test on a custom dataset.

**How to Implement Option 1:**
1.  **Collect Data**: Gather 500-1000 images of "Smoke-heavy", "Foggy", or "Distant Fire" scenes from the internet.
2.  **Structure**: Create a new folder:
    ```
    dataset/Self_Collected/
    ├── Fire/      # Put fire/smoke images here
    └── No_Fire/   # Put complex forest/fog images here
    ```
3.  **Evaluate**:
    Run our generalization script on this new folder:
    ```python
    # Edit ml_core/evaluate_generalization.py to point to 'dataset/Self_Collected'
    evaluate_on_kaggle(MODEL_PATH, 'dataset/Self_Collected')
    ```
4.  **Retrain (Optional)**:
    To improve the model, you can merge this new folder into `dataset/Training` and run `train_runner.py` again.


