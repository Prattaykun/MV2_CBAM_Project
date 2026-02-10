# Deploying to Hugging Face Spaces 🚀

This guide explains how to deploy the MV2-CBAM Fire Detection API to Hugging Face Spaces using the Docker runtime.

## 1. Create a New Space
1.  Log in to [Hugging Face](https://huggingface.co/).
2.  Go to **Spaces** -> **Create new Space**.
3.  **Owner**: Your username.
4.  **Space Name**: e.g., `mv2-cbam-fire-detection`.
5.  **License**: MIT (or your choice).
6.  **SDK**: Select **Docker**.
7.  **Hardware**: `CPU basic • 2 vCPU • 16GB RAM` (Free).
8.  **Visibility**: Public or Private.
9.  Click **Create Space**.

## 2. Deploy via Web Upload (Easiest)
1.  In your new Space, go to the **Files** tab.
2.  Click **+ Add file** -> **Upload files**.
3.  Drag and drop the following files/folders from your local project:
    *   `backend/` (Folder)
    *   `ml_core/` (Folder)
    *   `Dockerfile`
    *   `backend/requirements.txt`
4.  **Important**: Do NOT upload `dataset/` or `venv/`.
5.  Commit the changes.
6.  The Space will automatically start building. You can watch the "Logs" tab.

## 3. Deploy via Git (Advanced)
If you have Git installed, you can clone the Space and push files.

```bash
# Clone your Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/mv2-cbam-fire-detection
cd mv2-cbam-fire-detection

# Copy project files into this folder
cp -r ../MV2_CBAM_Project/backend .
cp -r ../MV2_CBAM_Project/ml_core .
cp ../MV2_CBAM_Project/Dockerfile .
cp ../MV2_CBAM_Project/.dockerignore .

# Push to Hugging Face
git add .
git commit -m "Initial commit"
git push
```

## 4. Accessing the API
Once the Space is "Running", your API URL will be:
`https://YOUR_USERNAME-mv2-cbam-fire-detection.hf.space`

You can test it by appending `/docs` to see the Swagger UI:
`https://YOUR_USERNAME-mv2-cbam-fire-detection.hf.space/docs`

## 5. Connecting Frontend
1.  Go to Vercel (where your frontend is).
2.  Update the Environment Variable `NEXT_PUBLIC_API_URL`.
3.  Set it to your new Hugging Face URL:
    `https://YOUR_USERNAME-mv2-cbam-fire-detection.hf.space`
4.  Redeploy Vercel.
