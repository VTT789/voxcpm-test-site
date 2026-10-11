# VoxCPM2 Speech Tester

A small FastAPI + HTML interface for testing VoxCPM2 text-to-speech.

## Important hosting note

This app attempts to load `openbmb/VoxCPM2` on the first generation request. Render's free web service is not suitable for reliably running this large model; deployment may build but generation can fail from RAM, CPU, PyTorch/CUDA compatibility, or model download limits. For reliable inference, use a compatible machine with sufficient RAM/GPU and host the backend there.

## Deploy on Render

1. Upload these files to a GitHub repository.
2. In Render, choose **New + → Web Service** and connect the repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Deploy and open the assigned `onrender.com` URL.
6. Click **Check server**, then **Generate speech**.

## Run locally

Use a compatible Python environment (VoxCPM's official docs specify Python >=3.10 and <3.13, PyTorch >=2.5; CUDA is required for GPU acceleration). Then:

```bash
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000.

Official docs: https://voxcpm.readthedocs.io/en/latest/quickstart.html
