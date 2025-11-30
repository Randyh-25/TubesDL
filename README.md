# Face Recognition Web App

A modern web application for real-time face recognition using AI models (CNN & ViT). Built with Streamlit, PyTorch, and DeepFace.

## Features

- 📸 Upload images or use camera input
- 🤖 Face detection with DeepFace
- 🧠 Dual model predictions (EfficientNet-B4 CNN & ViT-B/16)
- 🎨 Modern, responsive UI
- 🚀 Deploy-friendly for free platforms

## Setup

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/yourrepo.git
   cd yourrepo
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run locally:
   ```bash
   streamlit run app.py
   ```

## Deploy to Hugging Face Spaces (Free)

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Create a new Space (select Streamlit)
3. Connect your GitHub repo or upload files
4. Add `requirements.txt` and `app.py`
5. For models, either:
   - Upload checkpoint files to the Space
   - Or modify `load_models()` to download from Google Drive

## Deploy to Render (Free)

1. Go to [Render](https://render.com)
2. Create a new Web Service
3. Connect your GitHub repo
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `streamlit run app.py --server.port $PORT --server.headless true`
6. Deploy!

## Notes

- Models are loaded on startup for efficiency
- Camera input works best in HTTPS environments
- For production, consider model quantization for faster inference

## License

MIT