# 🧠 Brain Tumor AI Detector (Streamlit App)

This is a Streamlit-based AI tool that detects types of brain tumors from MRI scans using a trained deep learning model.

---

## 🚀 Setup Instructions (for VS Code)

> Follow these steps **exactly** to get the project working on your PC.

---

### 📁 Step 1: Clone the Repository

Open **VS Code terminal** and run:

```bash
git clone https://github.com/SangramKhandagale/Brain_Tumor_Detection_Model
cd Brain_Tumor_Detection_Model

🔍 Step 2: Verify Folder Structure
Your folder should look like this:

brain-tumor-ai-app/
├── app.py
├── brain_tumor_model.h5
├── requirements.txt
└── README.md
✅ Make sure the brain_tumor_model.h5 file is in the same folder as app.py.

🐍 Step 3: Install Python (if not already installed)
Go to https://www.python.org/downloads

Download and install the latest version.

Important: ✅ During installation, check the box that says "Add Python to PATH"

After installing, restart VS Code

📦 Step 4: Install Required Packages
In the VS Code terminal, run these commands one by one:

bash

python -m pip install --upgrade pip
pip install -r requirements.txt
If pip install -r requirements.txt doesn't work, try:

bash

pip install streamlit tensorflow pillow opencv-python numpy

🧠 Step 5: Run the App
Use this command to launch the Streamlit app:

bash

python -m streamlit run app.py

🟢 This will:

Start a local server

Automatically open your browser

Show the Brain Tumor Detector at: http://localhost:8501

✅ What You'll See
Upload an MRI scan (.jpg, .png, or .jpeg)

Click “Analyze with AI”

Get predictions like: Glioma, Meningioma, No Tumor, Pituitary

View confidence scores and progress bars

🧾 Requirements
This project uses:

Python 3.x

Streamlit

TensorFlow

Pillow

OpenCV

NumPy

All listed in requirements.txt.

⚠️ Troubleshooting
❌ Error: "streamlit not recognized"
Run the app with:

bash

python -m streamlit run app.py
❌ Error: "model not found"
Ensure this file is present:


brain_tumor_model.h5
It must be in the same folder as app.py.
