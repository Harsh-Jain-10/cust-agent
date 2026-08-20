# 🛡️ OmniSupport AI: Multimodal Visual Customer Support & Diagnostic Agent

A production-grade, multimodal customer support and visual diagnostic web application built with **Streamlit** and powered by **Google Gemini 1.5 Flash** (`google-generativeai` SDK).

---

## 🌟 Key Features

1. **Multimodal Visual Diagnostics:**
   - Real-time optical inspection of damaged products, fractured screens, burnt boards, receipts, barcodes, medication labels, or engine diagnostic codes.
   - Accepts JPG, PNG, WEBP uploads or live webcam photo capture.

2. **Smart Warranty & Policy Engine:**
   - Real-time policy checking (configurable return windows, accidental damage coverage, proof of purchase requirements, fast-track RMA replacements).

3. **Real-Time Sentiment & Urgency Triage:**
   - Automatic classification of customer emotional state (`Calm`, `Frustrated`, `Critical`) and urgency tier (`Normal`, `High`, `Urgent`) with dynamic pulsing status badges.

4. **Multi-Industry Preset Scenarios (1-Click Demos):**
   - 📱 **Consumer Electronics Defect:** Hardware LCD matrix crack diagnosis.
   - 🛍️ **Retail Return & Receipt Audit:** Invoice SKU, date, and return-window verification.
   - 💊 **Pharmacy & Prescription FAQ:** Medication dosage, instructions, and storage check.
   - 🚗 **Automotive / Appliance Fault:** OBD-II engine code (P0420) and dashboard alarm assessment.

5. **Multi-Turn Conversational Memory:**
   - Context-aware support thread preserving dialogue history and visual attachment tracking.

6. **Actionable Claim Summary & Audit Ticket:**
   - Downloadable Markdown claim report (`.md`) with official Ticket ID, sentiment rating, policy verdict, and conversation transcript.

---

## 🚀 Quick Start (Local Setup)

### 1. Clone the repository & enter directory
```bash
git clone https://github.com/your-username/cust-facing-agent.git
cd cust-facing-agent
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure your Gemini API Key
Create a `.env` file from `.env.example`:
```bash
cp .env.example .env
```
Add your key inside `.env`:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```
*(You can also input your key directly into the application's sidebar UI).*

### 5. Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push your repository to **GitHub**.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New App"** and select:
   - **Repository:** `your-username/cust-facing-agent`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Under **Advanced Settings > Secrets**, add your Gemini API Key:
   ```toml
   GEMINI_API_KEY = "your_actual_gemini_api_key_here"
   ```
5. Click **Deploy**! 🚀

---

## 🛠️ Tech Stack
- **Frontend & App Framework:** [Streamlit](https://streamlit.io/)
- **Vision & LLM Engine:** [Google Gemini 1.5 Flash](https://aistudio.google.com/) via `google-generativeai`
- **Image Processing:** [Pillow (PIL)](https://pillow.readthedocs.io/)
- **Configuration & Secrets:** `python-dotenv`
