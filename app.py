"""
OmniSupport AI: Multimodal Visual Customer Support & Diagnostic Agent
=====================================================================
A production-grade, multi-image visual customer support platform powered by
Google Gemini (Dynamic Multimodal Engine & Fallback Ladder) & Streamlit.

Supports multi-angle visual evidence inspection (2 to 5 images per claim).
Author: Principal Full-Stack AI Solutions Architect
"""

import os
import io
import time
import json
import uuid
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai
from dotenv import load_dotenv

# Load local environment variables if available
load_dotenv()

# =====================================================================
# 1. PAGE CONFIGURATION & STYLING
# =====================================================================
st.set_page_config(
    page_title="OmniSupport AI | Multi-Angle Visual Diagnostics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
/* ---------------- Main Theme Styling ---------------- */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Glassmorphism Top Banner */
.hero-container {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.98) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 24px 30px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
}

.hero-title {
    background: linear-gradient(90deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 6px;
}

.hero-subtitle {
    color: #94A3B8;
    font-size: 1.0rem;
    font-weight: 400;
    margin-bottom: 0;
}

/* Metrics Cards */
.metric-box {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 14px 18px;
    text-align: center;
    backdrop-filter: blur(8px);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.metric-box:hover {
    transform: translateY(-2px);
    border-color: rgba(96, 165, 250, 0.4);
}
.metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    margin-top: 4px;
}
.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #94A3B8;
    font-weight: 600;
}

/* Priority & Sentiment Badges */
.badge-calm {
    color: #34D399;
    background: rgba(52, 211, 153, 0.15);
    padding: 4px 10px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
    border: 1px solid rgba(52, 211, 153, 0.3);
}
.badge-frustrated {
    color: #FBBF24;
    background: rgba(251, 191, 36, 0.15);
    padding: 4px 10px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
    border: 1px solid rgba(251, 191, 36, 0.3);
}
.badge-critical {
    color: #F87171;
    background: rgba(248, 113, 113, 0.18);
    padding: 4px 10px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
    border: 1px solid rgba(248, 113, 113, 0.4);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(248, 113, 113, 0.4); }
    70% { box-shadow: 0 0 0 8px rgba(248, 113, 113, 0); }
    100% { box-shadow: 0 0 0 0 rgba(248, 113, 113, 0); }
}

/* Multi-Image Evidence Gallery */
.evidence-card {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 12px;
    padding: 10px;
    text-align: center;
    margin-bottom: 10px;
}

.evidence-tag {
    font-size: 0.78rem;
    color: #93C5FD;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Model Tag in Sidebar */
.model-pill {
    background: rgba(96, 165, 250, 0.12);
    border: 1px solid rgba(96, 165, 250, 0.3);
    color: #93C5FD;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-family: monospace;
}

/* Chat Container Enhancements */
.stChatMessage {
    border-radius: 14px;
    padding: 12px 18px;
    margin-bottom: 12px;
    animation: fadeIn 0.3s ease-in;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Ticket Preview Card */
.ticket-preview {
    background: #0F172A;
    border: 1px dashed #3B82F6;
    border-radius: 10px;
    padding: 16px;
    font-family: monospace;
    font-size: 0.85rem;
    color: #E2E8F0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =====================================================================
# 2. SESSION STATE MANAGEMENT
# =====================================================================
def init_session_state():
    """Initializes all necessary session state variables cleanly."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"OS-{uuid.uuid4().hex[:6].upper()}"
    if "start_time" not in st.session_state:
        st.session_state.start_time = datetime.now()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Store list of dicts: [{"image": PIL.Image, "name": str}]
    if "current_images" not in st.session_state:
        st.session_state.current_images = []
    if "sentiment" not in st.session_state:
        st.session_state.sentiment = "Calm"
    if "urgency" not in st.session_state:
        st.session_state.urgency = "Normal"
    if "claim_status" not in st.session_state:
        st.session_state.claim_status = "Inquiry"
    if "detected_issue" not in st.session_state:
        st.session_state.detected_issue = "Pending Analysis"
    if "active_model_name" not in st.session_state:
        st.session_state.active_model_name = "Auto-Detecting..."
    if "policy_config" not in st.session_state:
        st.session_state.policy_config = {
            "return_window_days": 30,
            "covers_accidental": False,
            "require_receipt": True,
            "allow_fast_replacement": True
        }

init_session_state()


# =====================================================================
# 3. SYNTHETIC PRESET GENERATOR (Multi-Angle Pairs: 2 Images Per Preset)
# =====================================================================
@st.cache_data(show_spinner=False)
def generate_synthetic_demo_pairs(scenario_type: str) -> List[Dict[str, Any]]:
    """
    Generates high-contrast visual diagnostic test pairs (at least 2 images)
    so hackathon judges can immediately test multi-angle inspection in 1 click.
    """
    pairs = []
    
    if scenario_type == "electronics":
        # Image 1: Front Cracked Screen
        img1 = Image.new("RGB", (700, 420), color=(245, 247, 250))
        d1 = ImageDraw.Draw(img1)
        d1.rounded_rectangle([(150, 40), (550, 380)], radius=20, fill=(30, 41, 59), outline=(100, 116, 139), width=4)
        d1.rectangle([(170, 60), (530, 340)], fill=(15, 23, 42))
        d1.text((220, 80), "OmniTech TabPro 11 [FRONT VIEW]", fill=(226, 232, 240))
        d1.text((220, 110), "Status: Screen Fracture Detected", fill=(248, 113, 113))
        crack_points = [(260, 180), (320, 230), (390, 210), (450, 270), (490, 260)]
        for i in range(len(crack_points) - 1):
            d1.line([crack_points[i], crack_points[i+1]], fill=(239, 68, 68), width=3)
        d1.line([(320, 230), (300, 290)], fill=(239, 68, 68), width=2)
        d1.rectangle([(230, 295), (470, 325)], fill=(220, 38, 38))
        d1.text((250, 302), "⚠️ PRIMARY ANGLE: LCD MATRIX CRACK", fill=(255, 255, 255))
        pairs.append({"image": img1, "name": "angle_1_screen_crack.png"})
        
        # Image 2: Rear Hardware Tag & Serial Number
        img2 = Image.new("RGB", (700, 420), color=(30, 41, 59))
        d2 = ImageDraw.Draw(img2)
        d2.rectangle([(160, 60), (540, 360)], fill=(15, 23, 42), outline=(71, 85, 105), width=3)
        d2.text((200, 90), "OMNITECH ELECTRONICS CORP", fill=(226, 232, 240))
        d2.text((200, 120), "REAR CHASSIS IDENTIFICATION TAG", fill=(96, 165, 250))
        d2.line([(180, 150), (520, 150)], fill=(51, 65, 85), width=2)
        d2.text((180, 170), "Model: OT-900X TabPro 11 (256GB)", fill=(203, 213, 225))
        d2.text((180, 200), "Serial Number: SN-88349-B2", fill=(52, 211, 153))
        d2.text((180, 230), "Mfg Date: 05/2024 | Rating: 15V 3A", fill=(148, 163, 184))
        d2.text((180, 260), "Barcode: ||| |||| || ||||| |||| |||", fill=(226, 232, 240))
        d2.text((180, 290), "Warranty Seal: INTACT / UNBROKEN", fill=(16, 185, 129))
        pairs.append({"image": img2, "name": "angle_2_serial_tag.png"})

    elif scenario_type == "retail_receipt":
        # Image 1: Retail Receipt
        img1 = Image.new("RGB", (700, 420), color=(255, 255, 255))
        d1 = ImageDraw.Draw(img1)
        d1.rectangle([(160, 20), (540, 400)], fill=(255, 255, 255), outline=(203, 213, 225), width=2)
        d1.text((240, 40), "APEX STORE RECEIPT", fill=(15, 23, 42))
        d1.text((240, 60), "Store #104 - Order #INV-94021", fill=(100, 116, 139))
        d1.line([(180, 85), (520, 85)], fill=(203, 213, 225), width=1)
        d1.text((180, 100), "Date of Purchase: 12-Days Ago", fill=(51, 65, 85))
        d1.text((180, 130), "Item: UltraNoiseCancelling Headset Gen 2", fill=(15, 23, 42))
        d1.text((180, 160), "SKU: 7729104 | Qty: 1 | Total: $249.99", fill=(15, 23, 42))
        d1.text((180, 190), "Return Policy: 30-Day Money Back Guarantee", fill=(100, 116, 139))
        pairs.append({"image": img1, "name": "evidence_1_official_receipt.png"})

        # Image 2: Product Box / Condition
        img2 = Image.new("RGB", (700, 420), color=(241, 245, 249))
        d2 = ImageDraw.Draw(img2)
        d2.rectangle([(150, 40), (550, 380)], fill=(30, 41, 59), outline=(96, 165, 250), width=3)
        d2.text((200, 80), "PRODUCT PACKAGING AUDIT", fill=(226, 232, 240))
        d2.text((200, 110), "Headset Gen 2 - SKU 7729104", fill=(148, 163, 184))
        d2.rectangle([(180, 150), (520, 270)], fill=(15, 23, 42))
        d2.text((200, 180), "Condition: Sealed Box / Factory Shrinkwrap", fill=(52, 211, 153))
        d2.text((200, 210), "UPC Barcode: 8 94820 01842 1", fill=(203, 213, 225))
        d2.text((200, 240), "Physical Damage: None (Mint Condition)", fill=(52, 211, 153))
        pairs.append({"image": img2, "name": "evidence_2_sealed_box.png"})

    elif scenario_type == "pharmacy":
        # Image 1: Front Prescription Label
        img1 = Image.new("RGB", (700, 420), color=(254, 243, 199))
        d1 = ImageDraw.Draw(img1)
        d1.rectangle([(150, 40), (550, 380)], fill=(254, 243, 199), outline=(217, 119, 6), width=3)
        d1.rectangle([(170, 60), (530, 130)], fill=(30, 58, 138))
        d1.text((220, 75), "CAREPLUS PHARMACY RX #67849", fill=(255, 255, 255))
        d1.text((220, 100), "Dr. Emily Hayes, MD | Refills: 2", fill=(191, 219, 254))
        d1.text((180, 150), "Medication: AMOXICILLIN 500MG CAPSULES", fill=(15, 23, 42))
        d1.text((180, 180), "Dosage: Take 1 capsule by mouth every 8 hours", fill=(180, 83, 9))
        pairs.append({"image": img1, "name": "rx_label_front.png"})

        # Image 2: Warning & Storage Details
        img2 = Image.new("RGB", (700, 420), color=(255, 251, 235))
        d2 = ImageDraw.Draw(img2)
        d2.rectangle([(150, 40), (550, 380)], fill=(255, 255, 255), outline=(239, 68, 68), width=3)
        d2.text((200, 70), "MEDICATION WARNING & STORAGE INSTRUCTIONS", fill=(185, 28, 28))
        d2.line([(170, 100), (530, 100)], fill=(252, 165, 165), width=2)
        d2.text((180, 120), "1. Complete entire 10-day prescribed course.", fill=(15, 23, 42))
        d2.text((180, 150), "2. Store between 20°C - 25°C (68°F - 77°F).", fill=(51, 65, 85))
        d2.text((180, 180), "3. Keep away from excessive moisture.", fill=(51, 65, 85))
        d2.text((180, 210), "Lot: #L88392 | Exp Date: 12/2026", fill=(180, 83, 9))
        pairs.append({"image": img2, "name": "rx_storage_warnings.png"})

    else:
        # Automotive Preset: Cluster Light + OBD Diagnostic Reader
        img1 = Image.new("RGB", (700, 420), color=(17, 24, 39))
        d1 = ImageDraw.Draw(img1)
        d1.circle((350, 200), 60, fill=(239, 68, 68, 50), outline=(239, 68, 68), width=4)
        d1.text((315, 185), "CHECK\nENGINE", fill=(254, 202, 202))
        d1.text((220, 290), "Dashboard Gauge Cluster: Active CEL", fill=(251, 191, 36))
        pairs.append({"image": img1, "name": "auto_dashboard_cel.png"})

        img2 = Image.new("RGB", (700, 420), color=(15, 23, 42))
        d2 = ImageDraw.Draw(img2)
        d2.rectangle([(140, 50), (560, 370)], fill=(30, 41, 59), outline=(96, 165, 250), width=3)
        d2.text((180, 80), "OBD-II DIGITAL DIAGNOSTIC SCANNER", fill=(226, 232, 240))
        d2.text((180, 120), "Diagnostic Trouble Code: P0420", fill=(239, 68, 68))
        d2.text((180, 150), "Description: Catalyst System Efficiency Below Threshold", fill=(251, 191, 36))
        d2.text((180, 180), "Bank: 1 | Severity: High | Freeze Frame: Stored", fill=(148, 163, 184))
        pairs.append({"image": img2, "name": "auto_obd_scanner_p0420.png"})

    return pairs


# =====================================================================
# 4. DYNAMIC MODEL DISCOVERY & MULTI-IMAGE REASONING ENGINE
# =====================================================================
CANDIDATE_MODEL_LADDER = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro-latest",
    "gemini-1.5-pro",
    "gemini-flash-latest",
    "gemini-pro-vision",
    "gemini-pro"
]

def resolve_available_models(api_key: str) -> List[str]:
    """Dynamically discovers and ranks available models supported by the API key."""
    try:
        os.environ["GOOGLE_API_KEY"] = api_key
        genai.configure(api_key=api_key, transport="rest")
        
        discovered = []
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                clean_name = m.name.replace("models/", "")
                discovered.append(clean_name)
        
        if discovered:
            def model_priority(name: str) -> int:
                n = name.lower()
                if "2.5" in n and "flash" in n:
                    return 1
                if "2.0" in n and "flash" in n:
                    return 2
                if "1.5" in n and "flash" in n and "latest" in n:
                    return 3
                if "1.5" in n and "flash" in n:
                    return 4
                if "1.5" in n and "pro" in n:
                    return 5
                if "flash" in n:
                    return 6
                if "gemini" in n:
                    return 7
                return 8
            
            return sorted(discovered, key=model_priority)
    except Exception:
        pass
    
    return CANDIDATE_MODEL_LADDER


def get_system_prompt(policy: Dict[str, Any], num_images: int) -> str:
    """Constructs the operational prompt with multi-image cross-referencing capabilities."""
    return f"""
You are OmniSupport AI, an elite Principal Multimodal Customer Support & Visual Diagnostic Agent.
Your objective is to provide compassionate, precise, and highly actionable customer support while visually inspecting customer evidence across multiple attached image angles ({num_images} images provided).

ACTIVE COMPANY POLICIES & BUSINESS RULES:
- Standard Return Window: {policy.get('return_window_days', 30)} days from purchase.
- Accidental Damage Coverage: {'INCLUDED in policy' if policy.get('covers_accidental') else 'EXCLUDED (Requires warranty upgrade or supervisor approval unless manufacturer defect)'}.
- Proof of Purchase Requirement: {'MANDATORY (Receipt, invoice, or serial number required)' if policy.get('require_receipt') else 'OPTIONAL'}.
- Fast Replacement Protocol: {'ENABLED for verified structural defects' if policy.get('allow_fast_replacement') else 'STANDARD REPAIR QUEUE ONLY'}.

CORE RESPONSIBILITIES:
1. MULTI-IMAGE VISUAL REASONING: Cross-reference ALL attached images (e.g. compare front defect vs rear serial number tag, or compare receipt SKU with product packaging). Highlight matches or discrepancies between the images clearly.
2. POLICY & WARRANTY EVALUATION: Evaluate if the issue qualifies for return, free replacement, warranty repair, or standard troubleshooting based on the combined visual evidence.
3. EMPATHIC COMMUNICATION: Adapt your tone to the customer's sentiment. If they are distressed or angry, validate their feelings and prioritize swift resolution.
4. STRUCTURED NEXT STEPS: Provide step-by-step guidance, troubleshooting tips, or RMA/Return instructions.
5. METADATA EXTRACTION: At the very end of your response, ALWAYS output a hidden JSON metadata block enclosed in `---OMNI_METADATA_START---` and `---OMNI_METADATA_END---`.

JSON SCHEMA TO INCLUDE IN THE METADATA BLOCK:
```json
{{
  "sentiment": "Calm" | "Frustrated" | "Critical",
  "urgency": "Normal" | "High" | "Urgent",
  "detected_issue": "Brief 3-6 word summary of detected issue",
  "claim_status": "Approved" | "Requires Inspection" | "Inquiry" | "Ineligible" | "Escalated",
  "action_recommended": "Specific next action for support rep or automated RMA"
}}
```

Ensure your direct message to the customer is professional, warm, clear, and structured with bold highlights and bullet points.
"""


def parse_agent_response(full_text: str) -> Tuple[str, Dict[str, Any]]:
    """Separates the customer-facing message from the agent metadata JSON."""
    meta_start = full_text.find("---OMNI_METADATA_START---")
    meta_end = full_text.find("---OMNI_METADATA_END---")
    
    metadata = {
        "sentiment": "Calm",
        "urgency": "Normal",
        "detected_issue": "Standard Inquiry",
        "claim_status": "In Progress",
        "action_recommended": "Continue Conversation"
    }
    
    if meta_start != -1 and meta_end != -1:
        clean_text = full_text[:meta_start].strip()
        json_str = full_text[meta_start + len("---OMNI_METADATA_START---"):meta_end].strip()
        
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        elif json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()
        
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                metadata.update(parsed)
        except Exception:
            pass
        return clean_text, metadata
    
    return full_text, metadata


def query_gemini_with_fallback(
    api_key: str,
    user_prompt: str,
    images: List[Dict[str, Any]],
    chat_history: List[Dict[str, Any]],
    policy: Dict[str, Any]
) -> Tuple[str, Dict[str, Any], str]:
    """
    Executes a multimodal generation call passing multiple images (2-5 images)
    with dynamic model discovery and sequential fallback ladder.
    """
    if not api_key:
        raise ValueError("Google Gemini API Key is missing. Please enter it in the sidebar or .env file.")

    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key, transport="rest")
    
    system_instruction = get_system_prompt(policy, len(images))
    model_candidates = resolve_available_models(api_key)
    recent_history = chat_history[-6:]
    
    # Construct multi-image content parts
    current_parts = []
    if images:
        current_parts.append(f"[Customer attached {len(images)} multi-angle visual evidence images for cross-inspection]:")
        for idx, img_entry in enumerate(images, 1):
            raw_img = img_entry["image"]
            if raw_img.mode != "RGB":
                raw_img = raw_img.convert("RGB")
            current_parts.append(raw_img)
            current_parts.append(f"Image {idx} Reference: {img_entry['name']}")
        current_parts.append(f"Customer Inquiry: {user_prompt}")
    else:
        current_parts.append(user_prompt)

    last_error = None
    
    for model_name in model_candidates:
        try:
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_instruction
                )
                formatted_contents = []
                for msg in recent_history:
                    role = "user" if msg["role"] == "user" else "model"
                    formatted_contents.append({"role": role, "parts": [msg["content"]]})
                formatted_contents.append({"role": "user", "parts": current_parts})
            except Exception:
                model = genai.GenerativeModel(model_name=model_name)
                formatted_contents = []
                formatted_contents.append({"role": "user", "parts": [f"System Instructions:\n{system_instruction}"]})
                formatted_contents.append({"role": "model", "parts": ["Understood. I will operate strictly as OmniSupport AI."]})
                for msg in recent_history:
                    role = "user" if msg["role"] == "user" else "model"
                    formatted_contents.append({"role": role, "parts": [msg["content"]]})
                formatted_contents.append({"role": "user", "parts": current_parts})
            
            generation_config = genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1500,
            )
            
            response = model.generate_content(
                contents=formatted_contents,
                generation_config=generation_config
            )
            
            if response and response.text:
                clean_text, meta = parse_agent_response(response.text)
                return clean_text, meta, model_name
                
        except Exception as e:
            err_str = str(e)
            last_error = err_str
            if "404" in err_str or "not found" in err_str.lower() or "not supported" in err_str.lower() or "400" in err_str:
                continue
            else:
                if "API_KEY_INVALID" in err_str or "403" in err_str or "quota" in err_str.lower():
                    raise e
                continue
                
    raise RuntimeError(f"All available Gemini model endpoints were attempted without success. Last error: {last_error}")


# =====================================================================
# 5. CLAIM TICKET GENERATOR
# =====================================================================
def build_claim_ticket_markdown() -> str:
    """Generates a downloadable formal Claim Summary & Diagnostic Report."""
    duration_secs = int((datetime.now() - st.session_state.start_time).total_seconds())
    mins, secs = divmod(duration_secs, 60)
    
    img_names = [img["name"] for img in st.session_state.current_images]
    evidence_str = ", ".join(img_names) if img_names else "No visual attachments"
    
    ticket = f"""# 🛡️ OmniSupport AI - Official Customer Claim & Diagnostic Ticket
**Ticket Reference ID:** `{st.session_state.session_id}`
**Generated Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
**Active AI Engine:** `{st.session_state.active_model_name}`
**Session Duration:** {mins}m {secs}s
**Total Interactions:** {len(st.session_state.messages)} messages

---

## 📊 1. MULTI-ANGLE DIAGNOSTIC ASSESSMENT & TRIAGE
- **Customer Sentiment:** {st.session_state.sentiment}
- **Triage Urgency:** {st.session_state.urgency}
- **Detected Issue Category:** {st.session_state.detected_issue}
- **Claim Eligibility Status:** {st.session_state.claim_status}
- **Visual Evidence Attached ({len(st.session_state.current_images)} Images):** {evidence_str}

---

## 📜 2. APPLIED WARRANTY & STORE POLICIES
- **Policy Window:** {st.session_state.policy_config.get('return_window_days', 30)} Days Return
- **Accidental Damage Covered:** {'Yes' if st.session_state.policy_config.get('covers_accidental') else 'No (Standard Defect Only)'}
- **Proof of Purchase Required:** {'Yes' if st.session_state.policy_config.get('require_receipt') else 'No'}
- **Fast-Track Replacement:** {'Authorized' if st.session_state.policy_config.get('allow_fast_replacement') else 'Standard Inspection Queue'}

---

## 💬 3. CONVERSATION AUDIT TRAIL
"""
    for idx, msg in enumerate(st.session_state.messages, 1):
        sender = "👤 Customer" if msg["role"] == "user" else "🤖 OmniSupport AI"
        img_tag = f" *(Evidence: {', '.join(msg['images_attached'])})*" if msg.get("images_attached") else ""
        ticket += f"\n### Turn {idx} - {sender}{img_tag}\n{msg['content']}\n"
        
    ticket += f"""
---
*OmniSupport AI Multimodal Engine | Powered by Google Gemini ({st.session_state.active_model_name})*
"""
    return ticket


# =====================================================================
# 6. SIDEBAR CONTROLS & CONFIGURATION
# =====================================================================
with st.sidebar:
    st.markdown("### ⚙️ Engine Configuration")
    
    env_api_key = os.getenv("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=env_api_key,
        type="password",
        help="Get your key at https://aistudio.google.com/"
    )
    
    active_api_key = api_key_input.strip() if api_key_input else env_api_key
    
    if active_api_key:
        st.success("✅ Gemini API Key connected", icon="🔑")
        st.markdown(f"**Engine:** <span class='model-pill'>{st.session_state.active_model_name}</span>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Enter Gemini API Key to enable AI reasoning", icon="⚠️")

    st.markdown("---")
    st.markdown("### 🧪 Multi-Angle Demo Presets")
    st.caption("Load 2 complementary visual evidence images in 1-click:")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button("📱 2x Electronics", use_container_width=True):
            st.session_state.current_images = generate_synthetic_demo_pairs("electronics")
            st.session_state.preset_prompt = "I have attached photos of both the cracked screen and the back serial tag. Is this eligible for a warranty replacement?"
            st.rerun()
            
        if st.button("💊 2x Rx Labels", use_container_width=True):
            st.session_state.current_images = generate_synthetic_demo_pairs("pharmacy")
            st.session_state.preset_prompt = "I uploaded both the front prescription label and the back storage warning label. Please verify dosage and safety rules."
            st.rerun()

    with col_p2:
        if st.button("🛍️ 2x Receipt+Box", use_container_width=True):
            st.session_state.current_images = generate_synthetic_demo_pairs("retail_receipt")
            st.session_state.preset_prompt = "I attached both my store invoice and the sealed product box. Can I get a full refund?"
            st.rerun()

        if st.button("🚗 2x CEL+Scanner", use_container_width=True):
            st.session_state.current_images = generate_synthetic_demo_pairs("automotive")
            st.session_state.preset_prompt = "I attached photos of the dashboard warning light and the OBD-II diagnostic trouble code report. What is the diagnosis?"
            st.rerun()

    st.markdown("---")
    st.markdown("### 📋 Smart Warranty Policies")
    with st.expander("Adjust Active Policy Rules", expanded=False):
        ret_days = st.slider("Return Window (Days)", 7, 90, st.session_state.policy_config["return_window_days"])
        accidental = st.checkbox("Cover Accidental Damage", value=st.session_state.policy_config["covers_accidental"])
        require_pop = st.checkbox("Require Proof of Purchase", value=st.session_state.policy_config["require_receipt"])
        fast_replace = st.checkbox("Allow Instant RMA Replacement", value=st.session_state.policy_config["allow_fast_replacement"])
        
        st.session_state.policy_config = {
            "return_window_days": ret_days,
            "covers_accidental": accidental,
            "require_receipt": require_pop,
            "allow_fast_replacement": fast_replace
        }

    st.markdown("---")
    st.markdown("### 📥 Claim Export & Reset")
    ticket_md = build_claim_ticket_markdown()
    st.download_button(
        label="📄 Download Claim Ticket (.md)",
        data=ticket_md,
        file_name=f"OmniSupport_Claim_{st.session_state.session_id}.md",
        mime="text/markdown",
        use_container_width=True
    )
    
    if st.button("🔄 Reset Conversation & State", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_images = []
        st.session_state.session_id = f"OS-{uuid.uuid4().hex[:6].upper()}"
        st.session_state.start_time = datetime.now()
        st.session_state.sentiment = "Calm"
        st.session_state.urgency = "Normal"
        st.session_state.claim_status = "Inquiry"
        st.session_state.detected_issue = "Pending Analysis"
        st.session_state.active_model_name = "Auto-Detecting..."
        if "preset_prompt" in st.session_state:
            del st.session_state.preset_prompt
        st.rerun()


# =====================================================================
# 7. MAIN INTERFACE: HEADER & METRICS DASHBOARD
# =====================================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🛡️ OmniSupport AI</div>
    <div class="hero-subtitle">Multi-Angle Visual Diagnostics & Multimodal Customer Support Agent Powered by Gemini</div>
</div>
""", unsafe_allow_html=True)

# Top Bar Live Diagnostic Metrics
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Ticket ID</div>
        <div class="metric-value" style="color: #60A5FA;">{st.session_state.session_id}</div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    sentiment_val = st.session_state.sentiment
    badge_class = "badge-calm" if sentiment_val == "Calm" else ("badge-frustrated" if sentiment_val == "Frustrated" else "badge-critical")
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">User Sentiment</div>
        <div class="metric-value"><span class="{badge_class}">{sentiment_val}</span></div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    urgency_val = st.session_state.urgency
    urgency_color = "#34D399" if urgency_val == "Normal" else ("#FBBF24" if urgency_val == "High" else "#F87171")
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Triage Urgency</div>
        <div class="metric-value" style="color: {urgency_color};">{urgency_val}</div>
    </div>
    """, unsafe_allow_html=True)

with col_m4:
    status_val = st.session_state.claim_status
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Claim Status</div>
        <div class="metric-value" style="color: #A78BFA;">{status_val}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)


# =====================================================================
# 8. MULTI-IMAGE EVIDENCE ATTACHMENT CHAMBER (2 to 5 Images)
# =====================================================================
num_attached = len(st.session_state.current_images)
chamber_expanded = True if num_attached > 0 else False

with st.expander(f"📷 Multi-Angle Evidence Chamber ({num_attached} / 5 Images Attached - Rule: 2 to 5 Images)", expanded=chamber_expanded):
    col_cam, col_upload = st.columns([1, 1])
    
    with col_upload:
        uploaded_files = st.file_uploader(
            "Select 2 to 5 Evidence Photos (PNG, JPG, JPEG, WEBP)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            help="Hold Ctrl/Cmd or select multiple files in your file dialog (attach between 2 and 5 images)."
        )
        if uploaded_files:
            if len(uploaded_files) > 5:
                st.warning("⚠️ Maximum 5 images allowed. Only the first 5 images will be attached.", icon="⚠️")
                uploaded_files = uploaded_files[:5]
                
            new_imgs = []
            for f in uploaded_files:
                try:
                    loaded_img = Image.open(f)
                    new_imgs.append({"image": loaded_img, "name": f.name})
                except Exception as e:
                    st.error(f"Error loading {f.name}: {e}")
            st.session_state.current_images = new_imgs

    with col_cam:
        enable_cam = st.toggle("📸 Snap Photo via Webcam", value=False)
        if enable_cam:
            camera_file = st.camera_input("Capture item photo to add to evidence")
            if camera_file is not None:
                if len(st.session_state.current_images) >= 5:
                    st.warning("⚠️ Maximum 5 images already attached. Please remove an image before capturing another.")
                else:
                    cam_img = Image.open(camera_file)
                    cam_name = f"webcam_angle_{len(st.session_state.current_images) + 1}_{int(time.time())}.png"
                    # Prevent duplicate additions of the same camera snapshot
                    if not any(img["name"] == cam_name for img in st.session_state.current_images):
                        st.session_state.current_images.append({"image": cam_img, "name": cam_name})
                        st.rerun()

    # Visual Evidence Validation Banner
    curr_count = len(st.session_state.current_images)
    if curr_count == 0:
        st.info("💡 **Evidence Requirement:** Please select or upload **at least 2 and at most 5 images** (e.g. front damage, rear serial number, invoice receipt, or packaging) or click a **Demo Preset** on the sidebar.", icon="ℹ️")
    elif curr_count == 1:
        st.warning("⚠️ **1 Image Attached:** Standard multi-angle diagnostic protocol requires **at least 2 images** (at most 5) for complete verification.", icon="📸")
    elif 2 <= curr_count <= 5:
        st.success(f"✅ **Multi-Angle Evidence Ready:** {curr_count} evidence angles attached (Within 2 to 5 image limit).", icon="🎯")
    else:
        st.error(f"🚫 **Too many images ({curr_count}):** Please keep attached images between 2 and 5.", icon="🛑")

    # Render Multi-Image Gallery Grid
    if st.session_state.current_images:
        st.markdown("---")
        st.markdown("##### 🔍 Attached Visual Evidence Gallery")
        
        cols = st.columns(min(max(len(st.session_state.current_images), 1), 5))
        for i, img_data in enumerate(st.session_state.current_images):
            with cols[i % len(cols)]:
                st.markdown(f"<div class='evidence-card'><div class='evidence-tag'>Angle {i+1}: {img_data['name']}</div></div>", unsafe_allow_html=True)
                st.image(img_data["image"], use_container_width=True)
                if st.button(f"❌ Remove #{i+1}", key=f"del_img_{i}", use_container_width=True):
                    st.session_state.current_images.pop(i)
                    st.rerun()
                    
        if st.button("🗑️ Clear All Attached Images", key="clear_all_imgs_btn"):
            st.session_state.current_images = []
            st.rerun()


# =====================================================================
# 9. CHAT HISTORY DISPLAY
# =====================================================================
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🛡️"):
        st.markdown("""
        **Hello! I'm OmniSupport AI**, your multimodal visual diagnostics and warranty specialist.
        
        How can I help you today?
        - 📸 **Attach 2 to 5 photos** (front defect, back serial tag, invoice, or warning lights) in the Evidence Chamber above.
        - 💬 Ask a question or use the **Quick Prompts** below or **Multi-Angle Presets** in the sidebar.
        """)

for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🛡️"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg.get("images_attached"):
            st.caption(f"📎 Attached Evidence ({len(msg['images_attached'])} Images): `{', '.join(msg['images_attached'])}`")
        st.markdown(msg["content"])


# =====================================================================
# 10. QUICK-PROMPT ACTION CHIPS
# =====================================================================
st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
qc1, qc2, qc3, qc4 = st.columns(4)

preset_triggered = None
with qc1:
    if st.button("🔍 Check Return Eligibility", use_container_width=True):
        preset_triggered = "Can you cross-reference all attached images (receipt and item condition) to verify if I qualify for a return?"
with qc2:
    if st.button("🛠️ Multi-Angle Diagnosis", use_container_width=True):
        preset_triggered = "Please inspect all attached evidence photos and provide step-by-step diagnostic troubleshooting."
with qc3:
    if st.button("🧾 Audit Serial & Invoice", use_container_width=True):
        preset_triggered = "Can you verify if the serial number on the product tag matches the invoice SKU across the attached images?"
with qc4:
    if st.button("⚡ Request Rapid RMA", use_container_width=True):
        preset_triggered = "The item is defective on arrival. I have attached all required photo evidence. Please authorize a rapid RMA ticket."

if "preset_prompt" in st.session_state and st.session_state.preset_prompt:
    preset_triggered = st.session_state.preset_prompt
    st.session_state.preset_prompt = None


# =====================================================================
# 11. CHAT INPUT & EXECUTION PIPELINE
# =====================================================================
chat_input_text = st.chat_input("Type your message or ask OmniSupport AI to inspect your visual evidence...")

final_prompt = chat_input_text if chat_input_text else preset_triggered

if final_prompt:
    if not active_api_key:
        st.error("🔑 Please provide a valid Google Gemini API Key in the sidebar or .env file to proceed.", icon="🚨")
    else:
        # Validate 2-5 image count rule if user has attached any images
        attached_count = len(st.session_state.current_images)
        if attached_count == 1:
            st.warning("⚠️ Notice: For optimal diagnostic precision, standard policy requires **at least 2 images** (up to 5). Proceeding with current single image...", icon="⚠️")
        elif attached_count > 5:
            st.error("🛑 Please remove excess images so that at most 5 images are attached.", icon="🛑")
            st.stop()
            
        img_names = [img["name"] for img in st.session_state.current_images]
        user_entry = {
            "role": "user",
            "content": final_prompt,
            "images_attached": img_names if img_names else None
        }
        st.session_state.messages.append(user_entry)
        
        with st.chat_message("user", avatar="👤"):
            if user_entry["images_attached"]:
                st.caption(f"📎 Attached Evidence ({len(user_entry['images_attached'])} Images): `{', '.join(user_entry['images_attached'])}`")
            st.markdown(final_prompt)
            
        with st.chat_message("assistant", avatar="🛡️"):
            with st.spinner(f"🤖 OmniSupport AI is cross-referencing {attached_count} visual evidence image(s) & policy rules..."):
                try:
                    response_text, meta, model_used = query_gemini_with_fallback(
                        api_key=active_api_key,
                        user_prompt=final_prompt,
                        images=st.session_state.current_images,
                        chat_history=st.session_state.messages[:-1],
                        policy=st.session_state.policy_config
                    )
                    
                    st.session_state.active_model_name = model_used
                    st.session_state.sentiment = meta.get("sentiment", st.session_state.sentiment)
                    st.session_state.urgency = meta.get("urgency", st.session_state.urgency)
                    st.session_state.claim_status = meta.get("claim_status", st.session_state.claim_status)
                    st.session_state.detected_issue = meta.get("detected_issue", st.session_state.detected_issue)
                    
                    st.markdown(response_text)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text
                    })
                    
                    st.rerun()
                    
                except Exception as err:
                    err_msg = str(err)
                    if "API_KEY_INVALID" in err_msg or "invalid api key" in err_msg.lower() or "403" in err_msg:
                        st.error("❌ The provided Google Gemini API Key is invalid or unauthorized. Please verify your key at https://aistudio.google.com/.", icon="🔑")
                    elif "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
                        st.error("⏳ Gemini API rate limit / quota exceeded. Please wait a moment before trying again.", icon="⏱️")
                    else:
                        st.error(f"⚠️ Service error: {err_msg}", icon="💥")
