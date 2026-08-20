"""
OmniSupport AI: Multimodal Visual Customer Support & Diagnostic Agent
=====================================================================
A production-grade, multi-image visual customer support platform powered by
Google Gemini (Dynamic Multimodal Engine & Fallback Ladder) & Streamlit.

Supports multi-angle visual evidence inspection (2 to 5 images per claim)
with strict real-world date grounding and zero-hallucination policy triage.

Author: Principal Full-Stack AI Solutions Architect
"""

import os
import io
import re
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
    page_title="OmniSupport AI | Multimodal Visual Diagnostics",
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
    Generates realistic, clean diagnostic test fixtures (2 images)
    without hardcoding fictional brand assumptions.
    """
    pairs = []
    
    if scenario_type == "electronics":
        # Image 1: Front View with Screen Crack
        img1 = Image.new("RGB", (700, 420), color=(245, 247, 250))
        d1 = ImageDraw.Draw(img1)
        d1.rounded_rectangle([(150, 40), (550, 380)], radius=20, fill=(30, 41, 59), outline=(100, 116, 139), width=4)
        d1.rectangle([(170, 60), (530, 340)], fill=(15, 23, 42))
        d1.text((210, 80), "TOUCHSCREEN DEVICE [FRONT VIEW]", fill=(226, 232, 240))
        d1.text((210, 110), "Status: Diagonal Glass Fracture Detected", fill=(248, 113, 113))
        crack_points = [(260, 180), (320, 230), (390, 210), (450, 270), (490, 260)]
        for i in range(len(crack_points) - 1):
            d1.line([crack_points[i], crack_points[i+1]], fill=(239, 68, 68), width=3)
        d1.line([(320, 230), (300, 290)], fill=(239, 68, 68), width=2)
        d1.rectangle([(220, 295), (480, 325)], fill=(220, 38, 38))
        d1.text((235, 302), "⚠️ PHYSICAL DAMAGE: LCD IMPACT CRACK", fill=(255, 255, 255))
        pairs.append({"image": img1, "name": "device_angle_1_screen_crack.png"})
        
        # Image 2: Rear View with Model / Serial Tag
        img2 = Image.new("RGB", (700, 420), color=(30, 41, 59))
        d2 = ImageDraw.Draw(img2)
        d2.rectangle([(160, 60), (540, 360)], fill=(15, 23, 42), outline=(71, 85, 105), width=3)
        d2.text((210, 90), "HARDWARE IDENTIFICATION PLATE", fill=(226, 232, 240))
        d2.text((210, 120), "REAR ENCLOSURE & SERIAL INFO", fill=(96, 165, 250))
        d2.line([(180, 150), (520, 150)], fill=(51, 65, 85), width=2)
        d2.text((180, 170), "Model: Standard Mobile Unit (128GB)", fill=(203, 213, 225))
        d2.text((180, 200), "Serial No: SN-90482-TX", fill=(52, 211, 153))
        d2.text((180, 230), "Rated Input: 5V 2A | FCC ID: Approved", fill=(148, 163, 184))
        d2.text((180, 260), "Barcode: ||| |||| || ||||| |||| |||", fill=(226, 232, 240))
        d2.text((180, 290), "Tamper Seal: Intact", fill=(16, 185, 129))
        pairs.append({"image": img2, "name": "device_angle_2_serial_tag.png"})

    elif scenario_type == "retail_receipt":
        # Image 1: Purchase Receipt
        img1 = Image.new("RGB", (700, 420), color=(255, 255, 255))
        d1 = ImageDraw.Draw(img1)
        d1.rectangle([(160, 20), (540, 400)], fill=(255, 255, 255), outline=(203, 213, 225), width=2)
        d1.text((230, 40), "OFFICIAL SALES RECEIPT", fill=(15, 23, 42))
        d1.text((230, 60), "Retail Store #204 - Order #INV-84920", fill=(100, 116, 139))
        d1.line([(180, 85), (520, 85)], fill=(203, 213, 225), width=1)
        d1.text((180, 100), "Date of Purchase: Recent (10 Days Ago)", fill=(51, 65, 85))
        d1.text((180, 130), "Item: Wireless Bluetooth Headset", fill=(15, 23, 42))
        d1.text((180, 160), "SKU: 6592014 | Qty: 1 | Total: $179.99", fill=(15, 23, 42))
        d1.text((180, 190), "Terms: 30-Day Return on Unopened Items", fill=(100, 116, 139))
        pairs.append({"image": img1, "name": "purchase_receipt_inv84920.png"})

        # Image 2: Unopened Product Box
        img2 = Image.new("RGB", (700, 420), color=(241, 245, 249))
        d2 = ImageDraw.Draw(img2)
        d2.rectangle([(150, 40), (550, 380)], fill=(30, 41, 59), outline=(96, 165, 250), width=3)
        d2.text((200, 80), "PRODUCT PACKAGING AUDIT", fill=(226, 232, 240))
        d2.text((200, 110), "Headset Package - SKU 6592014", fill=(148, 163, 184))
        d2.rectangle([(180, 150), (520, 270)], fill=(15, 23, 42))
        d2.text((200, 180), "Packaging State: Factory Sealed in Shrinkwrap", fill=(52, 211, 153))
        d2.text((200, 210), "UPC Barcode: 8 94820 01842 1", fill=(203, 213, 225))
        d2.text((200, 240), "Item Condition: Mint / Undamaged", fill=(52, 211, 153))
        pairs.append({"image": img2, "name": "product_packaging_sealed.png"})

    elif scenario_type == "pharmacy":
        # Image 1: Prescription Label
        img1 = Image.new("RGB", (700, 420), color=(254, 243, 199))
        d1 = ImageDraw.Draw(img1)
        d1.rectangle([(150, 40), (550, 380)], fill=(254, 243, 199), outline=(217, 119, 6), width=3)
        d1.rectangle([(170, 60), (530, 130)], fill=(30, 58, 138))
        d1.text((220, 75), "HEALTHCARE RX BOTTLE #RX-48201", fill=(255, 255, 255))
        d1.text((220, 100), "Prescribing Physician: General Clinic", fill=(191, 219, 254))
        d1.text((180, 150), "Medication: AMOXICILLIN 500MG CAPSULES", fill=(15, 23, 42))
        d1.text((180, 180), "Directions: Take 1 capsule by mouth every 8 hours", fill=(180, 83, 9))
        pairs.append({"image": img1, "name": "prescription_label_front.png"})

        # Image 2: Warnings & Expiration
        img2 = Image.new("RGB", (700, 420), color=(255, 251, 235))
        d2 = ImageDraw.Draw(img2)
        d2.rectangle([(150, 40), (550, 380)], fill=(255, 255, 255), outline=(239, 68, 68), width=3)
        d2.text((200, 70), "SAFETY INSTRUCTIONS & STORAGE INFO", fill=(185, 28, 28))
        d2.line([(170, 100), (530, 100)], fill=(252, 165, 165), width=2)
        d2.text((180, 120), "1. Complete entire 10-day prescribed course.", fill=(15, 23, 42))
        d2.text((180, 150), "2. Store between 20°C - 25°C in dry conditions.", fill=(51, 65, 85))
        d2.text((180, 180), "3. Keep out of reach of children.", fill=(51, 65, 85))
        d2.text((180, 210), "Lot: #L88392 | Exp: Valid until 2027", fill=(180, 83, 9))
        pairs.append({"image": img2, "name": "prescription_storage_warnings.png"})

    else:
        # Automotive: Dashboard CEL + OBD Code
        img1 = Image.new("RGB", (700, 420), color=(17, 24, 39))
        d1 = ImageDraw.Draw(img1)
        d1.circle((350, 200), 60, fill=(239, 68, 68, 50), outline=(239, 68, 68), width=4)
        d1.text((315, 185), "CHECK\nENGINE", fill=(254, 202, 202))
        d1.text((220, 290), "Dashboard Gauge Cluster: Active CEL Alarm", fill=(251, 191, 36))
        pairs.append({"image": img1, "name": "dashboard_check_engine_light.png"})

        img2 = Image.new("RGB", (700, 420), color=(15, 23, 42))
        d2 = ImageDraw.Draw(img2)
        d2.rectangle([(140, 50), (560, 370)], fill=(30, 41, 59), outline=(96, 165, 250), width=3)
        d2.text((180, 80), "OBD-II DIAGNOSTIC SCANNER OUTPUT", fill=(226, 232, 240))
        d2.text((180, 120), "Trouble Code: P0420", fill=(239, 68, 68))
        d2.text((180, 150), "Description: Catalyst System Efficiency Below Threshold", fill=(251, 191, 36))
        d2.text((180, 180), "Severity: Moderate | Diagnostic Status: Confirmed Fault", fill=(148, 163, 184))
        pairs.append({"image": img2, "name": "obd_scanner_fault_code.png"})

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
    """
    Constructs the operational prompt with strict date anchoring,
    zero-hallucination rules, and structured multi-asset cross-verification.
    """
    current_date_str = datetime.now().strftime("%B %d, %Y")
    current_iso_date = datetime.now().strftime("%Y-%m-%d")
    
    return f"""
You are OmniSupport AI, an elite Principal Multimodal Customer Support & Visual Diagnostic Agent.
Your objective is to provide compassionate, precise, objective, and highly actionable customer support while visually inspecting customer evidence across multiple attached image angles ({num_images} images provided).

CURRENT REAL-WORLD DATE REFERENCE:
- Today's Date: {current_date_str} (ISO: {current_iso_date}).
- Use today's date to accurately evaluate receipt/invoice purchase dates against the active return window. For example, if a receipt is from 2024 or earlier, it is clearly in the past relative to {current_date_str}. Do NOT mistake past purchase dates for future dates.

ACTIVE COMPANY POLICIES & BUSINESS RULES:
- Standard Return Window: {policy.get('return_window_days', 30)} days from invoice purchase date.
- Accidental Damage Coverage: {'INCLUDED in policy' if policy.get('covers_accidental') else 'EXCLUDED under standard warranty (accidental physical damage such as screen cracks, liquid spills, or impact drops is NOT covered under standard manufacturer warranty, requiring out-of-warranty paid repair or dedicated accidental protection plans)'}.
- Proof of Purchase Requirement: {'MANDATORY (Official receipt or invoice is required for warranty/return claims)' if policy.get('require_receipt') else 'OPTIONAL'}.
- Fast Replacement Protocol: {'ENABLED for verified manufacturer structural defects' if policy.get('allow_fast_replacement') else 'STANDARD REPAIR QUEUE ONLY'}.

CRITICAL ANTI-HALLUCINATION & EVIDENCE GROUNDING RULES:
1. STRICT VISUAL GROUNDING: Rely ONLY on text and details directly visible in the uploaded images or explicitly provided in the user's message.
2. NO FICTIONAL BIAS: Never assume fictional model names, brands, or serial numbers. If an image shows an Apple iPhone, Android phone, receipt, or other real device, identify and extract the EXACT brand, model, and invoice text visible in the image. If unbranded or unreadable, describe it generically (e.g., "touchscreen smartphone with shattered OLED display").
3. MULTI-ASSET CROSS-VERIFICATION: When multiple images are provided (e.g., physical item damage + invoice/receipt document):
   - Image 1 (Physical Item): Document the exact damage location, physical crack pattern, and cosmetic state.
   - Image 2 (Invoice / Receipt / Tag): Extract the Store Name, Invoice / Order Number, Date of Purchase, Item Purchased (e.g. Apple iPhone 13, Headset, etc.), and Total Amount.
   - Cross-Verification & Warranty Verdict: Cross-reference whether the item shown in the damage photo matches the invoice, calculate days elapsed since purchase date against the {policy.get('return_window_days', 30)}-day window, and explicitly determine if the observed damage (e.g., cracked screen) is covered under standard warranty or excluded as accidental physical damage.
4. ACTIONABLE GUIDANCE: Provide clear next steps (e.g., Authorized Service Center out-of-warranty repair quote options, accidental protection claim steps, or RMA return procedures).
5. STRUCTURED RESPONSE FORMAT: Format your response clearly with bold headers:
   - 🔍 **Visual Evidence & Cross-Verification Findings**
   - 📜 **Policy & Warranty Evaluation**
   - 🛠️ **Recommended Action Plan & Next Steps**
6. METADATA EXTRACTION: At the very end of your response, ALWAYS output a hidden JSON metadata block enclosed in `---OMNI_METADATA_START---` and `---OMNI_METADATA_END---`.

JSON SCHEMA TO INCLUDE IN THE METADATA BLOCK:
```json
{{
  "sentiment": "Calm" | "Frustrated" | "Critical",
  "urgency": "Normal" | "High" | "Urgent",
  "detected_issue": "Brief 3-6 word summary of detected issue",
  "claim_status": "Approved" | "Requires Inspection" | "Inquiry" | "Ineligible" | "Escalated",
  "action_recommended": "Specific next action for support rep or customer"
}}
```

Ensure your customer-facing message is empathetic, professional, clear, and cleanly formatted without raw JSON leaking into the main text.
"""


def parse_agent_response(full_text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Robustly separates the customer-facing message from the agent metadata JSON.
    Guarantees no raw metadata delimiters or json blocks leak into the user chat.
    """
    metadata = {
        "sentiment": "Calm",
        "urgency": "Normal",
        "detected_issue": "Standard Inquiry",
        "claim_status": "In Progress",
        "action_recommended": "Continue Conversation"
    }
    
    # 1. Regex search for metadata block
    pattern = r"---OMNI_METADATA_START---(.*?)---OMNI_METADATA_END---"
    match = re.search(pattern, full_text, flags=re.DOTALL)
    
    if match:
        clean_text = full_text[:match.start()].strip()
        json_content = match.group(1).strip()
        
        # Strip markdown code fences if present inside the tag
        if json_content.startswith("```json"):
            json_content = json_content[7:]
        elif json_content.startswith("```"):
            json_content = json_content[3:]
        if json_content.endswith("```"):
            json_content = json_content[:-3]
        json_content = json_content.strip()
        
        try:
            parsed = json.loads(json_content)
            if isinstance(parsed, dict):
                metadata.update(parsed)
        except Exception:
            pass
            
        return clean_text, metadata
    
    # 2. Fallback: Check if delimiter was partially written without ending tag
    if "---OMNI_METADATA_START---" in full_text:
        clean_text = full_text.split("---OMNI_METADATA_START---")[0].strip()
        return clean_text, metadata
        
    return full_text.strip(), metadata


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
        current_parts.append(f"[Customer attached {len(images)} visual evidence images for multimodal cross-inspection]:")
        for idx, img_entry in enumerate(images, 1):
            raw_img = img_entry["image"]
            if raw_img.mode != "RGB":
                raw_img = raw_img.convert("RGB")
            current_parts.append(raw_img)
            current_parts.append(f"Image {idx} Filename: {img_entry['name']}")
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
                temperature=0.2,
                max_output_tokens=1800,
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
- **Accidental Damage Covered:** {'Yes' if st.session_state.policy_config.get('covers_accidental') else 'No (Physical/Accidental Damage Excluded from Standard Warranty)'}
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
    st.caption("Load 2 real-world evidence test pairs in 1-click:")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button("📱 Screen Damage + Tag", use_container_width=True):
            st.session_state.current_images = generate_synthetic_demo_pairs("electronics")
            st.session_state.preset_prompt = "I have attached photos of the cracked screen and the back model/serial label. Is this repair covered under standard warranty?"
            st.rerun()
            
        if st.button("💊 Rx Bottle + Directions", use_container_width=True):
            st.session_state.current_images = generate_synthetic_demo_pairs("pharmacy")
            st.session_state.preset_prompt = "I uploaded photos of the prescription bottle label and the storage instructions. Can you verify my dosage schedule and safety warnings?"
            st.rerun()

    with col_p2:
        if st.button("🛍️ Invoice + Item Box", use_container_width=True):
            st.session_state.current_images = generate_synthetic_demo_pairs("retail_receipt")
            st.session_state.preset_prompt = "I have attached the store invoice and a photo of the product in its box. Can I return this item within the return window?"
            st.rerun()

        if st.button("🚗 Warning Light + OBD", use_container_width=True):
            st.session_state.current_images = generate_synthetic_demo_pairs("automotive")
            st.session_state.preset_prompt = "I attached photos of the dashboard warning light and the OBD-II trouble code scanner. What does this code indicate and is it safe to drive?"
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
# 8. EVIDENCE ATTACHMENT CHAMBER (1 to 2 Images)
# =====================================================================
num_attached = len(st.session_state.current_images)
chamber_expanded = True if num_attached > 0 else False

with st.expander(f"📷 Visual Evidence Chamber ({num_attached} / 2 Images Attached - Capacity: 1 to 2 Images)", expanded=chamber_expanded):
    col_cam, col_upload = st.columns([1, 1])
    
    with col_upload:
        uploaded_files = st.file_uploader(
            "Select 1 or 2 Evidence Photos (PNG, JPG, JPEG, WEBP)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            help="Select 1 or 2 files (e.g. damaged item photo and invoice/serial tag)."
        )
        if uploaded_files:
            if len(uploaded_files) > 2:
                st.warning("⚠️ Maximum 2 images allowed. Only the first 2 images will be attached.", icon="⚠️")
                uploaded_files = uploaded_files[:2]
                
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
                if len(st.session_state.current_images) >= 2:
                    st.warning("⚠️ Maximum 2 images already attached. Please remove an image before capturing another.")
                else:
                    cam_img = Image.open(camera_file)
                    cam_name = f"webcam_angle_{len(st.session_state.current_images) + 1}_{int(time.time())}.png"
                    if not any(img["name"] == cam_name for img in st.session_state.current_images):
                        st.session_state.current_images.append({"image": cam_img, "name": cam_name})
                        st.rerun()

    # Visual Evidence Validation Banner
    curr_count = len(st.session_state.current_images)
    if curr_count == 0:
        st.info("💡 **Evidence Attachment:** You can attach **1 or 2 images** (e.g. damaged product photo and/or invoice receipt) or click a **Demo Preset** on the sidebar.", icon="ℹ️")
    elif 1 <= curr_count <= 2:
        st.success(f"✅ **Evidence Ready:** {curr_count} visual file(s) attached (Within 1 to 2 image limit).", icon="🎯")
    else:
        st.error(f"🚫 **Too many images ({curr_count}):** Please keep attached images between 1 and 2.", icon="🛑")

    # Render Multi-Image Gallery Grid
    if st.session_state.current_images:
        st.markdown("---")
        st.markdown("##### 🔍 Attached Visual Evidence Gallery")
        
        cols = st.columns(max(len(st.session_state.current_images), 1))
        for i, img_data in enumerate(st.session_state.current_images):
            with cols[i % len(cols)]:
                st.markdown(f"<div class='evidence-card'><div class='evidence-tag'>Image {i+1}: {img_data['name']}</div></div>", unsafe_allow_html=True)
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
        - 📸 **Attach 1 or 2 photos** (physical damage, invoice/receipt, serial number tag) in the Evidence Chamber above.
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
    if st.button("🔍 Cross-Verify Claim & Receipt", use_container_width=True):
        preset_triggered = "Can you cross-reference the physical damage with my invoice/receipt to check return or warranty claim eligibility?"
with qc2:
    if st.button("🛠️ Visual Diagnosis", use_container_width=True):
        preset_triggered = "Please inspect all attached evidence photos and provide step-by-step diagnostic troubleshooting."
with qc3:
    if st.button("🧾 Audit Invoice & SKU", use_container_width=True):
        preset_triggered = "Can you extract the invoice number, purchase date, item name, and store details from the attached receipt?"
with qc4:
    if st.button("⚡ Request Rapid RMA / Repair", use_container_width=True):
        preset_triggered = "I have attached photos of the item defect and proof of purchase. Please evaluate whether this qualifies for warranty repair or replacement."

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
        attached_count = len(st.session_state.current_images)
        if attached_count > 2:
            st.error("🛑 Please remove excess images so that at most 2 images are attached.", icon="🛑")
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
            with st.spinner(f"🤖 OmniSupport AI is cross-referencing {attached_count} visual evidence image(s) with active policies..."):
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
