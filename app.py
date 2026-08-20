"""
OmniSupport AI: Multimodal Visual Customer Support & Diagnostic Agent
=====================================================================
A production-grade, multimodal customer support platform powered by
Google Gemini (Dynamic Multimodal Engine & Fallback Ladder) & Streamlit.

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
    page_title="OmniSupport AI | Visual Support & Diagnostics",
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
    if "current_image" not in st.session_state:
        st.session_state.current_image = None
    if "current_image_name" not in st.session_state:
        st.session_state.current_image_name = None
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
# 3. SYNTHETIC PRESET GENERATOR (For Instant 1-Click Multimodal Demos)
# =====================================================================
@st.cache_data(show_spinner=False)
def generate_synthetic_demo_image(scenario_type: str) -> Tuple[Image.Image, str]:
    """
    Generates high-contrast visual diagnostic test fixtures programmatically
    so hackathon judges can test vision capabilities without needing local files.
    """
    img = Image.new("RGB", (700, 420), color=(245, 247, 250))
    draw = ImageDraw.Draw(img)

    if scenario_type == "electronics":
        # Draw tablet device with visible screen crack & serial number
        draw.rounded_rectangle([(150, 40), (550, 380)], radius=20, fill=(30, 41, 59), outline=(100, 116, 139), width=4)
        draw.rectangle([(170, 60), (530, 340)], fill=(15, 23, 42))
        
        draw.text((220, 90), "OmniTech TabPro 11 [POWER ON]", fill=(226, 232, 240))
        draw.text((220, 120), "Model: OT-900X | SN: SN-88349-B2", fill=(148, 163, 184))
        
        # Screen fracture lines
        fracture_color = (239, 68, 68)
        crack_points = [(260, 180), (320, 230), (390, 210), (450, 270), (490, 260)]
        for i in range(len(crack_points) - 1):
            draw.line([crack_points[i], crack_points[i+1]], fill=fracture_color, width=3)
        draw.line([(320, 230), (300, 290)], fill=fracture_color, width=2)
        draw.line([(390, 210), (420, 170)], fill=fracture_color, width=2)
        
        draw.rectangle([(230, 295), (470, 325)], fill=(220, 38, 38))
        draw.text((245, 302), "⚠️ HARDWARE FAULT: LCD MATRIX CRACK", fill=(255, 255, 255))
        name = "demo_electronics_damaged_screen.png"

    elif scenario_type == "retail_receipt":
        # Draw a purchase receipt
        draw.rectangle([(160, 20), (540, 400)], fill=(255, 255, 255), outline=(203, 213, 225), width=2)
        draw.text((240, 40), "APEX STORE RECEIPT", fill=(15, 23, 42))
        draw.text((240, 60), "Store #104 - Order #INV-94021", fill=(100, 116, 139))
        draw.line([(180, 85), (520, 85)], fill=(203, 213, 225), width=1)
        
        draw.text((180, 100), "Date of Purchase: 12-Days Ago", fill=(51, 65, 85))
        draw.text((180, 125), "Payment Method: Visa Ending in 4192", fill=(51, 65, 85))
        draw.text((180, 150), "Item: UltraNoiseCancelling Headset Gen 2", fill=(15, 23, 42))
        draw.text((180, 175), "SKU: 7729104 | Qty: 1 | Price: $249.99", fill=(15, 23, 42))
        draw.text((180, 200), "Condition: Unopened / Sealed in Box", fill=(16, 185, 129))
        draw.line([(180, 230), (520, 230)], fill=(203, 213, 225), width=1)
        
        draw.text((180, 245), "Return Policy: 30-Day Money Back Guarantee", fill=(100, 116, 139))
        draw.text((180, 270), "Barcode: ||| |||| || ||||| |||| |||", fill=(15, 23, 42))
        name = "demo_retail_receipt_inv94021.png"

    elif scenario_type == "pharmacy":
        # Draw prescription medication label
        draw.rectangle([(150, 40), (550, 380)], fill=(254, 243, 199), outline=(217, 119, 6), width=3)
        draw.rectangle([(170, 60), (530, 130)], fill=(30, 58, 138))
        draw.text((220, 75), "CAREPLUS PHARMACY RX #67849", fill=(255, 255, 255))
        draw.text((220, 100), "Dr. Emily Hayes, MD | Refills: 2", fill=(191, 219, 254))
        
        draw.text((180, 150), "Medication: AMOXICILLIN 500MG CAPSULES", fill=(15, 23, 42))
        draw.text((180, 180), "Dosage: Take 1 capsule by mouth every 8 hours", fill=(180, 83, 9))
        draw.text((180, 210), "Warning: Finish full prescribed course of 10 days.", fill=(220, 38, 38))
        draw.text((180, 240), "Storage: Store at 20°C to 25°C. Avoid moisture.", fill=(71, 85, 105))
        draw.text((180, 270), "Lot: #L88392 | Exp: 12/2026", fill=(71, 85, 105))
        name = "demo_pharmacy_prescription_rx67849.png"

    else:
        # Automotive Dashboard Alarm
        draw.rectangle([(120, 50), (580, 370)], fill=(17, 24, 39), outline=(75, 85, 99), width=4)
        draw.text((200, 80), "VEHICLE TELEMATICS DIAGNOSTIC", fill=(243, 244, 246))
        draw.circle((350, 200), 55, fill=(239, 68, 68, 50), outline=(239, 68, 68), width=4)
        draw.text((315, 185), "CHECK\nENGINE", fill=(254, 202, 202))
        draw.text((220, 280), "Error Code: P0420 - Catalyst System Low", fill=(251, 191, 36))
        draw.text((220, 310), "Mileage: 42,150 mi | Status: Active Alarm", fill=(156, 163, 175))
        name = "demo_automotive_obd_p0420.png"

    return img, name


# =====================================================================
# 4. DYNAMIC MODEL DISCOVERY & MULTIMODAL REASONING ENGINE
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
    """
    Dynamically discovers and ranks available models supported by the provided API key.
    Falls back gracefully to the candidate ladder if list_models is restricted.
    """
    try:
        os.environ["GOOGLE_API_KEY"] = api_key
        genai.configure(api_key=api_key, transport="rest")
        
        discovered = []
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                clean_name = m.name.replace("models/", "")
                discovered.append(clean_name)
        
        if discovered:
            # Custom sorting priority: 2.5 flash > 2.0 flash > 1.5 flash > 1.5 pro > other gemini
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
            
            sorted_models = sorted(discovered, key=model_priority)
            return sorted_models
    except Exception:
        pass
    
    return CANDIDATE_MODEL_LADDER


def get_system_prompt(policy: Dict[str, Any]) -> str:
    """Constructs the operational prompt adhering to active warranty and support policies."""
    return f"""
You are OmniSupport AI, an elite Principal Multimodal Customer Support & Visual Diagnostic Agent.
Your objective is to provide compassionate, precise, and highly actionable customer support while visually inspecting customer evidence (products, screens, defects, receipts, labels, invoices).

ACTIVE COMPANY POLICIES & BUSINESS RULES:
- Standard Return Window: {policy.get('return_window_days', 30)} days from purchase.
- Accidental Damage Coverage: {'INCLUDED in policy' if policy.get('covers_accidental') else 'EXCLUDED (Requires warranty upgrade or supervisor approval unless manufacturer defect)'}.
- Proof of Purchase Requirement: {'MANDATORY (Receipt, invoice, or serial number required)' if policy.get('require_receipt') else 'OPTIONAL'}.
- Fast Replacement Protocol: {'ENABLED for verified structural defects' if policy.get('allow_fast_replacement') else 'STANDARD REPAIR QUEUE ONLY'}.

CORE RESPONSIBILITIES:
1. MULTIMODAL VISUAL DIAGNOSIS: When an image is provided, carefully inspect the visual evidence. Detect cracks, burns, defect types, serial numbers, order IDs, or expiration dates. Point them out clearly.
2. POLICY & WARRANTY EVALUATION: Evaluate if the issue qualifies for return, free replacement, warranty repair, or standard troubleshooting.
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
    image: Optional[Image.Image],
    chat_history: List[Dict[str, Any]],
    policy: Dict[str, Any]
) -> Tuple[str, Dict[str, Any], str]:
    """
    Executes a multimodal generation call with dynamic discovery and sequential model fallback ladder.
    Returns (response_text, metadata_dict, model_used).
    """
    if not api_key:
        raise ValueError("Google Gemini API Key is missing. Please enter it in the sidebar or .env file.")

    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key, transport="rest")
    
    system_instruction = get_system_prompt(policy)
    
    # 1. Discover models dynamically
    model_candidates = resolve_available_models(api_key)
    
    # Construct history context
    recent_history = chat_history[-6:]
    
    # Multimodal image handling
    current_parts = []
    if image is not None:
        if image.mode != "RGB":
            image = image.convert("RGB")
        current_parts.append(image)
        current_parts.append(f"[Customer attached an image for visual inspection].\nCustomer Message: {user_prompt}")
    else:
        current_parts.append(user_prompt)

    last_error = None
    
    # 2. Iterate through candidate ladder
    for model_name in model_candidates:
        try:
            # Attempt to initialize model with system instruction
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
                # Fallback for models that do not support system_instruction in constructor
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
            # If 404 (model not supported on this endpoint) or unsupported feature, continue ladder
            if "404" in err_str or "not found" in err_str.lower() or "not supported" in err_str.lower() or "400" in err_str:
                continue
            else:
                # If API key is fundamentally unauthorized or exhausted, raise immediately
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
    
    ticket = f"""# 🛡️ OmniSupport AI - Official Customer Claim & Diagnostic Ticket
**Ticket Reference ID:** `{st.session_state.session_id}`
**Generated Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
**Active AI Engine:** `{st.session_state.active_model_name}`
**Session Duration:** {mins}m {secs}s
**Total Interactions:** {len(st.session_state.messages)} messages

---

## 📊 1. DIAGNOSTIC ASSESSMENT & TRIAGE
- **Customer Sentiment:** {st.session_state.sentiment}
- **Triage Urgency:** {st.session_state.urgency}
- **Detected Issue Category:** {st.session_state.detected_issue}
- **Claim Eligibility Status:** {st.session_state.claim_status}
- **Visual Evidence Attached:** {'Yes (' + st.session_state.current_image_name + ')' if st.session_state.current_image else 'No direct visual attachment'}

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
        ticket += f"\n### Turn {idx} - {sender}\n{msg['content']}\n"
        
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
    
    # API Key Handling (Env or Manual Input)
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
    st.markdown("### 🧪 Quick Demo Presets")
    st.caption("Load pre-configured multimodal scenarios with instant test images:")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button("📱 Cracked LCD", use_container_width=True):
            img, name = generate_synthetic_demo_image("electronics")
            st.session_state.current_image = img
            st.session_state.current_image_name = name
            st.session_state.preset_prompt = "My tablet screen suddenly cracked with vertical lines. Is this covered under warranty for replacement?"
            st.rerun()
            
        if st.button("💊 Rx Label", use_container_width=True):
            img, name = generate_synthetic_demo_image("pharmacy")
            st.session_state.current_image = img
            st.session_state.current_image_name = name
            st.session_state.preset_prompt = "Can you check my prescription dosage instructions and storage guidelines from this bottle photo?"
            st.rerun()

    with col_p2:
        if st.button("🛍️ Receipt Audit", use_container_width=True):
            img, name = generate_synthetic_demo_image("retail_receipt")
            st.session_state.current_image = img
            st.session_state.current_image_name = name
            st.session_state.preset_prompt = "I bought these headphones 12 days ago. Can I return them for a full refund with this receipt?"
            st.rerun()

        if st.button("🚗 Engine Fault", use_container_width=True):
            img, name = generate_synthetic_demo_image("automotive")
            st.session_state.current_image = img
            st.session_state.current_image_name = name
            st.session_state.preset_prompt = "This warning light came on my dashboard with error P0420. Is it safe to drive, and what needs fixing?"
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
        st.session_state.current_image = None
        st.session_state.current_image_name = None
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
    <div class="hero-subtitle">Multimodal Visual Diagnostics & Intelligent Customer Support Agent Powered by Gemini 1.5 Flash</div>
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
# 8. MULTIMODAL ATTACHMENT TRAY
# =====================================================================
with st.expander("📷 Visual Evidence Attachment & Inspection Chamber", expanded=bool(st.session_state.current_image)):
    col_cam, col_upload = st.columns([1, 1])
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Customer Photo / Invoice / Error Log (PNG, JPG, WEBP)",
            type=["png", "jpg", "jpeg", "webp"],
            help="Attach visual evidence for automated optical defect analysis."
        )
        if uploaded_file is not None:
            try:
                st.session_state.current_image = Image.open(uploaded_file)
                st.session_state.current_image_name = uploaded_file.name
            except Exception as e:
                st.error(f"Error loading uploaded image: {e}")

    with col_cam:
        enable_cam = st.toggle("📸 Capture Live via Webcam", value=False)
        if enable_cam:
            camera_file = st.camera_input("Take a photo of the product or receipt")
            if camera_file is not None:
                st.session_state.current_image = Image.open(camera_file)
                st.session_state.current_image_name = f"webcam_capture_{int(time.time())}.png"

    # Display Active Image Thumbnail & Removal Option
    if st.session_state.current_image is not None:
        st.markdown("---")
        img_col1, img_col2 = st.columns([1, 3])
        with img_col1:
            st.image(st.session_state.current_image, caption=f"Evidence: {st.session_state.current_image_name}", use_container_width=True)
        with img_col2:
            st.info(f"🔍 **Active Visual Evidence Attached:** `{st.session_state.current_image_name}`\n\nThe AI will inspect this visual feed alongside your inquiry.", icon="👁️")
            if st.button("🗑️ Detach / Clear Image", key="clear_img_btn"):
                st.session_state.current_image = None
                st.session_state.current_image_name = None
                st.rerun()


# =====================================================================
# 9. CHAT HISTORY DISPLAY
# =====================================================================
# Welcome message if chat is empty
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🛡️"):
        st.markdown("""
        **Hello! I'm OmniSupport AI**, your multimodal visual support and warranty specialist.
        
        How can I help you today?
        - 📸 **Upload or snap a photo** of any defective item, receipt, or error code above.
        - 💬 Ask a question or use the **Quick Prompts** below or **Presets** in the sidebar.
        """)

# Render all existing messages
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🛡️"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg.get("image_attached"):
            st.caption(f"📎 Attached Image: `{msg['image_attached']}`")
        st.markdown(msg["content"])


# =====================================================================
# 10. QUICK-PROMPT ACTION CHIPS
# =====================================================================
st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
qc1, qc2, qc3, qc4 = st.columns(4)

preset_triggered = None
with qc1:
    if st.button("🔍 Check Return Eligibility", use_container_width=True):
        preset_triggered = "Can you verify if my item and receipt qualify for a return or replacement under current policy?"
with qc2:
    if st.button("🛠️ Troubleshoot My Device", use_container_width=True):
        preset_triggered = "Please inspect my product image and provide step-by-step diagnostic troubleshooting."
with qc3:
    if st.button("🧾 Audit Invoice & SKU", use_container_width=True):
        preset_triggered = "Can you extract the order number, item name, purchase date, and total from this receipt?"
with qc4:
    if st.button("⚡ Request Rapid RMA", use_container_width=True):
        preset_triggered = "The item is defective on arrival. Please expedite a replacement RMA authorization ticket."

# Check if a preset prompt was loaded from the sidebar
if "preset_prompt" in st.session_state and st.session_state.preset_prompt:
    preset_triggered = st.session_state.preset_prompt
    st.session_state.preset_prompt = None


# =====================================================================
# 11. CHAT INPUT & EXECUTION PIPELINE
# =====================================================================
chat_input_text = st.chat_input("Type your message or ask OmniSupport AI to inspect your visual evidence...")

# Determine prompt source (chat box or quick button)
final_prompt = chat_input_text if chat_input_text else preset_triggered

if final_prompt:
    if not active_api_key:
        st.error("🔑 Please provide a valid Google Gemini API Key in the sidebar or .env file to proceed.", icon="🚨")
    else:
        user_entry = {
            "role": "user",
            "content": final_prompt,
            "image_attached": st.session_state.current_image_name if st.session_state.current_image else None
        }
        st.session_state.messages.append(user_entry)
        
        with st.chat_message("user", avatar="👤"):
            if user_entry["image_attached"]:
                st.caption(f"📎 Attached Image: `{user_entry['image_attached']}`")
            st.markdown(final_prompt)
            
        with st.chat_message("assistant", avatar="🛡️"):
            with st.spinner("🤖 OmniSupport AI is performing multimodal inspection & policy evaluation..."):
                try:
                    response_text, meta, model_used = query_gemini_with_fallback(
                        api_key=active_api_key,
                        user_prompt=final_prompt,
                        image=st.session_state.current_image,
                        chat_history=st.session_state.messages[:-1],
                        policy=st.session_state.policy_config
                    )
                    
                    # Update live state from agent metadata & active model
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
                    
                    # Instant rerun to synchronize top metric cards & active model badge
                    st.rerun()
                    
                except Exception as err:
                    err_msg = str(err)
                    if "API_KEY_INVALID" in err_msg or "invalid api key" in err_msg.lower() or "403" in err_msg:
                        st.error("❌ The provided Google Gemini API Key is invalid or unauthorized. Please verify your key at https://aistudio.google.com/.", icon="🔑")
                    elif "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
                        st.error("⏳ Gemini API rate limit / quota exceeded. Please wait a moment before trying again.", icon="⏱️")
                    else:
                        st.error(f"⚠️ Service error: {err_msg}", icon="💥")
