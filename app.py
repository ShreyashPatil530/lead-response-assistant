"""
Lead Response Assistant – UrbanRoof Assignment
================================================
An AI workflow that reads a customer enquiry and drafts a helpful,
human-sounding reply using the Groq LLM API (Llama 3).

Features
--------
* Understands customer intent
* Asks relevant follow-up / clarifying questions
* Avoids hallucinated claims and false promises
* Sounds natural and empathetic
"""

import os
import streamlit as st
from groq import Groq

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL_ID = "llama-3.3-70b-versatile"

# ──────────────────────────────────────────────
# System prompt — the heart of reliability
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are a customer response assistant for a property inspection company.

Your task is to generate a polite, professional, and human-sounding reply to a customer enquiry.

STRICT RULES (must follow):
- Do NOT assume or guess the cause of the issue
- Do NOT diagnose the problem
- Do NOT mention specific causes (roof, leakage, walls, cracks, mold, inspection, repair, pricing, services) unless the customer explicitly mentions them
- Do NOT offer inspections, callbacks, site visits, or estimates
- Do NOT make promises or guarantees
- Do NOT use emojis or overly casual language

RESPONSE GUIDELINES:
- Acknowledge the customer's concern empathetically
- Ask only neutral, relevant clarifying questions
- Suggest only safe and general next steps (observation, keeping area clear, normal ventilation)
- Avoid technical jargon
- Keep the tone calm, supportive, and professional
- Keep the response concise and clear
- Reply in the same language the customer used

CRITICAL — Response Formatting Rules (YOU MUST FOLLOW):
You MUST format your entire response using Markdown with clear section headers. Do NOT write plain paragraphs without structure.

Use this exact template:

---

**Acknowledgement**

(1-2 sentences acknowledging the customer's concern empathetically)

---

**Clarifying Questions**

To understand your situation better, could you share:

1. (First neutral clarifying question)
2. (Second neutral clarifying question)
3. (Third neutral clarifying question)

---

**Suggested Next Steps**

In the meantime, here are a few things you can do:

- (First safe, general step)
- (Second safe, general step)
- (Third safe, general step)

---

**Closing**

(A warm, professional closing — let them know someone will get back once they share more details)

---
"""

# ──────────────────────────────────────────────
# Groq client
# ──────────────────────────────────────────────
@st.cache_resource
def get_groq_client():
    return Groq(api_key=GROQ_API_KEY)


def generate_response(enquiry: str, tone: str) -> str:
    """Call Groq LLM and return the drafted reply."""
    client = get_groq_client()

    tone_instruction = ""
    if tone == "Friendly & Casual":
        tone_instruction = "Use a friendly, casual tone — like talking to a neighbour."
    elif tone == "Formal & Professional":
        tone_instruction = "Use a formal, professional tone suitable for corporate communication."
    elif tone == "Empathetic & Supportive":
        tone_instruction = "Use an extra empathetic and supportive tone — the customer may be stressed."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + tone_instruction},
        {"role": "user", "content": enquiry},
    ]

    chat_completion = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        temperature=0.5,      # balanced creativity / accuracy
        max_tokens=1024,
        top_p=0.9,
    )
    return chat_completion.choices[0].message.content


# ──────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="UrbanRoof Lead Response Assistant",
        page_icon="🏠",
        layout="wide",
    )

    # ── Custom CSS ─────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header banner */
    .hero {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: #ffffff !important;
        text-align: center;
    }
    .hero h1 {
        color: #ffffff !important;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: #e2e8f0 !important;
        margin: 0.5rem 0 0;
        font-size: 1.05rem;
        opacity: 0.95;
    }

    /* Cards */
    .response-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.8rem 2rem;
        margin-top: 1rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.06);
        line-height: 1.7;
        color: #1a202c;
    }

    .info-card {
        background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-top: 0.6rem;
        font-size: 0.92rem;
        color: #3730a3;
        line-height: 1.6;
    }

    /* Badges */
    .badge-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.6rem;
    }
    .badge {
        background: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
        border-radius: 999px;
        padding: 0.25rem 0.85rem;
        font-size: 0.78rem;
        font-weight: 500;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        font-size: 0.8rem;
        color: #94a3b8;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ─────────────────────────────────
    st.markdown("""
    <div class="hero">
        <h1>🏠 UrbanRoof Lead Response Assistant</h1>
        <p>AI-powered draft replies that are accurate, empathetic, and hallucination-free.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Layout columns ─────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("📩 Customer Enquiry")

        enquiry = st.text_area(
            "Paste the customer message below:",
            height=200,
            placeholder='e.g. "Hi, I am getting damp patches on my bedroom wall after rains. What should I do?"',
        )

        tone = st.selectbox(
            "🎨 Response Tone",
            ["Friendly & Casual", "Formal & Professional", "Empathetic & Supportive"],
            index=0,
        )

        generate_btn = st.button("🚀 Generate Response", type="primary", use_container_width=True)

        # ── How it works panel ─────────────────
        st.markdown("""
        <div class="info-card">
            <strong>How this assistant works:</strong><br/>
            1️⃣ Reads the customer enquiry and identifies intent.<br/>
            2️⃣ Asks relevant clarifying questions.<br/>
            3️⃣ Suggests safe, actionable next steps.<br/>
            4️⃣ Avoids any fabricated claims or guarantees.<br/>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="badge-row">
            <span class="badge">✅ No Hallucinations</span>
            <span class="badge">✅ Intent Detection</span>
            <span class="badge">✅ Follow-up Questions</span>
            <span class="badge">✅ Safe Advice</span>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.subheader("✏️ Drafted Reply")

        if generate_btn:
            if not enquiry.strip():
                st.warning("⚠️ Please enter a customer enquiry first.")
            else:
                with st.spinner("Thinking…"):
                    try:
                        response = generate_response(enquiry, tone)
                        st.session_state["last_response"] = response
                        st.session_state["last_enquiry"] = enquiry
                    except Exception as e:
                        st.error(f"❌ Error from Groq API: {e}")

        # Display latest response (persists across reruns)
        if "last_response" in st.session_state:
            st.markdown('<div class="response-card">', unsafe_allow_html=True)
            st.markdown(st.session_state["last_response"])
            st.markdown('</div>', unsafe_allow_html=True)

            # Copy-friendly text box
            with st.expander("📋 Copy-friendly plain text"):
                st.code(st.session_state["last_response"], language=None)

            # Regenerate button
            if st.button("🔄 Regenerate", use_container_width=True):
                with st.spinner("Regenerating…"):
                    try:
                        response = generate_response(
                            st.session_state.get("last_enquiry", ""),
                            tone,
                        )
                        st.session_state["last_response"] = response
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        else:
            st.info("👈 Enter a customer enquiry on the left and click **Generate Response**.")

    # ── Footer ─────────────────────────────────
    st.markdown("""
    <div class="footer">
        Built with ❤️ by Shreyash Patil · Powered by Groq (Llama 3) · UrbanRoof Assignment
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
