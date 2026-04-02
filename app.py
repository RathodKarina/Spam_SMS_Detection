# import streamlit as st
# import joblib
# import re
# import string

# st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")

# # ── Simple CSS ──
# st.markdown("""
# <style>
# #MainMenu, footer, header { visibility: hidden; }

# .main { background-color: #0f172a; }
# .stApp { background-color: #0f172a; color: #f1f5f9; }

# .stTextArea textarea {
#     background-color: #1e293b !important;
#     color: #f1f5f9 !important;
#     border: 1px solid #334155 !important;
#     border-radius: 10px !important;
#     font-size: 15px !important;
# }

# .stButton > button {
#     background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
#     color: white !important;
#     font-size: 16px !important;
#     font-weight: 600 !important;
#     border: none !important;
#     border-radius: 10px !important;
#     padding: 12px 24px !important;
#     width: 100% !important;
# }

# .stButton > button:hover {
#     opacity: 0.9 !important;
# }
# </style>
# """, unsafe_allow_html=True)


# # ── Load Model ──
# @st.cache_resource
# def load_model():
#     try:
#         model = joblib.load("spam_model.pkl")
#         tfidf = joblib.load("tfidf_vectorizer.pkl")
#         return model, tfidf
#     except:
#         return None, None

# model, tfidf = load_model()


# # ── Clean Text ──
# def clean_text(text):
#     text = text.lower()
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     text = re.sub(r'\d+', '', text)
#     return text.strip()


# # ══════════════════════════════
# #  TITLE
# # ══════════════════════════════
# st.markdown("## 📩 SMS Spam Detector")
# st.markdown("Type a message below to check if it is **Spam** or **Ham (Safe)**.")
# st.markdown("---")


# # ── Model Status ──
# if model is None:
#     st.error("❌ Model not found! Please run the Jupyter Notebook first to create `spam_model.pkl` and `tfidf_vectorizer.pkl`.")
#     st.stop()
# else:
#     st.success("✅ Model loaded successfully!")


# st.markdown("---")


# # ══════════════════════════════
# #  INPUT
# # ══════════════════════════════
# st.markdown("### ✏️ Enter Your Message")
# user_input = st.text_area("", placeholder="Type or paste an SMS here...", height=150)

# predict_btn = st.button("🔍 Check Message")


# # ══════════════════════════════
# #  RESULT
# # ══════════════════════════════
# if predict_btn:
#     if user_input.strip() == "":
#         st.warning("⚠️ Please enter a message first!")
#     else:
#         cleaned  = clean_text(user_input)
#         vec      = tfidf.transform([cleaned])
#         pred     = model.predict(vec)[0]
#         prob     = model.predict_proba(vec)[0]

#         spam_prob = round(prob[1] * 100, 1)
#         ham_prob  = round(prob[0] * 100, 1)
#         conf      = round(prob[pred] * 100, 1)

#         st.markdown("---")
#         st.markdown("### 📊 Result")

#         if pred == 1:
#             st.error(f"🚨 **SPAM DETECTED!**")
#         else:
#             st.success(f"✅ **HAM — This message is Safe!**")

#         st.markdown("---")

#         # ── Confidence ──
#         st.markdown("### 📈 Confidence")
#         st.write(f"**Confidence:** {conf}%")
#         st.progress(int(conf))

#         st.markdown("---")

#         # ── Probabilities ──
#         st.markdown("### 🔢 Probabilities")

#         col1, col2 = st.columns(2)

#         with col1:
#             st.markdown("**🚨 Spam Probability**")
#             st.write(f"{spam_prob}%")
#             st.progress(int(spam_prob))

#         with col2:
#             st.markdown("**✅ Ham Probability**")
#             st.write(f"{ham_prob}%")
#             st.progress(int(ham_prob))

#         st.markdown("---")

#         # ── Message Info ──
#         st.markdown("### 📝 Message Info")

#         col3, col4 = st.columns(2)

#         with col3:
#             st.metric("Total Characters", len(user_input))

#         with col4:
#             st.metric("Total Words", len(user_input.split()))


# # ══════════════════════════════
# #  SAMPLE MESSAGES
# # ══════════════════════════════
# st.markdown("---")
# st.markdown("### 💬 Try Sample Messages")

# samples = [
#     "Congratulations! You won a FREE iPhone. Click now to claim!",
#     "URGENT: Your account is suspended. Call 0800-FREE now!",
#     "Win £1000 cash! Text WIN to 87654. Limited offer!!!",
#     "Hey, are you coming to the party tonight?",
#     "Can you pick up some milk on your way home?",
#     "Meeting at 3pm tomorrow. Please bring your laptop!"
# ]

# selected = st.selectbox("Choose a sample message:", samples)

# if st.button("📋 Load This Sample"):
#     st.info(f"**Loaded:** {selected}")
#     st.write("👆 Copy this message, paste it above, and click **Check Message**!")

# import streamlit as st

import streamlit as st
import joblib
import re
import string
import os

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")

# ── Simple CSS ──
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }

.main { background-color: #0f172a; }
.stApp { background-color: #0f172a; color: #f1f5f9; }

.stTextArea textarea {
    background-color: #1e293b !important;
    color: #f1f5f9 !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    font-size: 15px !important;
}

.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    width: 100% !important;
}

.stButton > button:hover {
    opacity: 0.9 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load Model Safely ──
@st.cache_resource
def load_model():
    try:
        base_dir = os.path.dirname(__file__)
        model_path = os.path.join(base_dir, "spam_model.pkl")
        tfidf_path = os.path.join(base_dir, "tfidf_vectorizer.pkl")

        model = joblib.load(model_path)
        tfidf = joblib.load(tfidf_path)

        return model, tfidf

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None


model, tfidf = load_model()


# ── Clean Text ──
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── Session State for Input ──
if "text" not in st.session_state:
    st.session_state.text = ""


# ══════════════════════════════
# TITLE
# ══════════════════════════════
st.markdown("## 📩 SMS Spam Detector")
st.markdown("Type a message below to check if it is **Spam** or **Ham (Safe)**.")
st.markdown("---")


# ── Model Status ──
if model is None:
    st.stop()
else:
    st.success("✅ Model loaded successfully!")


st.markdown("---")


# ══════════════════════════════
# INPUT
# ══════════════════════════════
st.markdown("### ✏️ Enter Your Message")

user_input = st.text_area(
    "",
    value=st.session_state.text,
    placeholder="Type or paste an SMS here...",
    height=150
)

predict_btn = st.button("🔍 Check Message")


# ══════════════════════════════
# RESULT
# ══════════════════════════════
if predict_btn:

    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message first!")

    else:
        cleaned = clean_text(user_input)
        vec = tfidf.transform([cleaned]) #type: ignore
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        spam_prob = round(prob[1] * 100, 1)
        ham_prob = round(prob[0] * 100, 1)
        conf = round(prob[pred] * 100, 1)

        st.markdown("---")
        st.markdown("### 📊 Result")

        if pred == 1:
            st.error("🚨 **SPAM DETECTED!**")
        else:
            st.success("✅ **HAM — This message is Safe!**")

        st.markdown("---")

        # ── Confidence ──
        st.markdown("### 📈 Confidence")
        st.write(f"**Confidence:** {conf}%")
        st.progress(int(conf))

        if conf > 80:
            st.success("🔥 High confidence prediction!")
        elif conf > 60:
            st.info("🙂 Moderate confidence")
        else:
            st.warning("🤔 Low confidence")

        st.markdown("---")

        # ── Probabilities ──
        st.markdown("### 🔢 Probabilities")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🚨 Spam Probability**")
            st.write(f"{spam_prob}%")
            st.progress(int(spam_prob))

        with col2:
            st.markdown("**✅ Ham Probability**")
            st.write(f"{ham_prob}%")
            st.progress(int(ham_prob))

        st.markdown("---")

        # ── Spam Keyword Detection ──
        st.markdown("### 🧠 Smart Detection")

        spam_words = ["free", "win", "urgent", "offer", "click"]
        found = [word for word in spam_words if word in cleaned]

        if found:
            st.warning(f"⚠️ Suspicious words detected: {', '.join(found)}")
        else:
            st.success("✅ No obvious spam keywords found")

        st.markdown("---")

        # ── Message Info ──
        st.markdown("### 📝 Message Info")

        col3, col4 = st.columns(2)

        with col3:
            st.metric("Total Characters", len(user_input))

        with col4:
            st.metric("Total Words", len(user_input.split()))


# ══════════════════════════════
# SAMPLE MESSAGES
# ══════════════════════════════
st.markdown("---")
st.markdown("### 💬 Try Sample Messages")

samples = [
    "Congratulations! You won a FREE iPhone. Click now to claim!",
    "URGENT: Your account is suspended. Call 0800-FREE now!",
    "Win £1000 cash! Text WIN to 87654. Limited offer!!!",
    "Hey, are you coming to the party tonight?",
    "Can you pick up some milk on your way home?",
    "Meeting at 3pm tomorrow. Please bring your laptop!"
]

selected = st.selectbox("Choose a sample message:", samples)

if st.button("📋 Load This Sample"):
    st.session_state.text = selected
    st.rerun()
