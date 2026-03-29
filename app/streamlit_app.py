import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import sys
import cv2

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.denoiser.dncnn import DnCNN
from models.super_resolution.rrdbnet import RRDBNet
from models.classifier.resnet import Classifier
from evaluation.metrics import calculate_psnr, calculate_ssim

# Page Config
st.set_page_config(
    page_title="CXR Enhancement Suite",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_models():
    """Load models once and cache them"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Denoiser
    denoiser = DnCNN(depth=10, n_channels=64, image_channels=1).to(device)
    denoiser_path = os.path.join("models", "denoiser", "dncnn_best.pth")
    if os.path.exists(denoiser_path):
        denoiser.load_state_dict(torch.load(denoiser_path, map_location=device))
        denoiser.eval()
    else:
        denoiser = None
        st.warning(f"Denoiser weights not found at {denoiser_path}")

    # 2. Load SR Model
    sr_model = RRDBNet(in_channels=1, out_channels=1).to(device)
    sr_path = os.path.join("models", "super_resolution", "generator_latest.pth")
    if os.path.exists(sr_path):
        sr_model.load_state_dict(torch.load(sr_path, map_location=device))
        sr_model.eval()
    else:
        sr_model = None
        st.warning(f"SR Model weights not found at {sr_path}")

    # 3. Load Classifier
    classifier = Classifier(num_classes=2, in_channels=1).to(device)
    cls_path = os.path.join("models", "classifier", "classifier_enhanced.pth")
    if not os.path.exists(cls_path):
        cls_path = os.path.join("models", "classifier", "classifier_degraded.pth")
    
    if os.path.exists(cls_path):
        classifier.load_state_dict(torch.load(cls_path, map_location=device))
        classifier.eval()
    else:
        classifier = None
        # st.warning("Classifier weights not found.") # Optional warning
        
    return denoiser, sr_model, classifier, device

def run_classification(model, img_arr, device):
    """Run ResNet classifier on a numpy image (H, W) uint8"""
    if model is None:
        return "N/A", 0.0

    # Resize to 224x224 (ResNet standard)
    img_resized = cv2.resize(img_arr, (224, 224))
    
    # Normalize (0-1 -> -1 to 1 based on Mean=[0.5], Std=[0.5])
    img_t = img_resized.astype(np.float32) / 255.0
    img_t = (img_t - 0.5) / 0.5
    img_t = torch.from_numpy(img_t).unsqueeze(0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
        
    class_names = {0: "Normal", 1: "Abnormal"}
    return class_names.get(pred_idx, "Unknown"), confidence

def process_image(img, denoiser, sr_model, device):
    """Run the enhancement pipeline"""
    # Preprocess
    img_gray = img.convert('L')
    img_np = np.array(img_gray)
    
    # Degrade manually for demo if needed, or assume input IS degraded.
    # For this demo, we simulate degradation to show restoration capability.
    
    h, w = img_np.shape
    # Resize to 64x64 for model input (simulating low res)
    img_lr = cv2.resize(img_np, (64, 64), interpolation=cv2.INTER_CUBIC)
    
    # Add noise
    noise = np.random.normal(0, 15, img_lr.shape).astype(np.float32)
    img_lr_noisy = np.clip(img_lr + noise, 0, 255).astype(np.uint8)
    
    # Prepare Tensor
    input_tensor = torch.from_numpy(img_lr_noisy / 255.0).float().unsqueeze(0).unsqueeze(0).to(device)
    
    enhanced_np = None
    
    if denoiser and sr_model:
        with torch.no_grad():
            # 1. Denoise
            cleaned_tensor = denoiser(input_tensor)
            
            # 2. Super-Resolution
            enhanced_tensor = sr_model(cleaned_tensor)
            
        enhanced_np = enhanced_tensor.squeeze().cpu().numpy()
        enhanced_np = np.clip(enhanced_np * 255.0, 0, 255).astype(np.uint8)
        
        # Resize output to match original if needed for metrics (it should be 256x256)
        # Our SR model does 4x upscaling. 64x4 = 256.
        if enhanced_np.shape != img_np.shape:
             enhanced_np = cv2.resize(enhanced_np, (w, h))
    else:
        enhanced_np = img_lr_noisy # Fallback

    return img_lr_noisy, enhanced_np, img_np

CUSTOM_CSS = """
<style>
/* ── Global ─────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background-color: #0d1117;
    color: #e6edf3;
}
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}
[data-testid="stHeader"] { background: transparent; }

/* ── Hero Banner ─────────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a3a5c 50%, #0d2137 100%);
    border: 1px solid #1f6feb;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
}
.hero-banner h1 { font-size: 2rem; font-weight: 700; color: #58a6ff; margin: 0; }
.hero-banner p  { color: #8b949e; margin: 0.4rem 0 0; font-size: 0.95rem; }

/* ── Section Headers ─────────────────────────────────────────── */
.section-header {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #58a6ff;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}

/* ── Image Cards ─────────────────────────────────────────────── */
.img-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.img-card-title {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 0.6rem;
}

/* ── Badge Pills ─────────────────────────────────────────────── */
.badge {
    display: inline-block;
    border-radius: 20px;
    padding: 0.35rem 0.9rem;
    font-size: 0.82rem;
    font-weight: 600;
    margin-top: 0.5rem;
}
.badge-normal   { background: #1a4731; color: #3fb950; border: 1px solid #2ea043; }
.badge-abnormal { background: #4a1515; color: #f85149; border: 1px solid #da3633; }
.badge-neutral  { background: #1c2128; color: #8b949e; border: 1px solid #30363d; }

/* ── Metric Cards ────────────────────────────────────────────── */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-label { font-size: 0.72rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.1em; }
.metric-value { font-size: 1.6rem; font-weight: 700; color: #58a6ff; }
.metric-delta { font-size: 0.78rem; color: #3fb950; margin-top: 0.2rem; }

/* ── Pipeline Steps ──────────────────────────────────────────── */
.pipeline-step {
    background: #161b22;
    border-left: 3px solid #1f6feb;
    border-radius: 0 8px 8px 0;
    padding: 0.5rem 0.9rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
    color: #c9d1d9;
}

/* ── Buttons ─────────────────────────────────────────────────── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 0.9rem;
    width: 100%;
    transition: opacity 0.2s;
}
[data-testid="stButton"] > button:hover { opacity: 0.85; }

/* ── Expandable / Divider ────────────────────────────────────── */
hr { border-color: #21262d; }
[data-testid="stExpander"] { border: 1px solid #30363d; border-radius: 8px; }

/* ── Sidebar labels ──────────────────────────────────────────── */
[data-testid="stSidebar"] label { color: #8b949e; font-size: 0.82rem; }
</style>
"""

def _badge(label: str, badge_class: str) -> str:
    return f'<div class="badge {badge_class}">{label}</div>'

def _cls_badge(cls_name: str, confidence: float) -> str:
    cls_lower = cls_name.lower()
    if cls_lower == "normal":
        css = "badge-normal"
    elif cls_lower == "abnormal":
        css = "badge-abnormal"
    else:
        css = "badge-neutral"
    return _badge(f"{cls_name}  {confidence:.1%}", css)


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Hero ────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-banner">
        <h1>🩻 Chest X-Ray Enhancement Suite</h1>
        <p>AI-powered image restoration and pathology classification pipeline &nbsp;·&nbsp;
           DnCNN Denoiser &nbsp;+&nbsp; Real-ESRGAN Super-Resolution &nbsp;+&nbsp; ResNet-18 Classifier</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Models ──────────────────────────────────────────────────
    denoiser, sr_model, classifier, device = load_models()
    if denoiser is None or sr_model is None:
        st.error("⚠️  Enhancement models not found. Check the `models/` directory.")

    # ── Sidebar ─────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Pipeline Settings")
        st.markdown("---")
        option = st.radio("**Input Source**", ["Upload Image", "Use Test Sample"],
                          label_visibility="collapsed")
        st.markdown(f"<small style='color:#8b949e'>Source: <b style='color:#58a6ff'>{option}</b></small>",
                    unsafe_allow_html=True)
        st.markdown("---")
        show_cls = st.checkbox("Show Classification Results", value=True)
        show_heatmap = st.checkbox("Show Error Heatmap", value=False)
        st.markdown("---")
        st.markdown("""
        <div style='font-size:0.78rem; color:#484f58;'>
        <div class="pipeline-step">① Degrade: resize to 64×64 + noise</div>
        <div class="pipeline-step">② Denoise: DnCNN</div>
        <div class="pipeline-step">③ Upscale: Real-ESRGAN (×4)</div>
        <div class="pipeline-step">④ Classify: ResNet-18</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Image Selection ─────────────────────────────────────────
    input_image = None
    if option == "Upload Image":
        st.markdown('<p class="section-header">Upload X-Ray Image</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drag & drop or click to browse",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed"
        )
        if uploaded_file:
            input_image = Image.open(uploaded_file)
    else:
        test_dir = os.path.join("data", "original", "test")
        if os.path.exists(test_dir):
            files = [f for f in os.listdir(test_dir) if f.endswith('.png')]
            if files:
                st.markdown('<p class="section-header">Select Test Sample</p>', unsafe_allow_html=True)
                selected_file = st.selectbox("", files, label_visibility="collapsed")
                input_image = Image.open(os.path.join(test_dir, selected_file))
            else:
                st.error("No images found in `data/original/test`.")
        else:
            st.error("Data directory not found. Run `setup_data.py` first.")

    # ── Run Pipeline ────────────────────────────────────────────
    if input_image is not None:
        st.markdown("---")
        run_col, _ = st.columns([1, 3])
        with run_col:
            run = st.button("▶  Run Enhancement Pipeline")

        if run:
            with st.spinner("Running AI pipeline — please wait…"):
                degraded, enhanced, original = process_image(input_image, denoiser, sr_model, device)

                psnr_val = calculate_psnr(original, enhanced)
                ssim_val = calculate_ssim(original, enhanced)

                cls_orig, conf_orig = run_classification(classifier, original, device)
                cls_deg,  conf_deg  = run_classification(classifier, degraded, device)
                cls_enh,  conf_enh  = run_classification(classifier, enhanced, device)

            # ── Image Grid ──────────────────────────────────────
            st.markdown('<p class="section-header">Visual Results</p>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown('<div class="img-card"><div class="img-card-title">① Degraded Input</div>', unsafe_allow_html=True)
                st.image(degraded, use_container_width=True, clamp=True)
                if show_cls:
                    st.markdown(_cls_badge(cls_deg, conf_deg), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with c2:
                st.markdown('<div class="img-card"><div class="img-card-title">② AI Enhanced</div>', unsafe_allow_html=True)
                st.image(enhanced, use_container_width=True, clamp=True)
                if show_cls:
                    st.markdown(_cls_badge(cls_enh, conf_enh), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with c3:
                st.markdown('<div class="img-card"><div class="img-card-title">③ Ground Truth</div>', unsafe_allow_html=True)
                st.image(original, use_container_width=True, clamp=True)
                if show_cls:
                    st.markdown(_cls_badge(cls_orig, conf_orig), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Metrics ─────────────────────────────────────────
            st.markdown("---")
            st.markdown('<p class="section-header">Image Quality Metrics</p>', unsafe_allow_html=True)
            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">PSNR (Enhanced)</div>
                  <div class="metric-value">{psnr_val:.2f}</div>
                  <div class="metric-delta">dB</div>
                </div>""", unsafe_allow_html=True)
            with mc2:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">SSIM (Enhanced)</div>
                  <div class="metric-value">{ssim_val:.4f}</div>
                  <div class="metric-delta">vs original</div>
                </div>""", unsafe_allow_html=True)
            with mc3:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">Diagnosis (Degraded)</div>
                  <div class="metric-value" style="font-size:1.1rem">{cls_deg}</div>
                  <div class="metric-delta">{conf_deg:.1%} confidence</div>
                </div>""", unsafe_allow_html=True)
            with mc4:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">Diagnosis (Enhanced)</div>
                  <div class="metric-value" style="font-size:1.1rem">{cls_enh}</div>
                  <div class="metric-delta">{conf_enh:.1%} confidence</div>
                </div>""", unsafe_allow_html=True)

            # ── Heatmap ─────────────────────────────────────────
            if show_heatmap:
                st.markdown("---")
                st.markdown('<p class="section-header">Error Heatmap</p>', unsafe_allow_html=True)
                diff = cv2.absdiff(original, enhanced)
                diff_c = cv2.applyColorMap(diff, cv2.COLORMAP_MAGMA)
                hm_col, _ = st.columns([1, 2])
                with hm_col:
                    st.image(diff_c, caption="Per-pixel absolute error  (bright = high error)",
                             use_container_width=True, clamp=True)

    # ── Footer ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#484f58; font-size:0.75rem;'>"
        "CXR Enhancement Suite &nbsp;·&nbsp; Final Year Project &nbsp;·&nbsp; "
        "DnCNN + Real-ESRGAN + ResNet-18</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
