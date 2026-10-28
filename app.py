
import io
import json
import os
import threading
from datetime import datetime

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision import transforms
from PIL import Image, ImageOps


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PY18 Research Console",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

DENSENET_REPO_ID = os.getenv(
    "DENSENET_REPO_ID", "Mpvr/autism-densenet169"
)
QWEN_MODEL_ID = os.getenv(
    "QWEN_MODEL_ID", "Mpvr/autism-qwen3-1.7b"
)
HF_TOKEN = os.getenv("HF_TOKEN") or None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CLASS_NAMES = ["Autistic", "Non-autistic"]

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

PAGES = [
    "Submissions",
    "Compare experiments",
    "Research assistant",
    "Reports",
    "Knowledge store",
]


# =========================================================
# UI STYLE
# =========================================================

st.markdown(
    """
    <style>
    :root {
        color-scheme: light dark;
    }

    .stApp {
        background: var(--st-background-color, transparent);
        color: var(--st-text-color, inherit);
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stMain"] {
        background: transparent;
        color: inherit;
    }

    [data-testid="stMain"] .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .stApp h1, .stApp h2, .stApp h3,
    .stApp h4, .stApp h5, .stApp h6 {
        color: var(--st-heading-color, var(--st-text-color, inherit));
        font-weight: 700;
        overflow-wrap: anywhere;
    }

    .stApp p, .stApp li, .stApp label {
        color: var(--st-text-color, inherit);
        overflow-wrap: anywhere;
    }

    .stApp a {
        color: var(--st-primary-color, #3984ff);
    }

    [data-testid="stSidebar"] {
        background: var(
            --st-secondary-background-color,
            var(--st-background-color, transparent)
        );
        border-right: 1px solid var(--st-border-color, transparent);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stSidebar"] label {
        color: var(--st-text-color, inherit);
    }

    .stButton > button,
    .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] > button {
        border-radius: 9px;
        font-weight: 600;
        min-height: 2.6rem;
        transition: border-color 0.15s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        border-color: var(--st-primary-color, #3984ff);
    }

    .stButton > button:focus-visible,
    .stDownloadButton > button:focus-visible,
    [data-testid="stFormSubmitButton"] > button:focus-visible {
        outline: 2px solid var(--st-primary-color, #3984ff);
        outline-offset: 3px;
    }

    .stApp input,
    .stApp textarea,
    .stApp [data-baseweb="select"] > div,
    .stApp [data-baseweb="input"] > div,
    .stApp [data-baseweb="textarea"] > div {
        color: var(--st-text-color, inherit);
        border-color: var(--st-border-color, #888);
    }

    .stApp input::placeholder,
    .stApp textarea::placeholder {
        color: var(--st-text-color, inherit);
        opacity: 0.7;
    }

    div[data-testid="stMetric"] {
        background: var(
            --st-secondary-background-color,
            var(--st-background-color, transparent)
        );
        color: var(--st-text-color, inherit);
        border: 1px solid var(--st-border-color, #888);
        border-radius: 12px;
        padding: 16px;
        min-width: 0;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: var(--st-text-color, inherit);
        overflow-wrap: anywhere;
    }

    [data-testid="stChatMessage"] {
        border: 1px solid var(--st-border-color, transparent);
        border-radius: 12px;
        padding: 1rem;
        color: var(--st-text-color, inherit);
    }

    [data-testid="stChatMessage"] p {
        color: var(--st-text-color, inherit);
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border-color: var(--st-border-color, #888);
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    .stApp pre,
    .stApp code {
        border-radius: 8px;
        overflow-wrap: anywhere;
        white-space: pre-wrap;
    }

    [data-testid="stFileUploader"] {
        border-radius: 10px;
    }

    [data-testid="stFileUploader"] section {
        border-color: var(--st-border-color, #888);
    }

    .stApp hr {
        border-color: var(--st-border-color, #888);
        opacity: 0.65;
    }

    .stApp img {
        max-width: 100%;
        height: auto;
        border-radius: 8px;
    }

    @media (max-width: 768px) {
        [data-testid="stMain"] .block-container {
            padding: 1rem 0.8rem 2rem;
        }

        div[data-testid="stMetric"] {
            padding: 12px;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            transition: none !important;
            animation: none !important;
            scroll-behavior: auto !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "Submissions"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Welcome to PY18 Research Console. "
                "I can help discuss model evaluation, "
                "research methodology, and reporting."
            ),
        }
    ]

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "report" not in st.session_state:
    st.session_state.report = None


# =========================================================
# CHECKPOINT HELPERS
# =========================================================

def find_densenet_checkpoint():
    """Download a DenseNet checkpoint from the configured Hugging Face repo."""
    from huggingface_hub import hf_hub_download

    errors = []
    for filename in ("best_densenet169.pth", "densenet169_best.pth"):
        try:
            return hf_hub_download(
                repo_id=DENSENET_REPO_ID,
                filename=filename,
                token=HF_TOKEN,
            )
        except Exception as exc:
            errors.append(f"{filename}: {type(exc).__name__}: {exc}")

    raise FileNotFoundError(
        "Could not download a DenseNet checkpoint from "
        f"{DENSENET_REPO_ID}. Checked both expected filenames. "
        + " | ".join(errors)
    )


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format.")

    for key in ("model_state_dict", "state_dict", "model", "weights"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            checkpoint = value
            break

    cleaned = {}
    for key, value in checkpoint.items():
        if isinstance(key, str):
            cleaned[key.removeprefix("module.")] = value

    return cleaned


def find_checkpoint_class_names(checkpoint):
    if not isinstance(checkpoint, dict):
        return None

    for key in ("class_names", "classes", "label_classes"):
        value = checkpoint.get(key)
        if isinstance(value, (list, tuple)) and value:
            return [str(item) for item in value]

    encoder = checkpoint.get("label_encoder")
    if encoder is not None and hasattr(encoder, "classes_"):
        return [str(item) for item in encoder.classes_]

    return None


# =========================================================
# LOAD DENSENET-169
# =========================================================

@st.cache_resource(show_spinner="Loading DenseNet-169...")
def load_densenet():
    checkpoint_path = find_densenet_checkpoint()

    if checkpoint_path is None:
        raise FileNotFoundError(
            "No DenseNet-169 checkpoint could be downloaded. "
            "Check the Hugging Face repository and uploaded checkpoint filenames."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    saved_class_names = find_checkpoint_class_names(checkpoint)
    state_dict = extract_state_dict(checkpoint)

    classifier_weight = state_dict.get("classifier.weight")
    if classifier_weight is None:
        raise ValueError(
            "The checkpoint does not contain the expected "
            "DenseNet classifier weights."
        )

    num_classes = int(classifier_weight.shape[0])

    model = tv_models.densenet169(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(DEVICE)
    model.eval()

    # Warm-up pass so the first real prediction is fast.
    with torch.inference_mode():
        model(torch.zeros(1, 3, 224, 224, device=DEVICE))

    if saved_class_names and len(saved_class_names) == num_classes:
        class_names = saved_class_names
    elif len(DEFAULT_CLASS_NAMES) == num_classes:
        class_names = DEFAULT_CLASS_NAMES
    else:
        class_names = [f"Class {i}" for i in range(num_classes)]

    return model, class_names, num_classes


def predict_image(image):
    model, class_names, num_classes = load_densenet()

    tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())

    return {
        "predicted_class": class_names[predicted_index],
        "confidence_percent": round(
            float(probabilities[predicted_index].item()) * 100, 2
        ),
        "class_probabilities": {
            class_names[i]: round(float(probabilities[i].item()) * 100, 2)
            for i in range(num_classes)
        },
        "model": "DenseNet-169",
        "research_use_only": True,
    }


def run_prediction(image):
    """Run DenseNet on an image and store the result in session state."""

    try:
        with st.spinner("Running image classification..."):
            result = predict_image(image)

    except FileNotFoundError:
        st.error(
            "DenseNet-169 checkpoint not found. "
            "Check the Hugging Face repository and checkpoint filenames."
        )
        return

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        st.error("GPU memory was insufficient to run prediction.")
        return

    except Exception as exc:
        st.error(
            f"Prediction failed: {type(exc).__name__}: {exc}"
        )
        return

    st.session_state.prediction = result
    st.session_state.report = {
        **result,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }
    st.rerun()


# =========================================================
# QWEN3-1.7B (BACKGROUND LOAD + STREAMING)
# =========================================================

@st.cache_resource
def get_qwen_holder():
    holder = {
        "ready": threading.Event(),
        "tokenizer": None,
        "model": None,
        "error": None,
    }

    def _load():
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            # Download/cache the model files from Hugging Face Hub instead of
            # expecting the Kaggle-only /kaggle/input path to exist at runtime.
            tokenizer = AutoTokenizer.from_pretrained(
                QWEN_MODEL_ID,
                token=HF_TOKEN,
            )

            use_cuda = torch.cuda.is_available()

            model = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL_ID,
                token=HF_TOKEN,
                torch_dtype=torch.float16 if use_cuda else torch.float32,
                low_cpu_mem_usage=True,
            )

            model = model.to(DEVICE)
            model.eval()

            holder["tokenizer"] = tokenizer
            holder["model"] = model

        except Exception as exc:
            holder["error"] = exc

        finally:
            holder["ready"].set()

    threading.Thread(target=_load, daemon=True).start()

    return holder


def stream_qwen(holder, question, conversation):
    from transformers import TextIteratorStreamer

    holder["ready"].wait()

    if holder["error"] is not None:
        raise holder["error"]

    tokenizer = holder["tokenizer"]
    model = holder["model"]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant for an experimental "
                "image-classification project. Help with PyTorch, "
                "model evaluation, statistics, and research writing. "
                "Do not claim that an image prediction is a medical "
                "diagnosis. Clearly distinguish evidence from "
                "uncertainty. Do not invent research citations. "
                "Keep answers concise."
            ),
        }
    ]

    for message in conversation[-4:]:
        if message["role"] in ("user", "assistant"):
            messages.append({
                "role": message["role"],
                "content": message["content"],
            })

    messages.append({"role": "user", "content": question})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=120,
    )

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=300,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    def _generate():
        with torch.inference_mode():
            model.generate(**generation_kwargs)

    threading.Thread(target=_generate, daemon=True).start()

    for piece in streamer:
        yield piece


# =========================================================
# STARTUP ORDER: DenseNet first, then Qwen in background
# =========================================================

try:
    load_densenet()
except Exception:
    pass

QWEN_HOLDER = get_qwen_holder()


# =========================================================
# PAGE FUNCTIONS
# =========================================================

def page_submissions():
    st.title("Image submission")
    st.caption(
        "Upload a new image to run the saved DenseNet-169 classifier. "
        "No dataset ID is required."
    )

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        render_upload_panel()

    with right:
        render_prediction_panel()

    st.divider()
    st.subheader("Research pipeline")
    st.markdown(
        "📤 Image upload  →  ⚙️ Preprocessing  →  "
        "🧠 DenseNet-169  →  📊 Prediction scores"
    )


def render_upload_panel():
    st.subheader("Test a new image")

    uploaded = st.file_uploader(
        "Choose a JPG, JPEG, or PNG image",
        type=["jpg", "jpeg", "png"],
        key="submission_image",
    )

    if uploaded is None:
        st.info(
            "Upload any supported image, including an image "
            "that was not part of your original dataset."
        )
        return

    try:
        image = Image.open(io.BytesIO(uploaded.getvalue()))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception:
        st.error("Unable to read this image. Try a valid JPG or PNG.")
        return

    st.image(image, caption="Uploaded image preview", width="stretch")

    clicked = st.button(
        "Predict with DenseNet-169",
        type="primary",
        width="stretch",
    )

    if clicked:
        run_prediction(image)


def render_prediction_panel():
    st.subheader("Latest prediction")

    result = st.session_state.prediction

    if not result:
        st.info(
            "Upload an image and run a prediction "
            "to see the results here."
        )
        return

    st.metric("Predicted class", result["predicted_class"])
    st.metric("Model confidence", f'{result["confidence_percent"]:.2f}%')

    st.caption(
        "Confidence is the model's softmax score, "
        "not a measure of clinical certainty."
    )

    st.markdown("**Class scores**")

    scores = pd.DataFrame([
        {"Class": label, "Score (%)": score}
        for label, score in result["class_probabilities"].items()
    ])

    st.dataframe(scores, width="stretch", hide_index=True)

    st.download_button(
        "Export prediction",
        data=json.dumps(result, indent=2),
        file_name="prediction_result.json",
        mime="application/json",
        width="stretch",
    )


def page_compare():
    st.title("Experiment comparison")
    st.caption("Previously supplied preliminary accuracy results.")

    models = [
        ("DenseNet-169", 85.5),
        ("RegNet X-400MF", 84.5),
        ("MobileNet V3 Large", 83.5),
        ("DenseNet-121", 83.5),
        ("MobileNet V2", 81.0),
        ("ViT-B/16", 77.0),
        ("AlexNet", 50.0),
    ]

    top_columns = st.columns(4)
    for index, (name, accuracy) in enumerate(models[:4]):
        with top_columns[index]:
            st.metric(name, f"{accuracy:.2f}%")

    bottom_columns = st.columns(3)
    for index, (name, accuracy) in enumerate(models[4:]):
        with bottom_columns[index]:
            st.metric(name, f"{accuracy:.2f}%")

    st.caption(
        "Verify these preliminary figures against your actual "
        "evaluation outputs before reporting them as results."
    )

    df = pd.DataFrame(models, columns=["Model", "Accuracy (%)"])

    st.subheader("Accuracy overview")
    st.bar_chart(df.set_index("Model"), y="Accuracy (%)")


def page_assistant():
    st.title("Research assistant — Qwen3 1.7B")
    st.caption(
        "Local language-model assistance for research and evaluation."
    )
    st.info(
        "Qwen is a separate language model. It does not perform "
        "the DenseNet image classification."
    )

    if QWEN_HOLDER["error"] is not None:
        st.error(
            f"Qwen failed to load: "
            f"{type(QWEN_HOLDER['error']).__name__}: "
            f"{QWEN_HOLDER['error']}"
        )
    elif not QWEN_HOLDER["ready"].is_set():
        st.caption(
            "Qwen is still loading in the background. "
            "Your first message will wait until it is ready."
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input(
        "Ask about PyTorch, metrics, or research methodology..."
    )

    if prompt:
        handle_chat(prompt)

    if st.button("Clear conversation"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Conversation cleared. What would you like to discuss?"
                ),
            }
        ]
        st.rerun()


def handle_chat(prompt):
    history = list(st.session_state.messages)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        try:
            response = st.write_stream(
                stream_qwen(QWEN_HOLDER, prompt, history)
            )
        except Exception as exc:
            st.error(
                f"Qwen could not generate a response "
                f"({type(exc).__name__}: {exc})."
            )
            return

    text = response.strip() if isinstance(response, str) else str(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": text or "The model returned an empty response.",
    })


def page_reports():
    st.title("Reports")
    st.caption("Export the latest DenseNet prediction.")

    report = st.session_state.report

    if report is None:
        st.info(
            "No prediction report is available yet. "
            "Run a prediction on the Submissions page first."
        )
        return

    # Present the report as a readable table rather than a raw JSON block.
    report_rows = [
        {"Field": "Predicted class", "Value": report.get("predicted_class", "—")},
        {"Field": "Confidence (%)", "Value": report.get("confidence_percent", "—")},
        {"Field": "Model", "Value": report.get("model", "—")},
        {"Field": "Evaluated at", "Value": report.get("evaluated_at", "—")},
        {"Field": "Research use only", "Value": report.get("research_use_only", "—")},
    ]

    st.subheader("Prediction summary")
    st.dataframe(pd.DataFrame(report_rows), width="stretch", hide_index=True)

    class_probabilities = report.get("class_probabilities", {})
    if class_probabilities:
        st.subheader("Class probabilities")
        probability_table = pd.DataFrame(
            [{"Class": label, "Score (%)": score}
             for label, score in class_probabilities.items()]
        )
        st.dataframe(probability_table, width="stretch", hide_index=True)

    st.download_button(
        "Download report as CSV",
        data=pd.DataFrame(report_rows).to_csv(index=False),
        file_name="py18_prediction_report.csv",
        mime="text/csv",
        type="primary",
    )


def page_knowledge():
    st.title("Knowledge store")
    st.caption("Research records and evaluation artifacts.")

    search = st.text_input(
        "Search research records",
        placeholder="Search by record name, type, or status",
    )

    densenet_status = (
        "Available" if find_densenet_checkpoint() else "Not found"
    )
    qwen_status = f"Hugging Face repo: {QWEN_MODEL_ID}"

    records = pd.DataFrame([
        {
            "Record": "DenseNet-169 checkpoint",
            "Type": "Model",
            "Status": densenet_status,
        },
        {
            "Record": "Qwen3 1.7B",
            "Type": "Language model",
            "Status": qwen_status,
        },
        {
            "Record": "Dataset audit",
            "Type": "Data quality",
            "Status": "Not connected",
        },
    ])

    if search.strip():
        mask = records.astype(str).apply(
            lambda column: column.str.contains(
                search,
                case=False,
                na=False,
                regex=False,
            )
        ).any(axis=1)
        records = records[mask]

    st.dataframe(records, width="stretch", hide_index=True)


# =========================================================
# SIDEBAR + ROUTER
# =========================================================

with st.sidebar:
    st.markdown("## 🧪 PY18")
    st.markdown("### Research Console")
    st.caption("Researcher workspace")
    st.divider()

    for page_name in PAGES:
        is_active = st.session_state.page == page_name

        if st.button(
            page_name,
            key=f"nav_{page_name}",
            width="stretch",
            type="primary" if is_active else "secondary",
        ):
            st.session_state.page = page_name
            st.rerun()

    st.divider()
    st.caption(
        "Experimental research prototype. "
        "Not validated for clinical diagnosis."
    )


st.warning(
    "Research use only: These predictions are experimental "
    "and must not be used to diagnose autism."
)

ROUTES = {
    "Submissions": page_submissions,
    "Compare experiments": page_compare,
    "Research assistant": page_assistant,
    "Reports": page_reports,
    "Knowledge store": page_knowledge,
}

ROUTES[st.session_state.page]()
