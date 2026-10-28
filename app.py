import json
import os
import tempfile
import threading
from datetime import datetime
from functools import lru_cache

import gradio as gr
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tv_models
from PIL import Image, ImageOps
from torchvision import transforms

DENSENET_REPO_ID = os.getenv("DENSENET_REPO_ID", "Mpvr/autism-densenet169")
QWEN_MODEL_ID = os.getenv("QWEN_MODEL_ID", "Mpvr/autism-qwen3-1.7b")
HF_TOKEN = os.getenv("HF_TOKEN") or None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CLASS_NAMES = ["Autistic", "Non-autistic"]
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
PRELIMINARY_MODELS = [
    ("DenseNet-169", 85.5), ("RegNet X-400MF", 84.5),
    ("MobileNet V3 Large", 83.5), ("DenseNet-121", 83.5),
    ("MobileNet V2", 81.0), ("ViT-B/16", 77.0), ("AlexNet", 50.0),
]

# Models are loaded only when a user opens the relevant feature.
def find_densenet_checkpoint():
    from huggingface_hub import hf_hub_download
    errors = []
    for filename in ("best_densenet169.pth", "densenet169_best.pth"):
        try:
            return hf_hub_download(repo_id=DENSENET_REPO_ID, filename=filename, token=HF_TOKEN)
        except Exception as exc:
            errors.append(f"{filename}: {type(exc).__name__}: {exc}")
    raise FileNotFoundError(
        f"Could not find a DenseNet checkpoint in {DENSENET_REPO_ID}. " + " | ".join(errors)
    )


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format.")
    for key in ("model_state_dict", "state_dict", "model", "weights"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            checkpoint = value
            break
    return {key.removeprefix("module."): value for key, value in checkpoint.items() if isinstance(key, str)}


def checkpoint_class_names(checkpoint):
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


@lru_cache(maxsize=1)
def load_densenet():
    checkpoint_path = find_densenet_checkpoint()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    saved_names = checkpoint_class_names(checkpoint)
    state_dict = extract_state_dict(checkpoint)
    classifier_weight = state_dict.get("classifier.weight")
    if classifier_weight is None:
        raise ValueError("Checkpoint is missing DenseNet classifier weights.")
    num_classes = int(classifier_weight.shape[0])
    model = tv_models.densenet169(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(DEVICE).eval()
    with torch.inference_mode():
        model(torch.zeros(1, 3, 224, 224, device=DEVICE))
    if saved_names and len(saved_names) == num_classes:
        names = saved_names
    elif len(DEFAULT_CLASS_NAMES) == num_classes:
        names = DEFAULT_CLASS_NAMES
    else:
        names = [f"Class {i}" for i in range(num_classes)]
    return model, names, num_classes


def predict_image(image):
    if image is None:
        raise ValueError("Upload an image first.")
    image = ImageOps.exif_transpose(image).convert("RGB")
    model, class_names, num_classes = load_densenet()
    tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    index = int(torch.argmax(probabilities).item())
    return {
        "predicted_class": class_names[index],
        "confidence_percent": round(float(probabilities[index].item()) * 100, 2),
        "class_probabilities": {
            class_names[i]: round(float(probabilities[i].item()) * 100, 2)
            for i in range(num_classes)
        },
        "model": "DenseNet-169",
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "research_use_only": True,
    }


_qwen_lock = threading.Lock()
_qwen_holder = {"tokenizer": None, "model": None, "error": None}


def get_qwen():
    if _qwen_holder["model"] is not None:
        return _qwen_holder["tokenizer"], _qwen_holder["model"]
    with _qwen_lock:
        if _qwen_holder["model"] is not None:
            return _qwen_holder["tokenizer"], _qwen_holder["model"]
        if _qwen_holder["error"] is not None:
            raise RuntimeError(str(_qwen_holder["error"]))
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, token=HF_TOKEN)
            model = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL_ID, token=HF_TOKEN,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
            ).to(DEVICE).eval()
            _qwen_holder.update(tokenizer=tokenizer, model=model)
            return tokenizer, model
        except Exception as exc:
            _qwen_holder["error"] = exc
            raise


def answer_question(message, history):
    from transformers import TextIteratorStreamer
    tokenizer, model = get_qwen()
    messages = [{
        "role": "system",
        "content": (
            "You are a research assistant for an experimental image-classification project. "
            "Help with PyTorch, model evaluation, statistics, and research writing. "
            "Do not claim image predictions are medical diagnoses. Distinguish evidence from "
            "uncertainty and do not invent citations. Keep answers concise."
        ),
    }]
    for item in (history or [])[-8:]:
        if isinstance(item, dict) and item.get("role") in ("user", "assistant"):
            content = item.get("content", "")
            if isinstance(content, str) and content.strip():
                messages.append({"role": item["role"], "content": content})
    messages.append({"role": "user", "content": message})
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=180)

    def generate():
        with torch.inference_mode():
            model.generate(
                **inputs, streamer=streamer, max_new_tokens=300, do_sample=False,
                use_cache=True, pad_token_id=tokenizer.eos_token_id,
            )

    threading.Thread(target=generate, daemon=True).start()
    response = ""
    for piece in streamer:
        response += piece
        yield response


def make_report_files(report):
    if not report:
        return None, None
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix="prediction_", delete=False, encoding="utf-8") as jf:
        json.dump(report, jf, indent=2)
        json_path = jf.name
    rows = [
        ["Predicted class", report.get("predicted_class", "—")],
        ["Confidence (%)", report.get("confidence_percent", "—")],
        ["Model", report.get("model", "—")],
        ["Evaluated at", report.get("evaluated_at", "—")],
        ["Research use only", report.get("research_use_only", "—")],
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", prefix="research_report_", delete=False, encoding="utf-8") as cf:
        pd.DataFrame(rows, columns=["Field", "Value"]).to_csv(cf.name, index=False)
        csv_path = cf.name
    return json_path, csv_path


def run_prediction_ui(image):
    try:
        result = predict_image(image)
        scores = [[label, score] for label, score in result["class_probabilities"].items()]
        json_path, csv_path = make_report_files(result)
        summary = (
            "### Prediction complete\n"
            f"**Predicted class:** {result['predicted_class']}  \n"
            f"**Model confidence:** {result['confidence_percent']:.2f}%  \n"
            f"**Model:** DenseNet-169  \n"
            f"**Evaluated:** {result['evaluated_at']}"
        )
        return summary, result, scores, json_path, csv_path
    except Exception as exc:
        error = f"### Prediction failed\n{type(exc).__name__}: {exc}"
        return error, {}, [], None, None


def chat_submit(message, history):
    if not message or not message.strip():
        yield history or [], ""
        return
    history = list(history or [])
    prior_history = history[:]
    history.append({"role": "user", "content": message})
    yield history, ""
    history.append({"role": "assistant", "content": "Loading Qwen3 and generating a response…"})
    yield history, ""
    try:
        partial = ""
        for partial in answer_question(message, prior_history):
            history[-1] = {"role": "assistant", "content": partial}
            yield history, ""
        if not partial.strip():
            history[-1] = {"role": "assistant", "content": "The model returned an empty response."}
            yield history, ""
    except Exception as exc:
        history[-1] = {
            "role": "assistant",
            "content": f"Qwen could not respond ({type(exc).__name__}: {exc}). Check Space logs and available memory.",
        }
        yield history, ""


def search_records(query):
    records = [
        ["DenseNet-169 checkpoint", "Model", f"Hugging Face repo: {DENSENET_REPO_ID}"],
        ["Qwen3 1.7B", "Language model", f"Hugging Face repo: {QWEN_MODEL_ID}"],
        ["Dataset audit", "Data quality", "Not connected"],
    ]
    if query and query.strip():
        q = query.strip().lower()
        records = [row for row in records if any(q in str(cell).lower() for cell in row)]
    return records


CSS = r"""
:root { --ink:#172033; --line:#e2e8f0; --canvas:#f5f7fb; }
.gradio-container { max-width:1480px !important; margin:0 auto !important; background:var(--canvas) !important; color:var(--ink) !important; }
body,.gradio-container { font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif !important; }
#hero { padding:25px 28px 23px; border:1px solid var(--line); border-radius:18px; background:linear-gradient(120deg,#fff 0%,#f3f6ff 72%,#eef2ff 100%); margin-bottom:16px; }
#brandline { color:#315efb; font-size:12px; font-weight:800; letter-spacing:.15em; text-transform:uppercase; margin-bottom:8px; }
#hero h1 { font-size:clamp(26px,3vw,38px) !important; line-height:1.12 !important; letter-spacing:-.035em; margin:0 0 9px !important; color:#172033 !important; }
#hero p { color:#5b667b !important; margin:0 !important; font-size:14px; }
.pill-row { display:flex; flex-wrap:wrap; gap:8px; margin-top:17px; }
.pill { display:inline-flex; padding:6px 10px; border-radius:999px; border:1px solid #dce4f5; background:#fff; color:#475569; font-size:11px; font-weight:700; }
.pill.blue { background:#edf2ff; border-color:#d9e2ff; color:#315efb; }
#main-tabs { border:1px solid var(--line); background:#fff; border-radius:16px; padding:12px 16px 18px; box-shadow:0 5px 24px rgba(22,34,58,.035); }
#main-tabs button[role=tab] { border-radius:10px !important; font-weight:650 !important; padding:11px 14px !important; }
#main-tabs button[role=tab][aria-selected=true] { background:#edf2ff !important; color:#244de2 !important; border:1px solid #dce5ff !important; }
.panel { border:1px solid var(--line); border-radius:14px; background:#fff; padding:18px; }
.section-title { font-size:18px; font-weight:750; color:#172033; margin-bottom:4px; }
.section-subtitle { color:#69758a; font-size:13px; margin-bottom:16px; }
#footer-note { color:#778196; font-size:12px; text-align:center; padding:14px 0 4px; }
.gradio-container .block { border-color:var(--line) !important; }
.gradio-container button.primary { background:#315efb !important; border-color:#315efb !important; }
.gradio-container button { border-radius:10px !important; }
.gradio-container input,.gradio-container textarea { border-radius:10px !important; }
@media(max-width:800px) { #hero{padding:20px 17px} #main-tabs{padding:8px} #main-tabs button[role=tab]{padding:9px 10px !important;font-size:12px} }
"""

with gr.Blocks(title="PY18 Research Console", theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"), css=CSS) as demo:
    gr.HTML("""
    <header id="hero">
      <div id="brandline">PY18 · Experimental AI Research</div>
      <h1>Research Console</h1>
      <p>A focused workspace for image submissions, model comparison, research assistance, and exportable evaluation records.</p>
      <div class="pill-row">
        <span class="pill blue">DenseNet-169 · Vision</span>
        <span class="pill">Qwen3 1.7B · Assistant</span>
        <span class="pill">Hugging Face model registry</span>
        <span class="pill">Research use only</span>
      </div>
    </header>
    """)
    gr.Markdown("⚠️ **Research use only:** predictions are experimental and must not be used to diagnose autism.", elem_classes=["panel"])

    with gr.Tabs(elem_id="main-tabs"):
        with gr.Tab("Submissions", id="submissions"):
            gr.HTML('<div class="section-title">Image submission</div><div class="section-subtitle">Upload an image to run the saved DenseNet-169 classifier. No dataset ID is required.</div>')
            with gr.Row(equal_height=False):
                with gr.Column(scale=11, elem_classes=["panel"]):
                    gr.Markdown("### Test a new image\nUpload a JPG, JPEG, or PNG. The image is resized and normalized before inference.")
                    input_image = gr.Image(label="Image upload", type="pil", height=330)
                    predict_btn = gr.Button("Predict with DenseNet-169", variant="primary")
                with gr.Column(scale=9, elem_classes=["panel"]):
                    gr.Markdown("### Latest prediction")
                    prediction_summary = gr.Markdown("Upload an image and run a prediction to see results here.")
                    prediction_json = gr.JSON(label="Prediction details")
                    score_table = gr.Dataframe(headers=["Class", "Score (%)"], datatype=["str", "number"], label="Class scores", interactive=False)
                    with gr.Row():
                        download_json = gr.File(label="Prediction JSON", interactive=False)
                        download_csv = gr.File(label="Report CSV", interactive=False)
            gr.Markdown("**Research pipeline**  \n📤 Image upload　→　⚙️ Preprocessing　→　🧠 DenseNet-169　→　📊 Prediction scores")

        with gr.Tab("Compare experiments", id="compare"):
            gr.HTML('<div class="section-title">Experiment comparison</div><div class="section-subtitle">Preliminary accuracy figures supplied for the prototype. Verify against actual evaluation outputs before publication.</div>')
            gr.Dataframe(value=[[n, a] for n, a in PRELIMINARY_MODELS], headers=["Model", "Accuracy (%)"], datatype=["str", "number"], label="Accuracy overview", interactive=False)
            gr.Markdown("### Accuracy overview")
            gr.BarPlot(value=pd.DataFrame(PRELIMINARY_MODELS, columns=["Model", "Accuracy (%)"]), x="Model", y="Accuracy (%)", title="Model accuracy comparison", y_lim=[0, 100], tooltip=["Model", "Accuracy (%)"])

        with gr.Tab("Research assistant", id="assistant"):
            gr.HTML('<div class="section-title">Research assistant · Qwen3 1.7B</div><div class="section-subtitle">Language-model assistance for research, evaluation, PyTorch, statistics, and writing.</div>')
            gr.Markdown("> Qwen is a separate language model; it does not perform the DenseNet image classification.")
            chatbot = gr.Chatbot(label="Research conversation", type="messages", height=420, show_copy_button=True, placeholder="Ask about PyTorch, evaluation metrics, or research methodology.")
            with gr.Row():
                chat_input = gr.Textbox(label="Your question", placeholder="Ask about PyTorch, metrics, or research methodology…", lines=2, scale=8)
                send_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear conversation", variant="secondary")

        with gr.Tab("Reports", id="reports"):
            gr.HTML('<div class="section-title">Reports</div><div class="section-subtitle">Download the latest prediction record in JSON or CSV format.</div>')
            gr.Markdown("Run a prediction on the **Submissions** tab to create a report. Downloads are generated from the current prediction.")
            report_json = gr.File(label="Latest prediction JSON", interactive=False)
            report_csv = gr.File(label="Research report CSV", interactive=False)

        with gr.Tab("Knowledge store", id="knowledge"):
            gr.HTML('<div class="section-title">Knowledge store</div><div class="section-subtitle">Model registry and research artifacts associated with this prototype.</div>')
            knowledge_search = gr.Textbox(label="Search research records", placeholder="Search by record name, type, or status")
            knowledge_table = gr.Dataframe(value=search_records(""), headers=["Record", "Type", "Status"], datatype=["str", "str", "str"], label="Research records", interactive=False)
            gr.Markdown("Dataset audit is not connected in this prototype.")

    gr.HTML('<div id="footer-note">PY18 Research Console · Experimental research prototype · Not validated for clinical diagnosis</div>')

    predict_btn.click(
        run_prediction_ui, inputs=[input_image],
        outputs=[prediction_summary, prediction_json, score_table, download_json, download_csv],
    )
    predict_btn.click(
        lambda image: make_report_files(predict_image(image)) if image is not None else (None, None),
        inputs=[input_image], outputs=[report_json, report_csv],
    )
    send_btn.click(chat_submit, inputs=[chat_input, chatbot], outputs=[chatbot, chat_input])
    chat_input.submit(chat_submit, inputs=[chat_input, chatbot], outputs=[chatbot, chat_input])
    clear_btn.click(lambda: [], outputs=[chatbot])
    knowledge_search.change(search_records, inputs=[knowledge_search], outputs=[knowledge_table])

if __name__ == "__main__":
    demo.queue(max_size=16).launch()
