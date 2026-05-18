import json
from pathlib import Path
from queue import Queue
import re
import shutil
from threading import Thread

import gradio as gr
from PIL import Image

from run_llmind import run_llmind_inference


ROOT = Path(__file__).resolve().parent
EXAMPLE_INFO_PATH = ROOT / "example_data" / "info.json"
EXAMPLE_IMAGE_DIR = ROOT / "example_data" / "images"
PRECOMPUTED_LOG_DIR = ROOT / "logs"
APP_LOG_DIR = ROOT / "gradio_logs"
APP_LOG_DIR.mkdir(exist_ok=True)

INPUT_HEADERS = ["Question", "Ground Truth Answers (use | between valid answers)"]
RESULT_HEADERS = [
    "Question",
    "Ground Truth Answers",
    "LLMind Answer",
    "LLMind Correct",
    "Uniform Answer",
    "Uniform Correct",
]
PIXEL_BUDGET_CHOICES = [("1%", "0.01"), ("3%", "0.03"), ("5%", "0.05")]
DEFAULT_PIXEL_BUDGET = "0.01"
ITERATION_CHOICES = [("20", "20"), ("40", "40"), ("60", "60"), ("80", "80"), ("100", "100")]
DEFAULT_ITERATIONS = "20"
IMAGE_SIZE_CHOICES = [("300 x 300", "300"), ("500 x 500", "500")]
DEFAULT_IMAGE_SIZE = "300"


def render_progress_bar(progress):
    clamped = max(0.0, min(1.0, float(progress)))
    width_pct = clamped * 100.0
    return f"""
    <div class="run-progress-track" aria-hidden="true">
        <div class="run-progress-fill" style="width: {width_pct:.2f}%;"></div>
    </div>
    """


def render_run_status(message="", tone="idle"):
    if not message:
        return """
        <div class="run-status-chip run-status-hidden" aria-live="polite">
            <span class="run-status-dot"></span>
            <span>Idle</span>
        </div>
        """

    return f"""
    <div class="run-status-chip run-status-{tone}" aria-live="polite">
        <span class="run-status-dot"></span>
        <span>{message}</span>
    </div>
    """


def normalize_text(text):
    text = (text or "").lower().strip()
    cleaned = []
    for char in text:
        cleaned.append(char if char.isalnum() or char.isspace() else " ")
    return " ".join("".join(cleaned).split())


def sample_lookup_key(name):
    stem = Path(name).stem if name else ""
    digits = re.findall(r"\d+", stem.lower())
    if digits:
        return f"n:{int(''.join(digits))}"

    compact = re.sub(r"[^a-z0-9]+", "", stem.lower())
    return f"s:{compact}"


def sample_display_name(name):
    stem = Path(name).stem if name else ""
    digits = re.findall(r"\d+", stem)
    if digits:
        return f"Image {int(''.join(digits))}"
    return stem.replace("_", " ").strip().title()


def sample_sort_key(sample_id):
    if sample_id.startswith("n:"):
        return (0, int(sample_id.split(":", 1)[1]))
    return (1, sample_id)


def build_sample_asset_index(directory, *, kind):
    entries = {}
    for path in sorted(directory.iterdir()):
        if kind == "file" and not path.is_file():
            continue
        if kind == "dir" and not path.is_dir():
            continue
        entries[sample_lookup_key(path.name)] = path
    return entries


def parse_pixel_budget(value):
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise gr.Error("Choose a valid pixel budget.") from exc


def parse_iterations(value):
    try:
        iterations = int(value)
    except (TypeError, ValueError) as exc:
        raise gr.Error("Choose a valid iteration count.") from exc

    if iterations <= 0:
        raise gr.Error("Choose a valid iteration count.")
    return iterations


def parse_image_size(value):
    try:
        image_size = int(value)
    except (TypeError, ValueError) as exc:
        raise gr.Error("Choose a valid image size.") from exc

    if image_size <= 0:
        raise gr.Error("Choose a valid image size.")
    return image_size


def prediction_matches_references(prediction, references):
    pred_norm = normalize_text(prediction)
    pred_tokens = set(pred_norm.split())
    for ref in references:
        ref_norm = normalize_text(ref)
        if not ref_norm:
            continue
        if ref_norm in pred_tokens or f" {ref_norm} " in f" {pred_norm} ":
            return True
    return False


def load_samples():
    with open(EXAMPLE_INFO_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_index = build_sample_asset_index(EXAMPLE_IMAGE_DIR, kind="file")
    log_index = build_sample_asset_index(PRECOMPUTED_LOG_DIR, kind="dir")

    grouped = {}
    for item in data:
        sample_key = sample_lookup_key(item["image_name"])
        resolved_image = image_index.get(sample_key)
        resolved_log_dir = log_index.get(sample_key)
        sample = grouped.setdefault(
            sample_key,
            {
                "image_path": None if resolved_image is None else str(resolved_image.resolve()),
                "log_dir": None if resolved_log_dir is None else str(resolved_log_dir.resolve()),
                "display_name": sample_display_name(
                    resolved_image.name if resolved_image is not None else item["image_name"]
                ),
                "questions": [],
                "answers": [],
                "rows": [],
            },
        )
        sample["questions"].append(item["question"])
        sample["answers"].append(item["answers"])
        sample["rows"].append([item["question"], " | ".join(item["answers"])])
    return grouped

def validate_samples(samples):
    if not samples:
        raise RuntimeError("No sample data was found in example_data/info.json.")

    missing_images = [sample["display_name"] for sample in samples.values() if not sample["image_path"]]
    if missing_images:
        names = ", ".join(sorted(missing_images))
        raise RuntimeError(f"Missing sample image files for: {names}")

    return samples


SAMPLES = validate_samples(load_samples())
SAMPLE_ORDER = sorted(SAMPLES.keys(), key=sample_sort_key)
DEFAULT_SAMPLE_KEY = SAMPLE_ORDER[0]


def load_sample_inputs(sample_key):
    sample = SAMPLES[sample_key]
    if not sample["image_path"]:
        raise gr.Error(f"No image file found for sample '{sample['display_name']}'.")
    return sample["image_path"], sample["rows"], sample_key


def clear_sample_state():
    return None


def blank_qa_rows():
    return [["", ""]]


def clear_sample_state_with_progress():
    return None, render_progress_bar(0.0)


def clear_uploaded_inputs():
    return None, blank_qa_rows(), render_progress_bar(0.0), render_run_status()


def load_sample_inputs_with_progress(sample_key):
    image_path, rows, selected_sample_key = load_sample_inputs(sample_key)
    return image_path, rows, selected_sample_key, render_progress_bar(0.0), render_run_status()


def parse_qa_rows(rows):
    questions = []
    answers = []

    for idx, row in enumerate(rows or [], start=1):
        row = row or []
        question = str(row[0]).strip() if len(row) > 0 and row[0] is not None else ""
        answer_cell = str(row[1]).strip() if len(row) > 1 and row[1] is not None else ""

        if not question and not answer_cell:
            continue
        if not question or not answer_cell:
            raise gr.Error(f"Row {idx} must include both a question and at least one ground-truth answer.")

        gt_answers = [part.strip() for part in answer_cell.split("|") if part.strip()]
        if not gt_answers:
            raise gr.Error(f"Row {idx} must contain at least one valid ground-truth answer.")

        questions.append(question)
        answers.append(gt_answers)

    if not questions:
        raise gr.Error("Provide an image plus at least one question/answer row.")

    return questions, answers


def is_exact_sample_request(image_path, questions, answers, sample_key):
    if not sample_key or sample_key not in SAMPLES or not image_path:
        return False

    sample = SAMPLES[sample_key]
    try:
        same_image = Path(image_path).resolve() == Path(sample["image_path"]).resolve()
    except FileNotFoundError:
        return False

    return same_image and questions == sample["questions"] and answers == sample["answers"]


def build_result_rows(result):
    rows = []
    questions = result.get("questions", [])
    gt_answers = result.get("gt_answers", [])
    llmind_answers = result.get("llmind_answers", [])
    uniform_answers = result.get("uniform_answers", [])

    for idx, question in enumerate(questions):
        refs = gt_answers[idx] if idx < len(gt_answers) else []
        llmind_answer = llmind_answers[idx] if idx < len(llmind_answers) else ""
        uniform_answer = uniform_answers[idx] if idx < len(uniform_answers) else ""
        rows.append(
            [
                question,
                " | ".join(refs),
                llmind_answer,
                "yes" if prediction_matches_references(llmind_answer, refs) else "no",
                uniform_answer,
                "yes" if prediction_matches_references(uniform_answer, refs) else "no",
            ]
        )
    return rows


def build_accuracy_summary(result):
    questions = result.get("questions", [])
    total = len(questions)
    gt_answers = result.get("gt_answers", [])
    llmind_answers = result.get("llmind_answers", [])
    uniform_answers = result.get("uniform_answers", [])

    llmind_correct = sum(
        prediction_matches_references(pred, refs) for pred, refs in zip(llmind_answers, gt_answers)
    )
    uniform_correct = sum(
        prediction_matches_references(pred, refs) for pred, refs in zip(uniform_answers, gt_answers)
    )

    llmind_accuracy = result.get("llmind_accuracy")
    uniform_accuracy = result.get("uniform_accuracy")

    llmind_line = "Pending"
    uniform_line = "Pending"
    if llmind_accuracy is not None:
        llmind_line = f"{llmind_accuracy * 100:.2f}% ({llmind_correct}/{total})"
    if uniform_accuracy is not None:
        uniform_line = f"{uniform_accuracy * 100:.2f}% ({uniform_correct}/{total})"

    return (
        f"**LLMind accuracy:** {llmind_line}\n\n"
        f"**Uniform accuracy:** {uniform_line}"
    )


def load_cached_sample_result(sample_key, pixel_budget, iterations, image_size):
    if (
        pixel_budget != DEFAULT_PIXEL_BUDGET
        or iterations != DEFAULT_ITERATIONS
        or image_size != DEFAULT_IMAGE_SIZE
    ):
        return None

    sample = SAMPLES[sample_key]
    if not sample["log_dir"]:
        return None

    sample_log_dir = Path(sample["log_dir"])
    results_path = sample_log_dir / "results.json"
    llmind_path = sample_log_dir / "llmind_sampled.png"
    uniform_path = sample_log_dir / "uniform_sampled.png"
    if not all(path.exists() for path in (results_path, llmind_path, uniform_path)):
        return None

    with open(results_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    return (
        build_accuracy_summary(result),
        build_result_rows(result),
        str(llmind_path.resolve()),
        str(uniform_path.resolve()),
    )


def load_image_copy(image_path):
    with Image.open(image_path) as image:
        return image.copy()


def remove_gradio_logs():
    if APP_LOG_DIR.exists():
        shutil.rmtree(APP_LOG_DIR, ignore_errors=True)


def progress_only_update(progress, status_html):
    return (
        render_progress_bar(progress),
        status_html,
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
    )


def run_app(image_path, qa_rows, sample_key, pixel_budget, iterations, image_size):
    if not image_path:
        raise gr.Error("Upload an image or choose one of the provided sample images.")
    if not Path(image_path).is_file():
        raise gr.Error("The selected image file could not be found.")

    percentage = parse_pixel_budget(pixel_budget)
    epoch_count = parse_iterations(iterations)
    parsed_image_size = parse_image_size(image_size)
    questions, answers = parse_qa_rows(qa_rows)

    if is_exact_sample_request(image_path, questions, answers, sample_key):
        cached_result = load_cached_sample_result(sample_key, pixel_budget, iterations, image_size)
        if cached_result is not None:
            yield (
                render_progress_bar(1.0),
                render_run_status("LLMind Finished", tone="done"),
                *cached_result,
            )
            return

    event_queue = Queue()

    def on_progress(step, total_steps):
        total = max(1, int(total_steps))
        event_queue.put(("progress", min(float(step) / total, 1.0)))

    def run_inference():
        try:
            run_outputs = run_llmind_inference(
                image_path=image_path,
                questions=questions,
                answers=answers,
                log_dir=str(APP_LOG_DIR),
                exp_name=Path(image_path).stem,
                overrides={"percentage": percentage, "epochs": epoch_count, "img_size": parsed_image_size},
                progress_callback=on_progress,
            )
            event_queue.put(("done", run_outputs))
        except Exception as exc:
            event_queue.put(("error", exc))

    Thread(target=run_inference, daemon=True).start()

    running_status = render_run_status("LLMind Running...", tone="running")
    yield progress_only_update(0.0, running_status)

    latest_progress = 0.0
    while True:
        event_type, payload = event_queue.get()

        if event_type == "progress":
            latest_progress = max(latest_progress, payload)
            yield progress_only_update(latest_progress, running_status)
            continue

        if event_type == "error":
            yield progress_only_update(latest_progress, render_run_status("LLMind Failed", tone="error"))
            remove_gradio_logs()
            raise payload

        result = payload["results"]
        llmind_image_value = load_image_copy(payload["llmind_path"])
        uniform_image_value = load_image_copy(payload["uniform_path"])
        remove_gradio_logs()
        yield (
            render_progress_bar(1.0),
            render_run_status("LLMind Finished", tone="done"),
            build_accuracy_summary(result),
            build_result_rows(result),
            llmind_image_value,
            uniform_image_value,
        )
        return


default_image, default_rows, default_sample_state = load_sample_inputs(DEFAULT_SAMPLE_KEY)


APP_CSS = """
body .gradio-container {
    width: 100% !important;
    max-width: none !important;
    padding-left: 12px !important;
    padding-right: 12px !important;
}

#top-row {
    min-height: 0 !important;
    max-height: 380px !important;
    align-items: flex-start !important;
    margin-bottom: 8px !important;
}

#top-row > div {
    min-height: 0 !important;
}

#sample-title {
    margin-top: 0 !important;
    margin-bottom: 6px !important;
}

#sample-grid .gradio-image,
#sample-grid .image-container {
    min-height: 0 !important;
}

#sample-grid {
    margin-top: 0 !important;
}

#sample-grid-row {
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    gap: 8px !important;
}

#sample-grid-row > div {
    min-width: 180px !important;
}

#run-controls-right {
    gap: 8px !important;
}

#run-progress-slot {
    min-height: 18px !important;
}

#run-status-slot {
    min-height: 26px !important;
}

#run-progress-slot .run-progress-track {
    width: 100%;
    height: 10px;
    background: #2b2c31;
    border: 1px solid #3f4148;
    border-radius: 999px;
    overflow: hidden;
}

#run-progress-slot .run-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #ff8a2b 0%, #ff5a00 100%);
    border-radius: inherit;
    transition: width 0.15s ease;
}

#run-status-slot .run-status-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    min-height: 26px;
    padding: 4px 12px;
    border-radius: 999px;
    border: 1px solid transparent;
    font-size: 13px;
    font-weight: 600;
}

#run-status-slot .run-status-dot {
    width: 8px;
    height: 8px;
    border-radius: 999px;
    background: currentColor;
    flex: 0 0 auto;
}

#run-status-slot .run-status-running {
    color: #ffb36b;
    background: rgba(255, 111, 0, 0.14);
    border-color: rgba(255, 111, 0, 0.32);
}

#run-status-slot .run-status-running .run-status-dot {
    animation: run-status-pulse 1.1s ease-in-out infinite;
}

#run-status-slot .run-status-done {
    color: #91d9a2;
    background: rgba(44, 140, 74, 0.16);
    border-color: rgba(44, 140, 74, 0.34);
}

#run-status-slot .run-status-error {
    color: #ff9a9a;
    background: rgba(178, 48, 48, 0.16);
    border-color: rgba(178, 48, 48, 0.34);
}

#run-status-slot .run-status-hidden {
    opacity: 0;
    pointer-events: none;
}

@keyframes run-status-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.45; transform: scale(0.8); }
}
"""


with gr.Blocks(title="LLMind Demo", fill_width=True, css=APP_CSS) as demo:
    gr.Markdown(
        """
        # LLMind : Bio-inspired Training-free Adaptive Visual Representations for Vision-Language Models
        Choose one of the  sample images or upload your own image. The backbone is qwen2.5-vl-3b-instruct.
        """
    )

    sample_state = gr.State(default_sample_state)

    with gr.Row(variant="compact", elem_id="top-row", max_height=380):
        image_input = gr.Image(
            type="filepath",
            label="Input Image",
            value=default_image,
            buttons=[],
            height=360,
            scale=1,
        )
        qa_input = gr.Dataframe(
            headers=INPUT_HEADERS,
            datatype=["str", "str"],
            type="array",
            row_count=(len(default_rows), "dynamic"),
            col_count=(2, "fixed"),
            value=default_rows,
            label="Questions And Ground Truth Answers",
            wrap=True,
            max_height=360,
            scale=2,
        )

    gr.Markdown("### Sample Images", elem_id="sample-title")

    sample_buttons = []
    with gr.Row(variant="compact", elem_id="sample-grid-row"):
        for sample_key in SAMPLE_ORDER:
            with gr.Column(min_width=180):
                gr.Image(
                    value=SAMPLES[sample_key]["image_path"],
                    label=SAMPLES[sample_key]["display_name"],
                    interactive=False,
                    buttons=[],
                    height=110,
                )
                sample_buttons.append((gr.Button(f"Use {SAMPLES[sample_key]['display_name']}"), sample_key))

    with gr.Row(variant="compact"):
        with gr.Column(scale=1, min_width=180):
            pixel_budget_input = gr.Dropdown(
                choices=PIXEL_BUDGET_CHOICES,
                value=DEFAULT_PIXEL_BUDGET,
                label="Pixel Budget",
                interactive=True,
            )
        with gr.Column(scale=1, min_width=180):
            iterations_input = gr.Dropdown(
                choices=ITERATION_CHOICES,
                value=DEFAULT_ITERATIONS,
                label="Iterations",
                interactive=True,
            )
        with gr.Column(scale=1, min_width=180):
            image_size_input = gr.Dropdown(
                choices=IMAGE_SIZE_CHOICES,
                value=DEFAULT_IMAGE_SIZE,
                label="Image Size",
                interactive=True,
            )
        with gr.Column(scale=3, elem_id="run-controls-right"):
            run_button = gr.Button("Run LLMind", variant="primary")
            progress_bar = gr.HTML(value=render_progress_bar(0.0), elem_id="run-progress-slot")
            run_status = gr.HTML(value=render_run_status(), elem_id="run-status-slot")

    with gr.Row(variant="compact"):
        llmind_image = gr.Image(label="LLMind Sampled Image", buttons=[], height=320)
        uniform_image = gr.Image(label="Uniform Sampled Image", buttons=[], height=320)

    accuracy_output = gr.Markdown()
    results_output = gr.Dataframe(
        headers=RESULT_HEADERS,
        datatype=["str", "str", "str", "str", "str", "str"],
        type="array",
        row_count=(1, "dynamic"),
        col_count=(len(RESULT_HEADERS), "fixed"),
        interactive=False,
        label="Answers And Per-Question Comparison",
        wrap=True,
        max_height=420,
    )

    image_input.upload(
        fn=clear_uploaded_inputs,
        outputs=[sample_state, qa_input, progress_bar, run_status],
        show_progress="hidden",
    )

    for button, sample_key in sample_buttons:
        button.click(
            fn=lambda current_key=sample_key: load_sample_inputs_with_progress(current_key),
            outputs=[image_input, qa_input, sample_state, progress_bar, run_status],
            show_progress="hidden",
        )

    pixel_budget_input.change(
        fn=lambda: (render_progress_bar(0.0), render_run_status()),
        outputs=[progress_bar, run_status],
        show_progress="hidden",
    )
    iterations_input.change(
        fn=lambda: (render_progress_bar(0.0), render_run_status()),
        outputs=[progress_bar, run_status],
        show_progress="hidden",
    )
    image_size_input.change(
        fn=lambda: (render_progress_bar(0.0), render_run_status()),
        outputs=[progress_bar, run_status],
        show_progress="hidden",
    )

    run_button.click(
        fn=run_app,
        inputs=[image_input, qa_input, sample_state, pixel_budget_input, iterations_input, image_size_input],
        outputs=[progress_bar, run_status, accuracy_output, results_output, llmind_image, uniform_image],
        show_progress="hidden",
    )


if __name__ == "__main__":
    demo.queue().launch()
