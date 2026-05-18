import os
import re
import string
import warnings

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

try:
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
except Exception:
    pass

import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import (
    AutoProcessor, AutoTokenizer, AutoConfig,
    AutoModelForCausalLM, GenerationConfig
)
from typing import Optional, List, Tuple
from transformers import logging
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

try:
    from transformers import AutoModelForVision2Seq as AutoVisionModel
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoVisionModel

def get_VLMScorer(name, device=None):
    assert device is not None, "Please specify device (e.g., 'cuda' or 'cpu')"
    name = name.lower()
    if name in ["qwen", "qwen2.5-vl-instruct", "qwen2.5-vl-3b-instruct"]:
        return VLMScorer_QWEN2(device=device)
    raise ValueError(f"Unknown VLM scorer: {name}")

CONTRACTIONS = {
    "aint":"ain't","arent":"aren't","cant":"can't","couldve":"could've","couldnt":"couldn't",
    "didnt":"didn't","doesnt":"doesn't","dont":"don't","hadnt":"hadn't","hasnt":"hasn't",
    "havent":"haven't","hed":"he'd","hes":"he's","howd":"how'd","howll":"how'll","hows":"how's",
    "id":"i'd","im":"i'm","ive":"i've","isnt":"isn't","itd":"it'd","itll":"it'll","lets":"let's",
    "maam":"ma'am","mightnt":"mightn't","mustnt":"mustn't","neednt":"needn't","shant":"shan't",
    "shed":"she'd","shes":"she's","shouldve":"should've","shouldnt":"shouldn't","somebodys":"somebody's",
    "someones":"someone's","thats":"that's","theres":"there's","theyd":"they'd","theyll":"they'll",
    "theyre":"they're","theyve":"they've","wasnt":"wasn't","we'd":"we'd","we're":"we're","weve":"we've",
    "werent":"weren't","whatd":"what'd","whatll":"what'll","whats":"what's","whens":"when's",
    "whered":"where'd","wheres":"where's","whod":"who'd","wholl":"who'll","whos":"who's",
    "wont":"won't","wouldve":"would've","wouldnt":"wouldn't","yall":"y'all","youd":"you'd",
    "youll":"you'll","youre":"you're","youve":"you've"
}
ARTICLES = {"a", "an", "the"}
NUMBER_MAP = {
    "zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5",
    "six":"6","seven":"7","eight":"8","nine":"9","ten":"10"
}
_punct_tbl = str.maketrans({p: " " for p in string.punctuation})
DEFAULT_VQA_SYSTEM_PROMPT = (
    "You are a visual question answering assistant. "
    "The image may be blurry, low-quality, or partially unclear. "
    "Infer the most probable answer from the visible context and common real-world knowledge. "
    "Give only a short, direct answer without explanation. "
    "If uncertain, provide the most likely answer."
)

def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.translate(_punct_tbl)
    s = re.sub(r"\s+", " ", s)
    tokens = []
    for w in s.split():
        w = CONTRACTIONS.get(w, w)
        if w in NUMBER_MAP:
            w = NUMBER_MAP[w]
        if w not in ARTICLES:
            tokens.append(w)
    return " ".join(tokens)

class VLMScorer_QWEN2:
    def __init__(
        self,
        vlm_name="Qwen/Qwen2.5-VL-3B-Instruct",
        device="cuda",
        system_prompt=DEFAULT_VQA_SYSTEM_PROMPT,
    ):
        assert device is not None, "Please specify device (e.g., 'cuda' or 'cpu')"
        self.device = device
        self.system_prompt = system_prompt
        self.torch_dtype = torch.float16 if str(device).startswith("cuda") else torch.float32

        self.processor = AutoProcessor.from_pretrained(vlm_name)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(vlm_name, use_fast=True)

        self.vlm = AutoVisionModel.from_pretrained(
            vlm_name, torch_dtype=self.torch_dtype
        ).to(device).eval()

        self.text_encoder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        ).to(device).eval()

        self.gen_cfg = GenerationConfig(
            do_sample=False,
            num_beams=1,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            max_new_tokens=40,
            repetition_penalty=1.0,
            length_penalty=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )

    def _build_messages(self, image_pil, question: str):
        return [
            {"role": "system", "content": [
                {"type": "text", "text": self.system_prompt}
            ]},
            {"role": "user", "content": [
                {"type": "image", "image": image_pil},
                {"type": "text", "text": question}
            ]},
        ]

    @torch.no_grad()
    def predict_answer(self, image_pil, question: str, **gen_kwargs) -> str:
        if "max_new_tokens" in gen_kwargs:
            self.gen_cfg.max_new_tokens = int(gen_kwargs["max_new_tokens"])

        messages = self._build_messages(image_pil, question)

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        out_ids = self.vlm.generate(
            **inputs,
            generation_config=self.gen_cfg
        )

        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = out_ids[0][prompt_len:]
        pred_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        pred_text = pred_text.replace(".<|im_end|>", "").replace("<|im_end|>", "")
        return normalize_text(pred_text)
    
    @torch.no_grad()
    def predict_answer_tensor(self, image_tensor, question: str, **gen_kwargs) -> str:
        image_pil = self.tensor_to_pil(image_tensor)

        if "max_new_tokens" in gen_kwargs:
            self.gen_cfg.max_new_tokens = int(gen_kwargs["max_new_tokens"])

        messages = self._build_messages(image_pil, question)

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        out_ids = self.vlm.generate(
            **inputs,
            generation_config=self.gen_cfg
        )

        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = out_ids[0][prompt_len:]
        pred_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        pred_text = pred_text.replace(".<|im_end|>", "").replace("<|im_end|>", "")
        return normalize_text(pred_text)
    
    def tensor_to_pil(self, t):
        from torchvision import transforms
        if t.dim() == 4:
            t = t[0]
        if t.max() > 1.0:
            t = t / 255.0
        to_pil = transforms.ToPILImage()
        return to_pil(t.detach().cpu().clamp(0, 1))

    @torch.no_grad()
    def __call__(self, image_pil, question, gt_answer):
        pred_text = self.predict_answer(image_pil, question)
        emb_pred = self.text_encoder.encode(
            pred_text, convert_to_tensor=True, device=self.device, normalize_embeddings=True
        )
        emb_gt = self.text_encoder.encode(
            gt_answer, convert_to_tensor=True, device=self.device, normalize_embeddings=True
        )
        sim = util.cos_sim(emb_pred, emb_gt).mean()
        return 1 - sim
