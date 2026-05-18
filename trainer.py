import json, os, random, re, string, math
from typing import Dict, List, Union

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm.auto import tqdm

from mobius_layers import apply_mobius_transform_torch, apply_inverse_mobius_transform_torch
from budget_sampler import uniform_sample
from losses import build_scorer
from utils_io import *
from vlm_scorer import *
from sentence_transformers import SentenceTransformer, util

def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compute_vqa_contains_accuracy(
    predictions: List[str],
    references: List[Union[str, List[str]]]
) -> Dict[str, float]:
    assert len(predictions) == len(references)
    norm_refs: List[List[str]] = []
    for r in references:
        if isinstance(r, list):
            norm_refs.append([normalize_text(x) for x in r if str(x).strip()])
        else:
            norm_refs.append([normalize_text(r)])
    correct = 0
    report = {"correct_index": [], "wrong_index": []}
    for i, (pred, refs) in enumerate(zip(predictions, norm_refs)):
        pred_norm = normalize_text(pred)
        pred_tokens = set(pred_norm.split())
        hit = any((ref in pred_tokens) or (f" {ref} " in f" {pred_norm} ") for ref in refs)
        correct += int(hit)
        (report["correct_index"] if hit else report["wrong_index"]).append(i)
    return {"accuracy": correct / len(predictions), "correct": correct, "n": len(predictions), "report": report}


class Trainer:
    def __init__(self, args, model, progress_callback=None):
        self.args = args
        self.progress_callback = progress_callback
        self.model = model.to(self.device)

        self.scorer = build_scorer(args.scorer, device=self.device)
        self.IMAGE = load_image_as_tensor(self.args.image_path, device=self.device, img_size=self.args.img_size)
        self._run_dir, vanilla_sampled, self._uniform_path, self._llmind_path, self._results_path = initialize_run_outputs(
            args, self.IMAGE
        )
        self.vlm_scorer = get_VLMScorer(args.vlm_model, device=self.device)
        raw_questions = getattr(self.args, "questions", None)
        raw_answers = getattr(self.args, "answers", None)
        self.output_questions = list(raw_questions) if isinstance(raw_questions, list) else self._load_questions(raw_questions)
        self.output_answers = self._load_answers_raw(raw_answers)

        self.questions = self._load_questions(raw_questions)
        self.answers = self._load_answers(raw_answers)
        if len(self.questions) != len(self.answers):
            raise ValueError(f"#questions ({len(self.questions)}) != #answers ({len(self.answers)})")

        self.vanilla_acc = self._get_pred_report_from_pil(tensor_to_pil(vanilla_sampled))
        self._write_results(self.vanilla_acc)
        self.text_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(self.device)
        self.text_encoder.eval()

        self.ref_texts = [[normalize_text(x) for x in refs if str(x).strip()] for refs in self.answers]
        with torch.no_grad():
            self.ref_embeds = [
                self.text_encoder.encode(refs, convert_to_tensor=True, device=self.device, normalize_embeddings=True)
                if len(refs) > 0 else torch.empty(0, 384, device=self.device)
                for refs in self.ref_texts
            ]

        self.spsa_every      = int(getattr(self.args, "spsa_every", 10))
        self.spsa_min_norm     = float(getattr(self.args, "spsa_min_norm", 1e-5))
        self.beta_spsa         = float(getattr(self.args, "beta_spsa", 0.5))
        self.spsa_sigma        = float(getattr(self.args, "spsa_sigma", 0.1))
        self.k_dirs            = int(getattr(self.args, "spsa_k_dirs", 8))
        self.spsa_samples      = int(getattr(self.args, "spsa_samples", 1))
        self.spsa_temp         = float(getattr(self.args, "spsa_temp", 0.0))
        self.spsa_top_p        = float(getattr(self.args, "spsa_top_p", 1.0))
        self.spsa_do_sample    = bool(getattr(self.args, "spsa_do_sample", False))
        self.spsa_q_fraction   = float(getattr(self.args, "spsa_q_fraction", 0.5))
        self.bandit_eta        = float(getattr(self.args, "bandit_eta", 0.2))
        self._q_weights        = torch.ones(len(self.questions), device="cpu")
        self._hard_q           = None
        self.clip_recon        = bool(getattr(self.args, "clip_recon", True))
        self.target_img_loss   = float(getattr(self.args, "target_img_loss", 0.0))
        self.eval_every        = 1

    def _normalize_q(self, t: str) -> str:
        t = (t or "").strip().lower()
        t = re.sub(r"\s+", " ", t)
        return t

    def _load_questions(self, qsrc) -> List[str]:
        if qsrc is None:
            q = getattr(self.args, "question", None)
            if q is None:
                raise ValueError("No questions provided. Set args.questions or args.question.")
            qsrc = q
        if isinstance(qsrc, list):
            return [self._normalize_q(x) for x in qsrc]
        if isinstance(qsrc, str) and os.path.isfile(qsrc):
            if qsrc.endswith(".json"):
                with open(qsrc, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "questions" in data:
                    data = data["questions"]
            else:
                with open(qsrc, "r", encoding="utf-8") as f:
                    data = [line.strip() for line in f if line.strip()]
            return [self._normalize_q(x) for x in data]
        parts = [s for s in str(qsrc).split(",") if s.strip()]
        return [self._normalize_q(s) for s in parts]

    def _load_answers(self, asrc) -> List[List[str]]:
        if asrc is None:
            raise ValueError("args.answers is required and must align with args.questions.")
        if isinstance(asrc, list):
            return [[normalize_text(a) for a in (x if isinstance(x, list) else [x]) if str(a).strip()] for x in asrc]
        if isinstance(asrc, str) and os.path.isfile(asrc):
            with open(asrc, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "answers" in data:
                data = data["answers"]
            return [a if isinstance(a, list) else [a] for a in data]
        raise ValueError("args.answers must be a list (of lists) or a path to JSON containing 'answers'")

    def _load_answers_raw(self, asrc) -> List[List[str]]:
        if asrc is None:
            raise ValueError("args.answers is required and must align with args.questions.")
        if isinstance(asrc, list):
            return [[str(a) for a in (x if isinstance(x, list) else [x]) if str(a).strip()] for x in asrc]
        if isinstance(asrc, str) and os.path.isfile(asrc):
            with open(asrc, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "answers" in data:
                data = data["answers"]
            return [[str(a) for a in (x if isinstance(x, list) else [x]) if str(a).strip()] for x in data]
        raise ValueError("args.answers must be a list (of lists) or a path to JSON containing 'answers'")

    @property
    def device(self):
        requested = getattr(self.args, "device", "cpu")
        if requested in (None, "", "auto"):
            if torch.cuda.is_available():
                return torch.device("cuda")
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is not None and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        try:
            return torch.device(requested)
        except Exception:
            print(f"[warn] invalid device '{requested}', falling back to 'cpu'")
            return torch.device("cpu")

    @torch.no_grad()
    def _text_loss(self, recon, q_idx_subset=None, K=1, temperature=0.0, top_p=1.0, do_sample=False) -> float:
        idxs = list(q_idx_subset) if q_idx_subset is not None else list(range(len(self.questions)))
        if len(idxs) == 0:
            return 0.0

        losses = []
        for qi in idxs:
            preds = []
            for _ in range(max(1, K)):
                pred = self.vlm_scorer.predict_answer_tensor(
                    recon, self.questions[qi],
                    temperature=temperature, top_p=top_p, do_sample=do_sample
                )
                preds.append(normalize_text(pred))

            pred_embs = self.text_encoder.encode(
                preds, convert_to_tensor=True, device=self.device, normalize_embeddings=True
            )
            refE = self.ref_embeds[qi]
            if refE.numel() == 0:
                losses.append(1.0)
            else:
                sims = util.cos_sim(pred_embs, refE).max(dim=1).values
                d = (1.0 - sims).mean().item()
                losses.append(float(d))
        return float(sum(losses) / max(1, len(losses)))

    @torch.no_grad()
    def _get_pred_report_from_pil(self, image_pil):
        preds = [self.vlm_scorer.predict_answer(image_pil, q, temperature=0.0, top_p=1.0, do_sample=False) for q in self.questions]
        acc_dict = compute_vqa_contains_accuracy(preds, self.answers)
        return {
            "questions": self.questions,
            "gt_answers": self.answers,
            "pred_answers": preds,
            "accuracy": acc_dict["accuracy"],
            "correct": int(acc_dict["correct"]),
            "n": int(acc_dict["n"]),
            "report": acc_dict["report"],
        }

    def _write_results(self, llmind_report=None):
        result = {
            "questions": self.output_questions,
            "gt_answers": self.output_answers,
            "llmind_answers": [] if llmind_report is None else llmind_report["pred_answers"],
            "llmind_accuracy": None if llmind_report is None else llmind_report["accuracy"],
            "uniform_answers": self.vanilla_acc["pred_answers"],
            "uniform_accuracy": self.vanilla_acc["accuracy"],
        }
        with open(self._results_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    def _sample_questions_bandit(self, m: int) -> List[int]:
        w = self._q_weights
        probs = (w / w.sum()).numpy()
        idxs = list(range(len(self.questions)))
        chosen = []
        for _ in range(m):
            pick = random.choices(idxs, weights=[probs[i] for i in idxs], k=1)[0]
            chosen.append(pick)
            idxs.remove(pick)
        return chosen

    def _select_q_subset(self):
        if self._hard_q:
            return list(self._hard_q)
        m = max(1, int(len(self.questions) * float(self.spsa_q_fraction)))
        return self._sample_questions_bandit(m) if m < len(self.questions) else list(range(len(self.questions)))

    def _reconstruct_from_params(self, params):
        a, b, c, d = params[0], params[1], params[2], params[3]
        warped = apply_mobius_transform_torch(self.IMAGE, a, b, c, d)
        for i in range(4, len(params), 4):
            a, b, c, d = params[i], params[i+1], params[i+2], params[i+3]
            warped = apply_mobius_transform_torch(warped, a, b, c, d)

        samp = uniform_sample(warped, p=self.args.percentage)

        a, b, c, d = params[-4], params[-3], params[-2], params[-1]
        recon = apply_inverse_mobius_transform_torch(samp, a, b, c, d)
        for i in range(len(params)-8, -1, -4):
            a, b, c, d = params[i], params[i+1], params[i+2], params[i+3]
            recon = apply_inverse_mobius_transform_torch(recon, a, b, c, d)

        if self.clip_recon:
            recon = recon + (recon.clamp(0.0, 1.0) - recon).detach()
        return recon

    @torch.no_grad()
    def _reconstruct_no_grad(self):
        params = self.model()
        return self._reconstruct_from_params(params)

    @torch.no_grad()
    def _spsa_text_grad(self, sigma: float, m_dirs: int, q_idx_subset=None,
                        K: int = 1, temperature: float = 0.0, top_p: float = 1.0, do_sample: bool = False):
        params = list(self.model.parameters())
        originals = [p.data.clone() for p in params]

        dirs = []
        for _ in range(m_dirs):
            one = []
            for p in params:
                d = torch.empty_like(p).bernoulli_(0.5).mul_(2).sub_(1)
                n = d.norm()
                if n > 0:
                    d = d / n
                scale = p.data.norm()
                if scale > 0:
                    d = d * scale
                one.append(d)
            dirs.append(one)

        def _apply(alpha, dset):
            for p, o, d in zip(params, originals, dset):
                p.data.copy_(o + alpha * d)

        f_plus, f_minus = [], []
        for i in range(m_dirs):
            dset = dirs[i]

            _apply(+sigma, dset)
            recon_p = self._reconstruct_no_grad()
            lp = self._text_loss(recon_p, q_idx_subset, K=K, temperature=temperature, top_p=top_p, do_sample=do_sample)
            f_plus.append(lp)

            _apply(-sigma, dset)
            recon_m = self._reconstruct_no_grad()
            lm = self._text_loss(recon_m, q_idx_subset, K=K, temperature=temperature, top_p=top_p, do_sample=do_sample)
            f_minus.append(lm)

            for p, o in zip(params, originals):
                p.data.copy_(o)

        g_list = [torch.zeros_like(p) for p in params]
        inv = 1.0 / (2.0 * sigma * max(1, m_dirs))
        for i in range(m_dirs):
            coef = (f_plus[i] - f_minus[i]) * inv
            for gi, di in zip(g_list, dirs[i]):
                gi.add_(coef * di)

        f_mean = float((sum(f_plus) + sum(f_minus)) / max(1, 2 * m_dirs))
        return g_list, f_mean

    def run(self):
        opt = optim.Adam(self.model.parameters(), lr=self.args.lr)
        scheduler = lr_scheduler.MultiStepLR(opt, milestones=[self.args.epochs // 2], gamma=0.1)

        best_loss = float("inf")
        best_accuracy = self.vanilla_acc["accuracy"]
        best_report = self.vanilla_acc

        total_steps = self.args.epochs
        if self.progress_callback is not None:
            self.progress_callback(0, total_steps)

        progress = tqdm(range(1, total_steps + 1), desc="Training", unit="step", dynamic_ncols=True)
        for step in progress:
            opt.zero_grad()

            do_text = (step % max(1, self.spsa_every) == 0)
            spsa_grad_list = None

            if do_text:
                q_idx_subset = self._select_q_subset()
                spsa_grad_list, _ = self._spsa_text_grad(
                    sigma=self.spsa_sigma,
                    m_dirs=self.k_dirs,
                    q_idx_subset=q_idx_subset,
                    K=self.spsa_samples,
                    temperature=self.spsa_temp,
                    top_p=self.spsa_top_p,
                    do_sample=self.spsa_do_sample
                )

            params = self.model()
            recon = self._reconstruct_from_params(params)

            alpha = 1.0
            if self.target_img_loss > 0:
                t = step / max(1, self.args.epochs)
                alpha = 0.5 + 0.25 * (1 + torch.cos(torch.tensor(t * math.pi)))
            loss_img = alpha * self.scorer(self.IMAGE, recon)
            loss_img.backward()

            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.parameters()), \
                "No image grad! Check transforms/sampler differentiability."

            if do_text and self.beta_spsa > 0 and spsa_grad_list is not None:
                with torch.no_grad():
                    img_g2 = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            img_g2 += float((p.grad * p.grad).sum().item())
                    img_norm = (img_g2 ** 0.5) if img_g2 > 0 else 0.0

                    spsa_g2 = 0.0
                    for g in spsa_grad_list:
                        spsa_g2 += float((g * g).sum().item())
                    spsa_norm = (spsa_g2 ** 0.5) if spsa_g2 > 0 else 0.0

                    target = max(self.spsa_min_norm, img_norm * self.beta_spsa)
                    scale = (target / (spsa_norm + 1e-12)) if spsa_norm > 0 else 0.0

                for p, g in zip(self.model.parameters(), spsa_grad_list):
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    p.grad.add_(scale * g)

            opt.step()
            scheduler.step()

            cur_loss = float(loss_img.item())
            should_eval = (step % self.eval_every == 0) or (step == total_steps)
            if not should_eval:
                self._write_results(best_report)
                if self.progress_callback is not None:
                    self.progress_callback(step, total_steps)
                continue

            llmind_pil = tensor_to_inpainted_pil(recon)
            pred_report = self._get_pred_report_from_pil(llmind_pil)

            if self._hard_q is None and "report" in pred_report:
                self._hard_q = sorted(pred_report["report"]["wrong_index"]) or list(range(len(self.questions)))
            wrong = set(pred_report["report"]["wrong_index"])
            for qi in range(len(self.questions)):
                gain = 1.0 if qi in wrong else 0.0
                self._q_weights[qi] *= math.exp(self.bandit_eta * gain / max(1, len(self.questions)))

            if pred_report["accuracy"] > best_accuracy or (
                pred_report["accuracy"] == best_accuracy and cur_loss < best_loss
            ):
                best_accuracy = pred_report["accuracy"]
                best_loss = cur_loss
                best_report = pred_report
                save_interpolated_pil(llmind_pil, self._llmind_path)

            self._write_results(best_report)
            if self.progress_callback is not None:
                self.progress_callback(step, total_steps)

        progress.close()
        if self.progress_callback is not None:
            self.progress_callback(total_steps, total_steps)
        self._write_results(best_report)
        return {
            "run_dir": self._run_dir,
            "results_path": self._results_path,
            "llmind_path": self._llmind_path,
            "uniform_path": self._uniform_path,
        }
