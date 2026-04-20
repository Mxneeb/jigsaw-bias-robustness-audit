"""
pipeline.py — Production Moderation Pipeline
Assignment 2: Auditing Content Moderation AI for Bias, Adversarial Robustness & Safety
FAST-NUCES | Responsible & Explainable AI

Implements a three-layer ModerationPipeline:
  Layer 1 — Regex pre-filter (instant, rule-based)
  Layer 2 — Calibrated DistilBERT model
  Layer 3 — Human review queue (uncertain cases)
"""

import re
import os
import numpy as np
import torch
from typing import Optional, Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ===========================================================================
# LAYER 1 — Regex Blocklist
# 5 categories, 20+ compiled regex patterns
# ===========================================================================

BLOCKLIST: Dict[str, List[re.Pattern]] = {

    # ------------------------------------------------------------------
    # Category 1: Direct threats of violence (6 patterns)
    # ------------------------------------------------------------------
    "direct_threat": [
        # "I will / I'm going to kill|murder|shoot|stab|hurt you"
        re.compile(
            r"\b(?:i|we)\s+(?:will|shall|am\s+going\s+to|'m\s+going\s+to|gonna)\s+"
            r"(?:kill|murder|shoot|stab|hurt|harm|end)\s+(?:you|u|them|him|her)\b",
            re.IGNORECASE,
        ),
        # "I'll find where you live / hunt you down"
        re.compile(
            r"\b(?:i'll|i\s+will|we'll|we\s+will)\s+(?:find|hunt|track|locate)\s+"
            r"(?:you|u|where\s+you\s+live|you\s+down)\b",
            re.IGNORECASE,
        ),
        # "you're going to die / get killed / get hurt"
        re.compile(
            r"\byou(?:'re|\s+are)\s+(?:going\s+to|gonna)\s+"
            r"(?:die|get\s+(?:killed|hurt|shot|stabbed|murdered))\b",
            re.IGNORECASE,
        ),
        # "someone should (kill|murder|shoot|stab|hurt) you"
        re.compile(
            r"\bsomeone\s+should\s+(?:kill|murder|shoot|stab|hurt|harm)\s+(?:you|u|them)\b",
            re.IGNORECASE,
        ),
        # Variable-verb pattern with capturing group: kill|murder|shoot|stab|hurt
        re.compile(
            r"\b(?:i(?:'ll|\s+will|\s+am\s+going\s+to))\s+"
            r"(kill|murder|shoot|stab|hurt|assault)\s+(?:you|u|them|him|her)\b",
            re.IGNORECASE,
        ),
        # "I'll make you pay / suffer / regret this"
        re.compile(
            r"\b(?:i(?:'ll|\s+will|'m\s+gonna|\s+am\s+going\s+to))\s+make\s+you\s+"
            r"(?:suffer|pay|bleed|regret|die)\b",
            re.IGNORECASE,
        ),
    ],

    # ------------------------------------------------------------------
    # Category 2: Directed self-harm / suicide calls (5 patterns)
    # ------------------------------------------------------------------
    "self_harm_directed": [
        # "you should kill / hang / cut yourself"
        re.compile(
            r"\byou\s+should\s+(?:kill|hang|cut|hurt|end)\s+yourself\b",
            re.IGNORECASE,
        ),
        # "go kill / go hang yourself"
        re.compile(
            r"\bgo\s+(?:kill|hang|cut|hurt)\s+yourself\b",
            re.IGNORECASE,
        ),
        # "nobody would miss you if you died / disappeared"
        re.compile(
            r"\bnobody\s+(?:would|will|cares?|cared?)\s+(?:miss|mourn|care\s+about)\s+"
            r"(?:you|u)\s+(?:if\s+you\s+(?:died?|were\s+gone|disappeared?))?\b",
            re.IGNORECASE,
        ),
        # "do everyone a favor and disappear / die / kill yourself"
        re.compile(
            r"\bdo\s+(?:everyone|the\s+world|us\s+all)\s+a\s+fav(?:ou?r)\s+and\s+"
            r"(?:disappear|die|kill\s+yourself|end\s+it)\b",
            re.IGNORECASE,
        ),
        # "the world would be better without you"
        re.compile(
            r"\bworld\s+would\s+be\s+better\s+(?:off\s+)?(?:without\s+you|if\s+you(?:\s+were\s+dead|\s+died))\b",
            re.IGNORECASE,
        ),
    ],

    # ------------------------------------------------------------------
    # Category 3: Doxxing and stalking threats (5 patterns)
    # ------------------------------------------------------------------
    "doxxing_stalking": [
        # "I know where you live / your address / your real name"
        re.compile(
            r"\b(?:i|we)\s+know\s+(?:where\s+you\s+live|your\s+(?:address|location|real\s+name|home))\b",
            re.IGNORECASE,
        ),
        # "I'll post / share / leak / expose your address / info"
        re.compile(
            r"\b(?:i(?:'ll|\s+will|\s+am\s+going\s+to|\s+gonna))\s+"
            r"(?:post|share|leak|expose|publish|blast)\s+your\s+"
            r"(?:address|location|info(?:rmation)?|personal\s+info|real\s+name|identity|dox)\b",
            re.IGNORECASE,
        ),
        # "everyone will know who you really are / your real name"
        re.compile(
            r"\beveryone\s+(?:will|is\s+going\s+to|gonna)\s+know\s+"
            r"(?:who\s+you\s+really\s+are|your\s+real\s+name|where\s+you\s+live)\b",
            re.IGNORECASE,
        ),
        # "I found / tracked / located your real name / home"
        re.compile(
            r"\b(?:i|we)\s+(?:found|tracked|located|discovered)\s+your\s+"
            r"(?:real\s+name|address|home|school|workplace|ip\s*(?:address)?)\b",
            re.IGNORECASE,
        ),
        # "I'll come / show up / find you at your home / school"
        re.compile(
            r"\b(?:i(?:'ll|\s+will|\s+am\s+going\s+to|\s+gonna))\s+"
            r"(?:come|show\s+up|find\s+you|visit\s+you)\s+(?:to|at)\s+your\s+"
            r"(?:home|house|school|work(?:place)?|apartment)\b",
            re.IGNORECASE,
        ),
    ],

    # ------------------------------------------------------------------
    # Category 4: Severe dehumanization (5 patterns)
    # ------------------------------------------------------------------
    "dehumanization": [
        # "[group] are not human / people / persons"
        re.compile(
            r"\b\w+(?:\s+\w+)?\s+are\s+not\s+(?:human|people|persons?|real\s+humans?)\b",
            re.IGNORECASE,
        ),
        # "[group] are animals / subhuman / vermin / parasites"
        re.compile(
            r"\b\w+(?:\s+\w+)?\s+are\s+(?:animals?|subhuman|vermin|parasites?|cockroaches?|rats?|pests?|insects?)\b",
            re.IGNORECASE,
        ),
        # "[group] should be exterminated / eliminated / wiped out"
        re.compile(
            r"\b\w+(?:\s+\w+)?\s+should\s+be\s+(?:exterminated|eliminated|wiped\s+out|purged|eradicated|cleansed)\b",
            re.IGNORECASE,
        ),
        # "[group] are a disease / plague / cancer on society"
        re.compile(
            r"\b\w+(?:\s+\w+)?\s+are\s+(?:a\s+)?(?:disease|plague|cancer|infestation|blight|curse)\s+"
            r"(?:on|to|of|for)\s+(?:society|humanity|this\s+country|the\s+world|civilization)\b",
            re.IGNORECASE,
        ),
        # "those / these [group] aren't human"
        re.compile(
            r"\b(?:those|these)\s+\w+(?:\s+\w+)?\s+(?:aren't|are\s+not|aren't\s+even)\s+"
            r"(?:human|people|real\s+humans?|(?:true\s+)?(?:human\s+)?beings?)\b",
            re.IGNORECASE,
        ),
    ],

    # ------------------------------------------------------------------
    # Category 5: Coordinated harassment signals (4 patterns)
    # ------------------------------------------------------------------
    "coordinated_harassment": [
        # "everyone report / flag / attack this account"
        re.compile(
            r"\beveryone\s+(?:report|flag|block|attack|go\s+after)\s+"
            r"(?:this|that|their)?\s*(?:account|profile|user|post|channel)?\b",
            re.IGNORECASE,
        ),
        # "let's all go after / target / mass report"
        re.compile(
            r"\blet'?s\s+(?:all\s+)?(?:go\s+after|target|attack|report|mass\s+report|dogpile|pile\s+on)\b",
            re.IGNORECASE,
        ),
        # "raid / brigade their profile / channel"
        re.compile(
            r"\b(?:raid|brigade|mass\s+report|dogpile)\s+(?:their|this|that)?\s*"
            r"(?:profile|account|channel|stream|page)?\b",
            re.IGNORECASE,
        ),
        # lookahead: "raid" regardless of what follows
        re.compile(
            r"\braid(?=\s)",
            re.IGNORECASE,
        ),
    ],
}


def input_filter(text: str) -> Optional[Dict]:
    """
    Layer 1 pre-filter: scan text against all regex blocklist categories.

    Returns a block decision dict if a match is found, else None.
    The returned dict includes the matched category for auditability.
    """
    for category, patterns in BLOCKLIST.items():
        for pattern in patterns:
            if pattern.search(text):
                return {
                    "decision": "block",
                    "layer": "input_filter",
                    "category": category,
                    "confidence": 1.0,
                }
    return None


# ===========================================================================
# sklearn-compatible wrapper for DistilBERT (used for isotonic calibration)
# ===========================================================================

class _DistilBERTWrapper:
    """
    Thin sklearn-compatible wrapper around a HuggingFace SequenceClassification
    model so we can apply isotonic regression calibration via sklearn.
    """

    def __init__(self, model, tokenizer, device, batch_size: int = 32):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.classes_ = np.array([0, 1])

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        self.model.eval()
        all_probs = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            enc = self.tokenizer(
                batch,
                max_length=128,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
        return np.vstack(all_probs)

    def predict(self, texts: List[str]) -> np.ndarray:
        return (self.predict_proba(texts)[:, 1] >= 0.5).astype(int)


# ===========================================================================
# ModerationPipeline — public API
# ===========================================================================

class ModerationPipeline:
    """
    Three-layer production content moderation pipeline.

    Layer 1 — Regex pre-filter: instant, rule-based, zero model cost.
    Layer 2 — Calibrated DistilBERT: probabilistic scoring.
              confidence >= 0.6  → block
              confidence <= 0.4  → allow
              0.4 < confidence < 0.6 → escalate (Layer 3)
    Layer 3 — Human review queue: uncertain decisions.

    Usage
    -----
    pipeline = ModerationPipeline(model_path="./best_mitigated_model")
    pipeline.calibrate(calib_texts, calib_labels)
    result = pipeline.predict("Some comment text")
    # result = {"decision": "block"|"allow"|"review",
    #            "layer": "input_filter"|"model",
    #            "confidence": float,
    #            "category": str (only for input_filter blocks)}
    """

    # Thresholds for Layer 2 routing
    BLOCK_THRESHOLD = 0.6
    ALLOW_THRESHOLD = 0.4

    def __init__(self, model_path: str, batch_size: int = 32):
        """
        Parameters
        ----------
        model_path : str
            Path to a saved HuggingFace model directory (contains config.json,
            pytorch_model.bin, tokenizer files, etc.).
        batch_size : int
            Batch size for model inference.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ModerationPipeline] Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self._wrapper = _DistilBERTWrapper(
            self.model, self.tokenizer, self.device, batch_size
        )
        self._calibrator = None  # set after calling .calibrate()
        print(f"[ModerationPipeline] Model loaded on {self.device}. "
              f"Call .calibrate(texts, labels) before .predict().")

    # ------------------------------------------------------------------
    def calibrate(
        self,
        calib_texts: List[str],
        calib_labels: List[int],
    ) -> None:
        """
        Fit isotonic regression on top of raw model probabilities.

        Parameters
        ----------
        calib_texts  : list of str  — held-out calibration comments
        calib_labels : list of int  — ground-truth binary labels (0/1)
        """
        from sklearn.isotonic import IsotonicRegression

        print(f"[ModerationPipeline] Calibrating on {len(calib_texts)} samples...")
        raw_probs = self._wrapper.predict_proba(calib_texts)[:, 1]
        self._calibrator = IsotonicRegression(out_of_bounds="clip")
        self._calibrator.fit(raw_probs, calib_labels)
        print("[ModerationPipeline] Calibration complete.")

    # ------------------------------------------------------------------
    def _score(self, text: str) -> float:
        """Return calibrated (or raw) toxic probability for a single text."""
        raw = self._wrapper.predict_proba([text])[0, 1]
        if self._calibrator is not None:
            return float(self._calibrator.predict([raw])[0])
        return float(raw)

    # ------------------------------------------------------------------
    def predict(self, text: str) -> Dict:
        """
        Run the full three-layer pipeline on a single comment.

        Returns
        -------
        dict with keys:
            decision   : "block" | "allow" | "review"
            layer      : "input_filter" | "model"
            confidence : float (1.0 for Layer 1 blocks)
            category   : str (only present for input_filter decisions)
        """
        # ── Layer 1: Regex pre-filter ──────────────────────────────────
        filter_result = input_filter(text)
        if filter_result is not None:
            return filter_result

        # ── Layer 2: Calibrated model ──────────────────────────────────
        confidence = self._score(text)

        if confidence >= self.BLOCK_THRESHOLD:
            return {"decision": "block", "layer": "model", "confidence": confidence}
        elif confidence <= self.ALLOW_THRESHOLD:
            return {"decision": "allow", "layer": "model", "confidence": confidence}
        else:
            # ── Layer 3: Human review ──────────────────────────────────
            return {"decision": "review", "layer": "model", "confidence": confidence}

    # ------------------------------------------------------------------
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Convenience method to run .predict() on a list of texts."""
        return [self.predict(t) for t in texts]
