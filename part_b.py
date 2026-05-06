# ============================================================
#  SML Project — Part B: Contextual Embeddings
# ============================================================

import os
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
# EMBEDDING EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_embeddings(reviews: list, model_name: str,
                       cache_cls: str, cache_mean: str,
                       batch_size: int = 32):

    import torch
    from transformers import AutoTokenizer, AutoModel

    # ── Load cache if exists ──────────────────────────────────
    if Path(cache_cls).exists() and Path(cache_mean).exists():
        print(f"Loading cached embeddings: {model_name}")
        cls_embs  = np.load(cache_cls)["arr_0"]
        mean_embs = np.load(cache_mean)["arr_0"]
        return cls_embs, mean_embs

    print(f"\nExtracting embeddings: {model_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name)

    model.eval()
    model.to(device)

    cls_list, mean_list = [], []

    with torch.no_grad():
        for i in range(0, len(reviews), batch_size):

            batch = reviews[i : i + batch_size]

            enc = tokenizer(
                batch,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )

            enc = {k: v.to(device) for k, v in enc.items()}

            out = model(**enc)
            hidden = out.last_hidden_state   # (B, T, H)

            # CLS
            cls = hidden[:, 0, :].cpu().numpy()

            # Mean pooling
            mask = enc["attention_mask"].unsqueeze(-1).float()
            mean = (hidden * mask).sum(1) / mask.sum(1)
            mean = mean.cpu().numpy()

            cls_list.append(cls)
            mean_list.append(mean)

            if (i // batch_size) % 10 == 0:
                print(f"Batch {i // batch_size + 1}")

    cls_embs  = np.vstack(cls_list)
    mean_embs = np.vstack(mean_list)

    # ── Save cache ────────────────────────────────────────────
    np.savez_compressed(cache_cls,  cls_embs)
    np.savez_compressed(cache_mean, mean_embs)

    print(f"Saved: {cache_cls} | {cache_mean}")
    print(f"CLS shape : {cls_embs.shape}")
    print(f"Mean shape: {mean_embs.shape}")

    return cls_embs, mean_embs


# ═══════════════════════════════════════════════════════════════
# PART B DRIVER
# ═══════════════════════════════════════════════════════════════

def part_b(reviews: list):

    print("\n" + "═" * 60)
    print("PART B — Contextual Embeddings")
    print("═" * 60)

    embs = {}

    for model_tag, model_name in [
        ("distilbert", "distilbert-base-uncased"),
        ("bert",       "bert-base-uncased"),
    ]:
        cls_embs, mean_embs = extract_embeddings(
            reviews,
            model_name,
            cache_cls  = f"embeddings_{model_tag}_cls.npz",
            cache_mean = f"embeddings_{model_tag}_mean.npz",
        )

        embs[f"{model_tag}_cls"]  = cls_embs
        embs[f"{model_tag}_mean"] = mean_embs

    return embs



