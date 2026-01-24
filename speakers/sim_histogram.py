# plot_similarity_histograms.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import itertools

def load_embeddings(db_path: Path):
    z = np.load(db_path, allow_pickle=False)
    speaker_to_embs = {}
    for name in z.files:
        arr = np.asarray(z[name], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        speaker_to_embs[name] = arr
    return speaker_to_embs

def l2_normalize(X, eps=1e-9):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

def cosine_matrix(A, B):
    return A @ B.T   # assumes L2 normalized

def main():
    db_path = Path("speakers.npz")   # <-- change if needed
    speakers = load_embeddings(db_path)

    # Normalize all embeddings
    for k in speakers:
        speakers[k] = l2_normalize(speakers[k])

    intra_sims = []
    inter_sims = []

    # Intra-speaker similarities
    for name, emb in speakers.items():
        if emb.shape[0] < 2:
            continue
        S = cosine_matrix(emb, emb)
        # take upper triangle without diagonal
        iu = np.triu_indices_from(S, k=1)
        intra_sims.extend(S[iu].tolist())

    # Inter-speaker similarities
    names = list(speakers.keys())
    for a, b in itertools.combinations(names, 2):
        A = speakers[a]
        B = speakers[b]
        S = cosine_matrix(A, B)
        inter_sims.extend(S.flatten().tolist())

    intra_sims = np.array(intra_sims)
    inter_sims = np.array(inter_sims)

    plt.figure(figsize=(8,5))
    plt.hist(intra_sims, bins=50, alpha=0.7, label="Intra-speaker")
    plt.hist(inter_sims, bins=50, alpha=0.7, label="Inter-speaker")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.title("Intra vs Inter speaker similarity distributions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"Intra-speaker mean: {intra_sims.mean():.3f}")
    print(f"Inter-speaker mean: {inter_sims.mean():.3f}")

if __name__ == "__main__":
    main()
