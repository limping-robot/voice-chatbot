# plot_similarity_matrix.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_embeddings(db_path: Path):
    z = np.load(db_path, allow_pickle=False)
    names = []
    X = []
    speaker_ids = []

    for name in z.files:
        arr = np.asarray(z[name], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        for k in range(arr.shape[0]):
            names.append(f"{name}#{k}")
            speaker_ids.append(name)
            X.append(arr[k])

    X = np.stack(X, axis=0)   # [N, D]
    return names, speaker_ids, X

def l2_normalize(X, eps=1e-9):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

def main():
    db_path = Path("speakers.npz")   # <-- change if needed

    names, speaker_ids, X = load_embeddings(db_path)
    X = l2_normalize(X)

    # Cosine similarity matrix
    S = X @ X.T   # since rows normalized

    plt.figure(figsize=(8, 7))
    plt.imshow(S, vmin=0.0, vmax=1.0, cmap="magma")
    plt.colorbar(label="Cosine similarity")
    plt.title("Speaker embedding similarity matrix")

    # draw separator lines between speakers
    last = None
    idx = 0
    for s in speaker_ids:
        if last is None:
            last = s
        elif s != last:
            plt.axhline(idx - 0.5, color="white", linewidth=1)
            plt.axvline(idx - 0.5, color="white", linewidth=1)
            last = s
        idx += 1

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
