#!/usr/bin/env python3
"""
Real-time-ish (sliding-window) 2-speaker separation + speaker ID POC.

Pipeline:
  mic -> ring buffer -> SepFormer separation (2 sources) -> choose "user" source
      -> ECAPA embeddings -> cosine similarity -> smooth -> print decisions

Notes:
- SepFormer here is run in sliding windows (NOT truly causal streaming).
- Expect ~window_size latency (e.g. 1.5s) in this POC.
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torchaudio
import sounddevice as sd
import soundfile as sf

from speechbrain.inference.separation import SepformerSeparation
from speechbrain.inference.speaker import SpeakerRecognition


SR = 16000


def l2_normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: [D], b: [N, D] => [N]
    a = l2_normalize(a)
    b = l2_normalize(b)
    return torch.matmul(b, a)


def rms_energy(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)) + 1e-12)


@dataclass
class EnrolledSpeaker:
    name: str
    emb: torch.Tensor  # [D]


def load_audio_mono_16k(path: str) -> torch.Tensor:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)  # [T, C]
    audio = audio.mean(axis=1)  # mono [T]
    wav = torch.from_numpy(audio)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav

def save_db(db_path: Path, speakers: Dict[str, torch.Tensor]) -> None:
    """
    speakers[name] = Tensor [K, D] (preferred) or [D] (we'll store as [1, D])
    """
    out = {}
    for name, emb in speakers.items():
        emb = emb.detach().cpu()
        out[name] = emb.numpy()
    np.savez(db_path, **out)


def load_db(db_path: Path) -> Dict[str, torch.Tensor]:
    """
    Returns speakers[name] = Tensor [K, D]
    Accepts legacy saved embeddings [D] too.
    """
    z = np.load(db_path, allow_pickle=False)
    speakers = {}
    for name in z.files:
        arr = z[name]
        t = torch.tensor(arr).float()
        speakers[name] = t  # [K, D]
    return speakers

class RingBuffer:
    def __init__(self, capacity_samples: int):
        self.capacity = capacity_samples
        self.buf = np.zeros((capacity_samples,), dtype=np.float32)
        self.write_pos = 0
        self.filled = 0

    def push(self, x: np.ndarray) -> None:
        x = x.astype(np.float32, copy=False).reshape(-1)
        n = len(x)
        if n >= self.capacity:
            self.buf[:] = x[-self.capacity:]
            self.write_pos = 0
            self.filled = self.capacity
            return

        end = self.write_pos + n
        if end <= self.capacity:
            self.buf[self.write_pos:end] = x
        else:
            first = self.capacity - self.write_pos
            self.buf[self.write_pos:] = x[:first]
            self.buf[: end % self.capacity] = x[first:]
        self.write_pos = end % self.capacity
        self.filled = min(self.capacity, self.filled + n)

    def get_last(self, n: int) -> np.ndarray:
        n = min(n, self.filled)
        start = (self.write_pos - n) % self.capacity
        if start + n <= self.capacity:
            return self.buf[start:start + n].copy()
        else:
            first = self.capacity - start
            return np.concatenate([self.buf[start:], self.buf[: n - first]]).copy()


def separate_window(
    sep_model: SepformerSeparation,
    window: np.ndarray,
    device: str
) -> np.ndarray:
    """
    window: [T] float32 @ 16k
    returns: [2, T] float32
    """
    # SpeechBrain SepformerSeparation expects [B, T]
    x = torch.from_numpy(window).to(device).unsqueeze(0)  # [1, T]

    with torch.inference_mode():
        est = sep_model.separate_batch(x)

        # Common SpeechBrain output: [B, T, Nsrc]
        if est.dim() == 3:
            est = est[0].transpose(0, 1)  # [Nsrc, T]
        # Some variants: [B, Nsrc, T]
        elif est.dim() == 3 and est.shape[1] in (2, 3):
            est = est[0]  # [Nsrc, T]
        # Some variants: [B, T, Nsrc, C] etc. (rare)
        elif est.dim() == 4:
            # try to collapse channel dimension if present
            est = est[0]
            # If shape [T, Nsrc, C] -> take C=0 and transpose
            if est.shape[-1] == 1:
                est = est[..., 0]
            # Now likely [T, Nsrc]
            if est.dim() == 2:
                est = est.transpose(0, 1)  # [Nsrc, T]
            else:
                raise RuntimeError(f"Unexpected 4D separation output shape after squeeze: {tuple(est.shape)}")
        else:
            raise RuntimeError(f"Unexpected separation output shape: {tuple(est.shape)}")

    return est.detach().cpu().float().numpy()


def embed_audio(
    spk_model: SpeakerRecognition,
    audio_16k: np.ndarray,
    device: str
) -> torch.Tensor:
    """
    audio_16k: [T] float32
    returns: [D] embedding
    """
    x = torch.from_numpy(audio_16k).to(device).unsqueeze(0)  # [B, T]
    with torch.inference_mode():
        emb = spk_model.encode_batch(x)  # [B, 1, D] or [B, D]
        if emb.dim() == 3:
            emb = emb[0, 0]
        else:
            emb = emb[0]
    return emb.float().detach().cpu()

def best_similarity_to_prototypes(live_emb: torch.Tensor, protos: torch.Tensor) -> float:
    """
    live_emb: [D]
    protos:   [K, D]
    returns: best cosine similarity (float)
    """
    live_emb = l2_normalize(live_emb)
    protos = l2_normalize(protos)
    sims = torch.matmul(protos, live_emb)  # [K]
    return float(torch.max(sims))

def topk_mean_similarity(live_emb: torch.Tensor, protos: torch.Tensor, k: int = 5) -> float:
    """
    live_emb: [D]
    protos:   [K, D]
    returns: mean of top k cosine similarities (float)
    """
    live_emb = l2_normalize(live_emb)
    protos = l2_normalize(protos)
    sims = torch.matmul(protos, live_emb)  # [K]
    k = min(k, sims.numel())
    topk = torch.topk(sims, k=k).values
    return float(torch.mean(topk))


def choose_user_stream_by_similarity(
    spk_model: SpeakerRecognition,
    streams: np.ndarray,               # [2, T]
    enrolled_user_protos: torch.Tensor,# [K, D]
    device: str
) -> Tuple[int, float]:
    scores = []
    for i in range(streams.shape[0]):
        emb_i = embed_audio(spk_model, streams[i], device=device)
        s = topk_mean_similarity(emb_i, enrolled_user_protos)
        scores.append(s)
    best_i = int(np.argmax(scores))
    return best_i, float(scores[best_i])

def peak_normalize_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    return x / (np.max(np.abs(x)) + 1e-9)



SR = 16000

def record_mic_chunk(seconds: float) -> np.ndarray:
    """Raw mic recording, returns float32 mono [T]."""
    print(f"Recording {seconds:.1f}s… speak now.")
    x = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    return x.reshape(-1)


def _rms_vad_mask(x: np.ndarray, sr: int = SR, frame_ms: int = 30, hop_ms: int = 10,
                  noise_percentile: int = 20, thr_mult: float = 2.5, thr_min: float = 0.003,
                  hang_ms: int = 120) -> tuple[np.ndarray, float]:
    """
    Returns (sample_mask, threshold_used).
    """
    x = x.astype(np.float32, copy=False)
    n = len(x)
    frame = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)

    if n < frame:
        return np.ones(n, dtype=bool), 0.0

    # Compute frame RMS
    rms = []
    spans = []
    for start in range(0, n - frame + 1, hop):
        seg = x[start:start + frame]
        r = float(np.sqrt(np.mean(seg * seg, dtype=np.float64) + 1e-12))
        rms.append(r)
        spans.append((start, start + frame))
    rms = np.asarray(rms, dtype=np.float32)

    # Adaptive threshold from noise floor estimate
    noise_floor = float(np.percentile(rms, noise_percentile))
    thr = max(noise_floor * thr_mult, thr_min)

    mask = np.zeros(n, dtype=bool)
    for (start, end), r in zip(spans, rms):
        if r >= thr:
            mask[start:end] = True

    # Hangover (dilate) to avoid chopping phonemes
    hang = int(sr * hang_ms / 1000)
    if mask.any() and hang > 0:
        idx = np.where(mask)[0]
        # Dilate by expanding each True sample (fast enough for small chunks)
        for i in idx[::max(1, len(idx)//2000)]:  # subsample for speed on long chunks
            lo = max(0, i - hang)
            hi = min(n, i + hang)
            mask[lo:hi] = True

    return mask, thr


def record_mic_chunk_voiced(
    record_s: float,
    target_voiced_s: float = 2.5,
    pad_if_short: bool = True,
    # VAD params (tunable)
    frame_ms: int = 30,
    hop_ms: int = 10,
    thr_mult: float = 2.5,
    thr_min: float = 0.003,
) -> np.ndarray:
    """
    Records from mic, then keeps only voiced samples using RMS-VAD.
    Returns float32 mono [T_voiced] roughly target_voiced_s long (if enough speech).
    """
    raw = record_mic_chunk(record_s)

    # Peak normalize for more stable RMS behavior (optional but helps)
    raw = raw.astype(np.float32, copy=False)
    raw = raw / (np.max(np.abs(raw)) + 1e-9)

    mask, thr = _rms_vad_mask(
        raw, sr=SR, frame_ms=frame_ms, hop_ms=hop_ms,
        thr_mult=thr_mult, thr_min=thr_min
    )
    voiced = raw[mask]

    target_n = int(target_voiced_s * SR)

    # If too little voiced audio, fall back to raw (better than returning nothing)
    if len(voiced) < int(0.6 * target_n):
        print(f"Enrollment VAD: insufficient voiced audio (thr={thr:.4f}, voiced={len(voiced)/SR:.2f}s). Skipping.")
        out = None
    else:
        out = voiced[:target_n]
        print(f"Enrollment VAD: thr={thr:.4f}, kept={len(voiced)/SR:.2f}s, using={len(out)/SR:.2f}s")

    # Optional: pad with zeros to a fixed length (some models like consistent length)
    if out is not None and pad_if_short and len(out) < target_n:
        out = np.pad(out, (0, target_n - len(out)), mode="constant")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", help="cpu or cuda")
    ap.add_argument("--sep_model", default="speechbrain/sepformer-whamr16k",
                    help="SpeechBrain SepFormer model id")
    ap.add_argument("--spk_model", default="speechbrain/spkrec-ecapa-voxceleb",
                    help="SpeechBrain speaker embedding model id")

    sub = ap.add_subparsers(dest="cmd", required=True)

    enroll = sub.add_parser("enroll", help="Enroll a speaker from an audio file into an .npz db")
    enroll.add_argument("--name", required=True)
    enroll.add_argument("--n_protos", type=int, default=5,
                        help="If recording from mic, record multiple chunks and store each as a prototype")
    enroll.add_argument("--chunk_s", type=float, default=2.0,
                        help="Chunk length (seconds) per prototype when recording")


    enroll.add_argument("--db", required=True)

    runmic = sub.add_parser("run-mic", help="Run mic separation + speaker ID")
    runmic.add_argument("--db", required=True, help="npz with enrolled speakers")
    runmic.add_argument("--window_s", type=float, default=1.0)
    runmic.add_argument("--hop_s", type=float, default=0.4)
    runmic.add_argument("--min_rms", type=float, default=0.003, help="Skip if window RMS below this")
    runmic.add_argument("--print_every", type=float, default=0.5)
    runmic.add_argument("--tail_s", type=float, default=0.8,
                        help="How much separated audio (seconds) to embed each step (1.0–1.0 is typical).")
    runmic.add_argument("--user_thr", type=float, default=0.30,
                        help="Speaker evidence threshold (tune ~0.28–0.40).")
    runmic.add_argument("--evidence_len", type=int, default=2,
                        help="How many recent hops to consider for evidence.")
    runmic.add_argument("--evidence_need", type=int, default=1,
                        help="How many positives in that window to identify a speaker.")


    args = ap.parse_args()
    device = args.device

    if args.cmd == "enroll":
        db_path = Path(args.db)
        speakers = load_db(db_path) if db_path.exists() else {}

        spk = SpeakerRecognition.from_hparams(
            source=args.spk_model,
            run_opts={"device": device},
            savedir="pretrained_models/spkrec"
        )

        all_embs = []
        for i in range(args.n_protos):
            print(f"[{i+1}/{args.n_protos}]")
            while True:
                wav = record_mic_chunk_voiced(
                    record_s=max(2.0, args.chunk_s * 2.0),   # record longer than target
                    target_voiced_s=args.chunk_s,            # extract this much voiced speech
                    thr_mult=2.5,                            # <-- keep sensible
                    thr_min=0.003,
                )
                if wav is not None:
                    break
            wav = peak_normalize_np(wav)
            emb = embed_audio(spk, wav, device=device)   # [D]
            all_embs.append(emb.unsqueeze(0))            # [1, D]

        embs = torch.cat(all_embs, dim=0)                # [K, D]

        ## Append if name exists (keeps older prototypes too)
        #if args.name in speakers:
        #    speakers[args.name] = torch.cat([speakers[args.name], embs], dim=0)
        #else:
        speakers[args.name] = embs

        save_db(db_path, speakers)
        print(f"Enrolled '{args.name}' into {db_path} (protos={speakers[args.name].shape[0]}, dim={emb.numel()}).")
        return

    if args.cmd == "run-mic":
        speakers = load_db(Path(args.db))
        if not speakers:
            raise SystemExit(f"DB is empty. Enroll speakers first with: enroll --name <name> ...")
        
        print(f"Loaded {len(speakers)} speakers: {list(speakers.keys())}")

        sep = SepformerSeparation.from_hparams(
            source=args.sep_model,
            run_opts={"device": device},
            savedir="pretrained_models/sepformer"
        )
        spk = SpeakerRecognition.from_hparams(
            source=args.spk_model,
            run_opts={"device": device},
            savedir="pretrained_models/spkrec"
        )

        window_n = int(args.window_s * SR)
        hop_n = int(args.hop_s * SR)
        tail_n = int(args.tail_s * SR)

        ring = RingBuffer(capacity_samples=max(window_n * 2, window_n + tail_n + hop_n))

        print("Listening… (Ctrl+C to stop)")
        last_print = 0.0

        from collections import deque, defaultdict
        
        # Track evidence for each speaker separately
        speaker_evidence = {name: deque(maxlen=args.evidence_len) for name in speakers.keys()}
        identified_speaker: Optional[str] = None

        with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=hop_n) as stream:
            while True:
                block, overflowed = stream.read(hop_n)  # block: [T, C]
                if overflowed:
                    # You can print or ignore
                    pass
                block = block.reshape(-1).astype(np.float32, copy=False)
                ring.push(block)

                window = ring.get_last(window_n)
                if len(window) < window_n:
                    continue

                if rms_energy(window) < args.min_rms:
                    continue

                est = separate_window(sep, window, device=device)  # [2, T]

                # Take a tail for embedding (more stable than only hop)
                t_n = min(tail_n, est.shape[1])
                streams = est[:, -t_n:]  # [2, t_n]

                # Compare with all speakers on both streams; take the best stream
                best_stream_scores = []  # Best score per stream across all speakers
                best_stream_speakers = []  # Best speaker per stream
                
                for i in range(2):
                    x = peak_normalize_np(streams[i])

                    # Skip if separated stream is basically silent
                    if rms_energy(x) < 0.002:
                        best_stream_scores.append(-1.0)
                        best_stream_speakers.append(None)
                        continue

                    emb = embed_audio(spk, x, device=device)
                    
                    # Compare with all speakers
                    best_score = -1.0
                    best_speaker = None
                    speaker_scores = {}
                    
                    for speaker_name, speaker_protos in speakers.items():
                        s = topk_mean_similarity(emb, speaker_protos, k=5)
                        speaker_scores[speaker_name] = s
                        if s > best_score:
                            best_score = s
                            best_speaker = speaker_name
                    
                    best_stream_scores.append(best_score)
                    best_stream_speakers.append(best_speaker)

                # Use the best stream
                best_stream_idx = int(np.argmax(best_stream_scores))
                best_speaker_name = best_stream_speakers[best_stream_idx]
                best_score = best_stream_scores[best_stream_idx]

                # Update evidence for all speakers
                for speaker_name in speakers.keys():
                    if speaker_name == best_speaker_name and best_score > args.user_thr:
                        speaker_evidence[speaker_name].append(1)
                    else:
                        speaker_evidence[speaker_name].append(0)
                
                # Check if any speaker has enough evidence
                for speaker_name in speakers.keys():
                    evidence_count = sum(speaker_evidence[speaker_name])
                    if evidence_count >= args.evidence_need:
                        if identified_speaker != speaker_name:
                            identified_speaker = speaker_name
                            print(f"\n>>> IDENTIFIED SPEAKER: {speaker_name} <<<")
                        break
                else:
                    # No speaker reached threshold - reset if we had one identified
                    if identified_speaker is not None:
                        identified_speaker = None

                now = time.time()
                if now - last_print >= args.print_every:
                    # Show scores for all speakers
                    score_strs = []
                    for speaker_name in speakers.keys():
                        evidence_count = sum(speaker_evidence[speaker_name])
                        score_strs.append(f"{speaker_name}={evidence_count}/{len(speaker_evidence[speaker_name])}")
                    
                    print(
                        f"stream{best_stream_idx} best={best_speaker_name or 'none'}({best_score:.3f}) "
                        f"[{', '.join(score_strs)}]"
                    )
                    last_print = now


if __name__ == "__main__":
    main()
