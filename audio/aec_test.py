import wave
import queue
import threading
import time
import sys
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
from aec_audio_processing import AudioProcessor


# ====== Audio format ======
SR = 16000
CH = 1
SAMPLE_WIDTH_BYTES = 2  # int16

# WebRTC AEC likes 10 ms frames
AEC_FRAME_MS = 10
AEC_SAMPLES_PER_FRAME = SR * AEC_FRAME_MS // 1000  # 160
AEC_FRAME_BYTES = AEC_SAMPLES_PER_FRAME * CH * SAMPLE_WIDTH_BYTES  # 320 bytes

# Use a larger device blocksize for stability (reduce xruns/crackling)
DEVICE_BLOCK_MS = 20
DEVICE_SAMPLES_PER_BLOCK = SR * DEVICE_BLOCK_MS // 1000  # 640
DEVICE_BLOCK_BYTES = DEVICE_SAMPLES_PER_BLOCK * CH * SAMPLE_WIDTH_BYTES  # 1280 bytes
FRAMES_PER_DEVICE_BLOCK = DEVICE_SAMPLES_PER_BLOCK // AEC_SAMPLES_PER_FRAME  # 4


@dataclass
class SharedState:
    stop: threading.Event
    playback_done: threading.Event


def assert_wav_format(path: str):
    with wave.open(path, "rb") as wf:
        if wf.getframerate() != SR:
            raise ValueError(f"{path}: expected {SR} Hz, got {wf.getframerate()} Hz")
        if wf.getnchannels() != CH:
            raise ValueError(f"{path}: expected mono ({CH} ch), got {wf.getnchannels()} ch")
        if wf.getsampwidth() != SAMPLE_WIDTH_BYTES:
            raise ValueError(f"{path}: expected 16-bit PCM, got {wf.getsampwidth()*8}-bit")


def wav_playback_feeder(
    wav_path: str,
    playback_q: "queue.Queue[bytes]",
    state: SharedState,
    prefill_frames: int = 50,
):
    """
    Reads 10 ms frames from WAV and pushes them into playback_q.
    When EOF, marks playback_done and then keeps pushing silence until stopped
    (so speakers stay clocked and AEC keeps stable timing).
    """
    try:
        with wave.open(wav_path, "rb") as wf:
            # Prefill
            for _ in range(prefill_frames):
                if state.stop.is_set():
                    return
                data = wf.readframes(AEC_SAMPLES_PER_FRAME)
                if not data or len(data) < AEC_FRAME_BYTES:
                    break
                playback_q.put(data)

            # Stream remainder
            while not state.stop.is_set():
                data = wf.readframes(AEC_SAMPLES_PER_FRAME)
                if not data:
                    state.playback_done.set()
                    break
                if len(data) < AEC_FRAME_BYTES:
                    data = data + b"\x00" * (AEC_FRAME_BYTES - len(data))
                    state.playback_done.set()

                playback_q.put(data)

    finally:
        # After EOF: keep providing silence so output callback doesn't underrun.
        silence = b"\x00" * AEC_FRAME_BYTES
        while not state.stop.is_set():
            try:
                playback_q.put(silence, timeout=0.1)
            except queue.Full:
                pass


def cleaned_wav_writer(
    out_path: str,
    clean_q: "queue.Queue[bytes]",
    state: SharedState,
):
    """
    Consumes cleaned audio frames (10 ms int16 mono) and writes to a WAV file.
    Stops when state.stop is set AND the queue has been drained.
    """
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(CH)
        wf.setsampwidth(SAMPLE_WIDTH_BYTES)
        wf.setframerate(SR)

        while True:
            if state.stop.is_set():
                # drain remaining frames quickly
                drained_any = False
                while True:
                    try:
                        data = clean_q.get_nowait()
                    except queue.Empty:
                        break
                    wf.writeframes(data)
                    drained_any = True
                if not drained_any:
                    break

            try:
                data = clean_q.get(timeout=0.1)
            except queue.Empty:
                continue
            wf.writeframes(data)


class FullDuplexAEC:
    def __init__(self, playback_q: "queue.Queue[bytes]", clean_q: "queue.Queue[bytes]"):
        self.playback_q = playback_q
        self.clean_q = clean_q

        self.ap = AudioProcessor(enable_aec=True, enable_ns=True, enable_agc=True)
        self.ap.set_stream_format(SR, CH)
        self.ap.set_reverse_stream_format(SR, CH)

        self.stream = None
        self._delay_set = False

        # Reused buffers to reduce allocations
        self._silence_frame = b"\x00" * AEC_FRAME_BYTES

    def _maybe_set_delay_from_stream(self):
        # sounddevice gives (input_latency, output_latency) in seconds
        in_lat_s, out_lat_s = self.stream.latency
        delay_ms = int((in_lat_s + out_lat_s) * 1000)

        # Add one AEC frame to account for callback scheduling
        delay_ms += AEC_FRAME_MS

        # Clamp to a sane range
        delay_ms = max(10, min(delay_ms, 200))

        self.ap.set_stream_delay(delay_ms)
        self._delay_set = True
        print(f"[AEC] Using stream_delay_ms={delay_ms} (in_lat={in_lat_s:.3f}s out_lat={out_lat_s:.3f}s)")

    def _post_gain_with_limiter(
        self,
        frame_i16: np.ndarray,
        target_peak: float = 1.0,  # 0..1 (0.5 ~= -6 dBFS)
        max_gain: float = 8.0,
    ) -> np.ndarray:
        """
        Post-process frame to limit peak to target_peak and apply gain.
        """
        x = frame_i16.astype(np.float32) / 32768.0
        peak = np.max(np.abs(x)) + 1e-9
        gain = min(max_gain, target_peak / peak)
        y = np.clip(x * gain, -1.0, 1.0)
        return (y * 32767.0).astype(np.int16)

    def _get_playback_block_i16(self) -> np.ndarray:
        """
        Pull a 40 ms (DEVICE_SAMPLES_PER_BLOCK) playback block from the queue,
        which stores 10 ms frames. Never blocks the audio callback.
        """
        needed_frames = FRAMES_PER_DEVICE_BLOCK  # 4 frames of 10ms
        chunks = []

        for _ in range(needed_frames):
            try:
                b = self.playback_q.get_nowait()
            except queue.Empty:
                b = self._silence_frame

            # normalize chunk size to exactly one 10ms frame
            if len(b) != AEC_FRAME_BYTES:
                if len(b) < AEC_FRAME_BYTES:
                    b = b + b"\x00" * (AEC_FRAME_BYTES - len(b))
                else:
                    b = b[:AEC_FRAME_BYTES]
            chunks.append(b)

        block_bytes = b"".join(chunks)  # 40ms worth of bytes
        return np.frombuffer(block_bytes, dtype=np.int16)

    def callback(self, indata, outdata, frames, time_info, status):
        if status:
            # underruns/overruns degrade AEC a lot
            print(status)

        if not self._delay_set and self.stream is not None:
            self._maybe_set_delay_from_stream()

        # Expect 40 ms blocks from the device
        # (sounddevice should honor this with blocksize=DEVICE_SAMPLES_PER_BLOCK)
        if frames != DEVICE_SAMPLES_PER_BLOCK:
            # If host API gives a different size, we can still handle it by
            # processing in 10ms chunks where possible, but the example assumes fixed 40ms.
            # Pad/trim to keep behavior deterministic.
            mic_block = indata[:, 0].copy()
            if mic_block.shape[0] < DEVICE_SAMPLES_PER_BLOCK:
                mic_block = np.pad(mic_block, (0, DEVICE_SAMPLES_PER_BLOCK - mic_block.shape[0]))
            else:
                mic_block = mic_block[:DEVICE_SAMPLES_PER_BLOCK]
        else:
            mic_block = indata[:, 0].copy()

        # Pull matching 40ms playback block
        play_block = self._get_playback_block_i16()

        # Output playback block to speakers
        outdata[:, 0] = play_block

        # Process in 10ms AEC frames
        for i in range(0, DEVICE_SAMPLES_PER_BLOCK, AEC_SAMPLES_PER_FRAME):
            play_frame = play_block[i : i + AEC_SAMPLES_PER_FRAME]
            mic_frame = mic_block[i : i + AEC_SAMPLES_PER_FRAME]

            play_bytes = play_frame.tobytes()
            mic_bytes = mic_frame.tobytes()

            # Feed reverse stream (exact samples played)
            self.ap.process_reverse_stream(play_bytes)

            # AEC process
            clean_bytes = self.ap.process_stream(mic_bytes)

            # Keep your gain stage (frame-based)
            clean_i16 = np.frombuffer(clean_bytes, dtype=np.int16)
            clean_i16 = self._post_gain_with_limiter(clean_i16)
            clean_bytes = clean_i16.tobytes()

            # Push cleaned frame to writer queue WITHOUT blocking callback
            try:
                self.clean_q.put_nowait(clean_bytes)
            except queue.Full:
                pass


def main(in_wav: str, out_wav: str, record_seconds: float = 10.0):
    assert_wav_format(in_wav)

    # Queues: tune sizes if needed
    playback_q: "queue.Queue[bytes]" = queue.Queue(maxsize=400)  # ~4 seconds at 10ms frames
    clean_q: "queue.Queue[bytes]" = queue.Queue(maxsize=800)     # ~8 seconds buffering

    state = SharedState(stop=threading.Event(), playback_done=threading.Event())

    feeder = threading.Thread(
        target=wav_playback_feeder,
        args=(in_wav, playback_q, state),
        daemon=True,
    )
    writer = threading.Thread(
        target=cleaned_wav_writer,
        args=(out_wav, clean_q, state),
        daemon=True,
    )

    aec = FullDuplexAEC(playback_q, clean_q)

    feeder.start()
    writer.start()

    # Full-duplex stream: int16 in/out, configurable device blocks, 10ms AEC internal frames
    with sd.Stream(
        samplerate=SR,
        channels=CH,
        dtype="int16",
        blocksize=DEVICE_SAMPLES_PER_BLOCK,
        latency="high",                      # improves stability / avoids xruns
        callback=aec.callback,
    ) as stream:
        aec.stream = stream
        print("Streaming... (Ctrl+C to stop)")
        t0 = time.time()
        try:
            while True:
                if time.time() - t0 >= record_seconds:
                    break
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass

    # Stop threads and flush writer
    state.stop.set()
    feeder.join(timeout=1.0)
    writer.join(timeout=2.0)
    print(f"Done. Wrote cleaned mic to: {out_wav}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python aec_example.py <playback.wav> <cleaned_out.wav> [record_seconds]")
        sys.exit(1)

    in_wav = sys.argv[1]
    out_wav = sys.argv[2]
    secs = float(sys.argv[3]) if len(sys.argv) >= 4 else 10.0
    main(in_wav, out_wav, secs)
