"""Live preview of a take as a short film, not one still.

ComfyUI's own previewer shows the first latent row of a video latent: a
still that sharpens step by step. A take is twenty seconds of motion, and
one still says nothing of where the people go. SwarmUI's sampler solved
this for its renders by decoding EVERY latent row through the cheap
latent-to-RGB matrix and pushing them as one animated WebP per step; the
studio shows that under "Create" and the person watches the take move
before it is decoded. This module does the same for the pack's samplers.

How: while a pack sampler runs, ComfyUI's ``latent_preview.prepare_callback``
is answered by ours (module attribute, looked up at call time by
SamplerCustomAdvanced). Our callback keeps the progress bar, decodes the
rows of the video stream (the audio stream of the nested latent is left
alone), encodes the WebP in a thread so sampling never waits, and sends
it as a standard PREVIEW_IMAGE binary event with image type 3 (1 = JPEG,
2 = PNG in ComfyUI; 3 = WebP is ours - a client that does not know it
reads the bytes as JPEG and shows nothing, exactly what it showed before).

Cost, measured 2026-09-23 on a PRO 4500 at 512x288 (31 rows): decoding
under 10 ms, WebP of 31 frames ~90 KB in a thread. Set H3_LIVE_PREVIEW=still
to go back to ComfyUI's still.
"""
import io
import os
import struct
import threading

import torch

try:
    import comfy.utils
    import latent_preview
    from server import PromptServer
    from protocol import BinaryEventTypes
except Exception:                                          # pragma: no cover
    latent_preview = None

# A 10 s take at 24 fps is 72 latent rows (measured 2026-09-23: 243 frames
# -> 72 rows), so a row stands for 140 ms: the film plays at the take's pace.
FRAME_MS = 140
WEBP_TYPE = 3

_depth = 0
_lock = threading.Lock()
_last_sent = [-1]


def enter():
    """A pack sampler starts: previews are films until leave()."""
    global _depth
    _depth += 1
    _last_sent[0] = -1


def leave():
    global _depth
    _depth = max(0, _depth - 1)


def _send(webp_bytes):
    server = PromptServer.instance
    data = bytearray(struct.pack(">I", WEBP_TYPE))
    data.extend(webp_bytes)
    server.send_sync(BinaryEventTypes.PREVIEW_IMAGE, data, sid=server.client_id)


def _film_callback(model, steps, x0_output_dict=None):
    previewer = latent_preview.get_previewer(model.load_device, model.model.latent_format)
    pbar = comfy.utils.ProgressBar(steps)

    def callback(step, x0, x, total_steps):
        if x0_output_dict is not None:
            x0_output_dict["x0"] = x0
        # The bar first: progress must never wait for a picture.
        pbar.update_absolute(step + 1, total_steps, None)
        if previewer is None:
            return
        v = x0.tensors[0] if getattr(x0, "is_nested", False) else x0
        if v.ndim != 5:
            return
        try:
            # (T, C, H, W). The latent's time axis runs backwards: the last row is the first frame
            # (seen 2026-09-24: the film played in reverse), so the rows are flipped before decoding.
            rows = v[0].permute(1, 0, 2, 3).flip(0)
            frames = [previewer.decode_latent_to_preview(rows[i:i + 1])
                      for i in range(rows.shape[0])]
        except Exception as e:
            print("[H3LivePreview] no film this step (%s)" % e, flush=True)
            return

        def encode_and_send():
            try:
                with _lock:
                    if step < _last_sent[0]:
                        return
                    buf = io.BytesIO()
                    frames[0].save(buf, format="WEBP", save_all=True, duration=FRAME_MS,
                                   append_images=frames[1:], lossless=False, quality=60, method=0)
                    _send(buf.getvalue())
                    _last_sent[0] = step
            except Exception as e:
                print("[H3LivePreview] not sent (%s)" % e, flush=True)
        threading.Thread(target=encode_and_send, daemon=True).start()
    return callback


if latent_preview is not None and not getattr(latent_preview.prepare_callback, "_h3_film", False):
    _original = latent_preview.prepare_callback

    def prepare_callback(model, steps, x0_output_dict=None):
        if _depth > 0 and os.environ.get("H3_LIVE_PREVIEW", "film") != "still":
            return _film_callback(model, steps, x0_output_dict)
        return _original(model, steps, x0_output_dict)
    prepare_callback._h3_film = True
    latent_preview.prepare_callback = prepare_callback
