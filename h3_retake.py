"""H3 Retake - regenerate one stretch of a finished H3 clip and keep the rest.

Feed it a rendered clip (frames + audio) and a time window. Everything outside the window is
frozen as raw latents (noise mask 0); only the window is denoised, from a prompt written for
that moment. Video and audio are independent:

  video + audio  - redo the moment completely.
  video only     - keep the performance: voice, timing and room tone untouched.
  audio only     - keep the picture, change the line. Lips are whatever was rendered, so keep
                   the new line about as long as the old one.

Port of the LTX-2.5 retake to H3's AV latent: video [1,24,T,h/16,w/16] on the 17k+5 frame
grid, audio [1,32,2,Ta] at 40 latent fps (time is the LAST axis on the audio side - the one
real difference from the LTX port). H3 samples with a BasicGuider at cfg 1, so there is no
negative branch; wire the same SAMPLER and SIGMAS the clip was rendered with.
"""
import math
import time

import torch

import comfy.model_management
import comfy.nested_tensor
import node_helpers

from comfy_extras import nodes_custom_sampler as ncs

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

MODES = ["video + audio (redo the moment)",
         "video only (keep the performance)",
         "audio only (keep the picture)"]


def _mm_utils():
    """h3_multishot_utils + the comfy H3 module, with the pack's loose-install fallback."""
    try:
        from . import h3_multishot_utils as u
    except ImportError:
        try:
            import h3_multishot_utils as u
        except ImportError:
            import importlib.util as _ilu, os as _os
            _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "h3_multishot_utils.py")
            _s = _ilu.spec_from_file_location("h3_multishot_utils", _p)
            u = _ilu.module_from_spec(_s)
            _s.loader.exec_module(u)
    from comfy_extras import nodes_minimax_h3 as mmh3
    return u, mmh3


class H3Retake:
    """Redo one time window of a finished H3 clip (picture, sound, or both)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "video_vae": ("VAE",),
            "audio_vae": ("VAE",),
            "images": ("IMAGE", {"tooltip": "The finished clip's frames, in order."}),
            "audio": ("AUDIO", {"tooltip": "That clip's audio - the same take as the frames."}),
            "prompt": ("STRING", {"multiline": True, "default": "", "tooltip":
                       "What should happen in the window. Write it like a shot prompt; the model "
                       "sees only this text plus the frozen material either side."}),
            "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 600.0, "step": 0.1, "tooltip":
                              "Where the retake starts. Snapped to H3's latent grid (~0.14 s per slot)."}),
            "end_seconds": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 600.0, "step": 0.1}),
            "mode": (MODES, {"default": MODES[0]}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "sampler": ("SAMPLER", {"tooltip": "The sampler the clip was rendered with (euler)."}),
            "sigmas": ("SIGMAS", {"tooltip": "The schedule the clip was rendered with (beta, 10-12 steps; "
                                  "wire the same sigma-shift chain as the render canvas)."}),
        }, "optional": {
            "context_seconds": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 30.0, "step": 0.5, "tooltip":
                                "Only this much of the clip on either side of the window is sampled with it, "
                                "the rest is never touched: a 3 s retake of a 20 s clip costs 7 s of sampling, "
                                "not 20. 0 = the whole clip, as before (measured 2026-09-24 on a 5090: 65 s -> "
                                "about 25 s for that case)."}),
        }}

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "info")
    FUNCTION = "run"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = "Redo one stretch of a finished H3 clip - picture, sound, or both - and keep the rest."

    @staticmethod
    def _window(n_latent, total_seconds, start_s, end_s):
        per = total_seconds / float(max(1, n_latent))     # seconds per latent slot
        a = int(max(0.0, start_s) / per)
        b = int(round(min(end_s, total_seconds) / per + 0.5))
        a = max(0, min(a, n_latent - 1))
        b = max(a + 1, min(b, n_latent))
        return a, b

    def run(self, model, clip, video_vae, audio_vae, images, audio, prompt,
            start_seconds, end_seconds, mode, seed, sampler, sigmas, context_seconds=2.0):
        if end_seconds <= start_seconds:
            raise ValueError("H3 Retake: end_seconds must be after start_seconds.")
        u, mmh3 = _mm_utils()
        t0 = time.time()
        do_video = not mode.startswith("audio only")
        do_audio = not mode.startswith("video only")
        fps = float(mmh3.FPS)

        # ---- the stretch sampled: the window plus its context on either side, on H3's 17k+5 grid. The
        # rest of the clip is never encoded, sampled or decoded - it is spliced back as it was.
        frames = images[..., :3]
        n_px = frames.shape[0]
        a0, a1 = 0, n_px
        if context_seconds > 0:
            a0 = max(0, int(math.floor((start_seconds - context_seconds) * fps)))
            a1 = min(n_px, int(math.ceil((end_seconds + context_seconds) * fps)))
            need = max(22, a1 - a0)
            length = 5 + 17 * int(math.ceil((need - 5) / 17.0))
            a1 = min(n_px, a0 + length)
            a0 = max(0, a1 - length)
        sub = frames[a0:a1]
        n_sub = sub.shape[0]
        keep = 5 + ((n_sub - 5) // 17) * 17 if n_sub >= 22 else n_sub
        if keep != n_sub:
            print("[H3Retake] %d frames -> %d (H3's 17k+5 grid; what lies beyond the grid is "
                  "kept untouched)" % (n_sub, keep), flush=True)
        grid_frames = sub[:keep]
        h, w = grid_frames.shape[1], grid_frames.shape[2]
        resized = h % 32 or w % 32
        if resized:
            tw, th = max(32, round(w / 32) * 32), max(32, round(h / 32) * 32)
            grid_frames = mmh3._resize(grid_frames, tw, th, "disabled")
        total_seconds = keep / fps
        offset = a0 / fps
        # the window, in the stretch's own time
        start_in = max(0.0, start_seconds - offset)
        end_in = min(total_seconds, end_seconds - offset)
        if a0 > 0 or a0 + keep < n_px:
            print("[H3Retake] sampling frames %d-%d of %d (%.1f-%.1f s with %.1f s of context); the rest is "
                  "spliced back as it was" % (a0, a0 + keep, n_px, offset, offset + total_seconds, context_seconds), flush=True)

        # ---- encode both sides to raw latents
        vz = video_vae.encode(grid_frames)                          # [1, 24, T, h/16, w/16]
        wav_all, vae_sr = u._wav_for_vae(audio_vae, audio, "retake audio")
        s0 = int(round(offset * vae_sr))
        s1 = int(round((offset + total_seconds) * vae_sr))
        wav = wav_all[..., s0:s1]
        az = audio_vae.encode(wav.movedim(1, -1))                   # [1, 32, 2, Ta] - time LAST
        # audio no longer than the video window
        ta_want = round(total_seconds * mmh3.AUDIO_LATENT_FPS)
        if az.shape[-1] > ta_want:
            az = az[..., :ta_want]

        mv, ma = torch.zeros_like(vz), torch.zeros_like(az)
        vw = aw = None
        if do_video:
            i, j = self._window(vz.shape[2], total_seconds, start_in, end_in)
            mv[:, :, i:j] = 1.0
            vw = (i, j, vz.shape[2])
        if do_audio:
            i, j = self._window(az.shape[-1], total_seconds, start_in, end_in)
            ma[..., i:j] = 1.0
            aw = (i, j, az.shape[-1])

        latent = {"samples": comfy.nested_tensor.NestedTensor((vz, az)),
                  "noise_mask": comfy.nested_tensor.NestedTensor((mv, ma))}

        # ---- H3 conditioning: cfg 1, BasicGuider, no negative branch
        tokens = clip.tokenize(prompt)
        cond = clip.encode_from_tokens_scheduled(tokens)
        guider = ncs.BasicGuider().get_guider(model, cond)[0]
        noise = ncs.RandomNoise().get_noise(seed)[0]
        out, _denoised = ncs.SamplerCustomAdvanced().sample(noise, guider, sampler, sigmas, latent)

        # ---- decode, then splice the stretch back into the clip: the frames and the sound outside it are
        # the originals, bit for bit; inside it, the retake.
        lat = out["samples"]
        if getattr(lat, "is_nested", False):
            lat = lat.unbind()[0]
        imgs = video_vae.decode(lat)
        if imgs.ndim == 5:
            imgs = imgs.reshape(-1, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1])
        if resized:
            imgs = mmh3._resize(imgs, w, h, "disabled")
        imgs = imgs[:keep].cpu()
        if do_video:
            imgs = torch.cat([frames[:a0].cpu(), imgs, frames[a0 + keep:].cpu()], 0)
        else:
            imgs = frames.cpu()
        from comfy_extras.nodes_audio import vae_decode_audio
        if do_audio:
            dec = vae_decode_audio(audio_vae, out)
            out_sr = int(dec["sample_rate"])
            whole = wav_all
            if out_sr != vae_sr:
                import torchaudio
                whole = torchaudio.functional.resample(wav_all, vae_sr, out_sr)
            d0 = int(round(offset * out_sr))
            d1 = int(round((offset + total_seconds) * out_sr))
            piece = dec["waveform"][..., :d1 - d0].cpu()
            # the decoder's level is its own (normalised on the way out): the piece is brought to the level the
            # original has in the same stretch, so nothing jumps at the splice
            ref = whole[..., d0:d1].cpu()
            n = min(piece.shape[-1], ref.shape[-1])
            if n > 0:
                gain = (ref[..., :n].pow(2).mean().sqrt() / piece[..., :n].pow(2).mean().sqrt().clamp_min(1e-6)).clamp(0.05, 20.0)
                piece = piece * gain
            if piece.shape[-1] < d1 - d0:
                piece = torch.nn.functional.pad(piece, (0, d1 - d0 - piece.shape[-1]))
            aud = {"waveform": torch.cat([whole[..., :d0].cpu(), piece, whole[..., d1:].cpu()], -1), "sample_rate": out_sr}
        else:
            aud = {"waveform": wav_all.cpu(), "sample_rate": vae_sr}

        info = ("retake %.1f-%.1f s of a %.1f s clip | %s | sampled %.1f-%.1f s | video slots %s | audio slots %s | %.0f s"
                % (start_seconds, end_seconds, n_px / fps, mode, offset, offset + total_seconds,
                   ("%d-%d of %d" % vw) if vw else "frozen",
                   ("%d-%d of %d" % aw) if aw else "frozen", time.time() - t0))
        print("[H3Retake] " + info, flush=True)
        comfy.model_management.soft_empty_cache()
        return (imgs, aud, info)


NODE_CLASS_MAPPINGS["H3Retake"] = H3Retake
NODE_DISPLAY_NAME_MAPPINGS["H3Retake"] = "H3 Retake (redo part of a clip)"
