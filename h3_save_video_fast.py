"""H3SaveVideoFast: the finished take written with the card's own encoder.

ComfyUI's SaveVideo encodes with libx264 on the CPU. NVENC on the same card
does the same H.264 in a seventh of the time and makes a smaller file, which
then also travels faster to the studio (measured 2026-09-22 on a 5090, 260
frames at 768^2: libx264 3.3 s / 88 MB of noise, h264_nvenc 0.5 s / 16 MB).
Where NVENC is missing (no NVIDIA encoder in this ffmpeg, another vendor's
card) libx264 is used and nothing else changes.

The node takes the sampler's frames and audio directly, so CreateVideo is
not needed either. Its output entry is the one SaveVideo makes, so anything
reading the history finds the file the same way.
"""
import os


def _has_encoder(name):
    try:
        import av
        return name in av.codecs_available
    except Exception:
        return False


class H3SaveVideoFast:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "filename_prefix": ("STRING", {"default": "video/h3"}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
            "quality": ("INT", {"default": 21, "min": 1, "max": 51,
                                "tooltip": "Constant-quality target (lower is "
                                           "better and bigger): NVENC cq / "
                                           "x264 crf. 19-23 is visually "
                                           "lossless for a take."}),
            "encoder": (["auto", "h264_nvenc", "libx264"], {
                "default": "auto",
                "tooltip": "auto: NVENC when this ffmpeg has it, else x264."}),
        }, "optional": {
            "audio": ("AUDIO",),
        }}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "video/minimax"
    DESCRIPTION = "Writes the take as MP4 with the card's H.264 encoder (NVENC), x264 when there is none."

    def save(self, images, filename_prefix, fps, quality, encoder, audio=None):
        import av
        import numpy as np
        import torch
        from fractions import Fraction
        import folder_paths

        n, h, w, _ = images.shape
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), w, h)
        file = "%s_%05d_.mp4" % (filename, counter)
        path = os.path.join(full_output_folder, file)

        codec = encoder
        if codec == "auto":
            codec = "h264_nvenc" if _has_encoder("h264_nvenc") else "libx264"
        if codec == "h264_nvenc" and not _has_encoder("h264_nvenc"):
            print("[H3SaveVideoFast] no NVENC in this ffmpeg - libx264 instead",
                  flush=True)
            codec = "libx264"
        if codec == "h264_nvenc":
            options = {"preset": "p5", "tune": "hq", "rc": "vbr",
                       "cq": str(int(quality)), "b:v": "0",
                       "profile": "high", "spatial_aq": "1", "temporal_aq": "1"}
        else:
            options = {"crf": str(int(quality)), "preset": "medium"}

        rate = Fraction(fps).limit_denominator(1000)

        def write(codec, options):
            container = av.open(path, "w", format="mp4")
            vs = container.add_stream(codec, rate=rate, options=options)
            vs.width, vs.height = w, h
            vs.pix_fmt = "yuv420p"
            # Even dimensions are what yuv420p needs; a 32-grid frame already has them.
            aus = None
            wave = None
            if audio is not None and isinstance(audio, dict) and audio.get("waveform") is not None:
                wave = audio["waveform"]
                if torch.is_tensor(wave):
                    wave = wave[0] if wave.ndim == 3 else wave
                    wave = wave.detach().float().cpu().numpy()
                sr = int(audio.get("sample_rate", 48000))
                ch = int(wave.shape[0]) if wave.ndim == 2 else 1
                aus = container.add_stream("aac", rate=sr)
                aus.layout = "stereo" if ch >= 2 else "mono"
            try:
                for i in range(n):
                    frame = (images[i].detach().float().clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
                    vf = av.VideoFrame.from_ndarray(frame, format="rgb24")
                    for pkt in vs.encode(vf):
                        container.mux(pkt)
                for pkt in vs.encode():
                    container.mux(pkt)
                if aus is not None and wave is not None:
                    if wave.ndim == 1:
                        wave = wave[None, :]
                    if aus.layout.name == "stereo" and wave.shape[0] == 1:
                        wave = np.repeat(wave, 2, axis=0)
                    if aus.layout.name == "mono" and wave.shape[0] > 1:
                        wave = wave[:1]
                    af = av.AudioFrame.from_ndarray(np.ascontiguousarray(wave.astype(np.float32)),
                                                    format="fltp", layout=aus.layout.name)
                    af.sample_rate = int(audio.get("sample_rate", 48000))
                    af.pts = 0
                    for pkt in aus.encode(af):
                        container.mux(pkt)
                    for pkt in aus.encode():
                        container.mux(pkt)
            finally:
                container.close()

        try:
            write(codec, options)
        except Exception as e:
            # An ffmpeg that has NVENC on a card that will not open it: a rented container without the
            # driver's video capability says "OpenEncodeSessionEx failed: unsupported device" at the first
            # frame, after the whole take was sampled. x264 then, and the take is not lost.
            if codec != "libx264":
                print("[H3SaveVideoFast] %s refused (%s) - libx264 instead" % (codec, str(e).splitlines()[0][:160]), flush=True)
                try:
                    os.remove(path)
                except OSError:
                    pass
                codec = "libx264"
                write(codec, {"crf": str(int(quality)), "preset": "medium"})
            else:
                raise
        size_mb = os.path.getsize(path) / 1e6
        print("[H3SaveVideoFast] %s: %d frames, %s, %.1f MB" % (file, n, codec, size_mb), flush=True)
        return {"ui": {"images": [{"filename": file, "subfolder": subfolder, "type": "output"}]}}


NODE_CLASS_MAPPINGS = {"H3SaveVideoFast": H3SaveVideoFast}
NODE_DISPLAY_NAME_MAPPINGS = {"H3SaveVideoFast": "H3 Save Video (NVENC)"}
