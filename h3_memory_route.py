"""GET /h3/memory - what ComfyUI holds on the card right now, as a studio panel shows it.

The pack keeps the text encoder and the DiT resident between looks, lets the VAEs go before a take and takes
them back after; nothing said which of them sat on the card at a given moment, and the only way to know was
nvidia-smi's one number. This answers with each loaded model's name and how much of it is on the device.
"""
import logging

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from server import PromptServer
    from aiohttp import web
    import comfy.model_management as mm

    def _name(lm):
        try:
            inner = getattr(lm.model, "model", None)
            n = type(inner).__name__ if inner is not None else type(lm.model).__name__
            # CLIP patchers wrap the encoder: say what is inside.
            if n in ("ModelPatcher", "ModelPatcherDynamic") and inner is not None:
                n = type(getattr(inner, "model", inner)).__name__
            return n
        except Exception:
            return "?"

    @PromptServer.instance.routes.get("/h3/memory")
    async def h3_memory(request):
        items = []
        for lm in list(mm.current_loaded_models):
            try:
                items.append({
                    "name": _name(lm),
                    "loaded_bytes": int(lm.model_loaded_memory()),
                    "size_bytes": int(lm.model_memory()),
                    "device": str(lm.device),
                    "in_use": bool(getattr(lm, "currently_used", False)),
                })
            except Exception as e:  # a model half unloaded: say so, do not fail the whole answer
                items.append({"name": _name(lm), "error": str(e)[:120]})
        dev = mm.get_torch_device()
        return web.json_response({
            "models": items,
            "device": str(dev),
            "vram_total_bytes": int(mm.get_total_memory(dev)),
            "vram_free_bytes": int(mm.get_free_memory(dev)),
        })
except Exception as e:  # pragma: no cover - outside ComfyUI (tests) there is no server to hang a route on
    logging.info("[H3-Multishot] /h3/memory not registered (%s)", e)
