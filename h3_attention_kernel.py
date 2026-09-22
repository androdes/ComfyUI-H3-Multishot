"""H3AttentionKernel: pick the attention kernel the video model runs on.

ComfyUI registers every attention implementation it can find (pytorch,
sage, sage3, flash, comfy_kitchen_int8, ...) but chooses one for the whole
process from the launch line, and a checkpoint that carries an int8
attention config pins its own. This node puts a chosen kernel on the
model's attention-override hook instead, so one take can run on
SageAttention 3 (Blackwell fp4) while the process default stays what it
is. Nothing changes when the kernel is not installed: the model comes back
untouched and a line says so.

Measured 2026-09-22 on a 5090, one 11 s shot at 768^2, 8 steps:
pytorch 98 s (Sage 1 on the launch line) -> comfy_kitchen_int8 73 s.
"""

KERNELS = ["sage3", "comfy_kitchen_int8", "sage", "flash", "pytorch", "(default)"]


def _make_override(fn, name):
    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):
        # The kernels decline masks and non-cuda tensors themselves and fall
        # back to pytorch attention; nothing to guard here.
        return fn(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                  skip_reshape=skip_reshape,
                  skip_output_reshape=skip_output_reshape, **kwargs)
    override.h3_kernel = name
    return override


class H3AttentionKernel:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "kernel": (KERNELS, {
                "default": "(default)",
                "tooltip": "sage3: SageAttention 3 (Blackwell fp4, needs the "
                           "sageattn3 package). comfy_kitchen_int8: ComfyUI's "
                           "own int8 kernel. sage / flash: those packages. "
                           "pytorch: scaled_dot_product_attention. "
                           "(default): whatever the process was launched "
                           "with."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "video/minimax"
    DESCRIPTION = "Runs this model's attention on a chosen kernel (SageAttention 3, int8, flash...)."

    def apply(self, model, kernel):
        if kernel == "(default)":
            return (model,)
        try:
            from comfy.ldm.modules.attention import get_attention_function
            fn = get_attention_function(kernel, None)
        except Exception as e:
            print("[H3AttentionKernel] ComfyUI has no attention registry "
                  "here (%s) - model unchanged" % e, flush=True)
            return (model,)
        if fn is None:
            print("[H3AttentionKernel] kernel %r is not available in this "
                  "ComfyUI (package not installed?) - model unchanged"
                  % kernel, flush=True)
            return (model,)
        m = model.clone()
        opts = m.model_options.setdefault("transformer_options", {})
        current = opts.get("optimized_attention_override")
        if getattr(current, "h3_kernel", None) == kernel:
            return (m,)
        opts["optimized_attention_override"] = _make_override(fn, kernel)
        print("[H3AttentionKernel] attention runs on %s" % kernel, flush=True)
        return (m,)


NODE_CLASS_MAPPINGS = {"H3AttentionKernel": H3AttentionKernel}
NODE_DISPLAY_NAME_MAPPINGS = {"H3AttentionKernel": "H3 Attention Kernel"}
