import os


LCMDir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../checkpoints", "lcm"
)


MODEL_CFG = {
    "lcm": {
        os.path.join(LCMDir, "lcm-lora-sdv1-5/pytorch_lora_weights.safetensors"): {
            "strength": 1.0,
            "lora_block_weight": "ALL",
            "strength_offset": 0,
        },
    },
}
