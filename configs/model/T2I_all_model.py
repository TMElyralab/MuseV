import os


T2IDir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../checkpoints", "t2i"
)

MODEL_CFG = {
    "majicmixRealv6Fp16": {
        "sd": os.path.join(T2IDir, "sd1.5/majicmixRealv6Fp16"),
    },
    "fantasticmix_v10": {
        "sd": os.path.join(T2IDir, "sd1.5/fantasticmix_v10"),
    },
}
