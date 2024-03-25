import os


MotionDIr = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../checkpoints", "motion"
)


MODEL_CFG = {
    "musev_referencenet": {
        "net": os.path.join(MotionDIr, "musev_referencenet"),
        "desp": "",
    },
}
