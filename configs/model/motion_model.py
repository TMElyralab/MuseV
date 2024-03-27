import os


MotionDIr = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../checkpoints", "motion"
)


MODEL_CFG = {
    "musev": {
        "unet": os.path.join(MotionDIr, "musev"),
        "desp": "only train unet motion module, fix t2i",
    },
    "musev_referencenet": {
        "unet": os.path.join(MotionDIr, "musev_referencenet"),
        "desp": "train referencenet, IPAdapter and unet motion module, fix t2i",
    },
    "musev_referencenet_pose": {
        "unet": os.path.join(MotionDIr, "musev_referencenet_pose"),
        "desp": "train  unet motion module and IPAdapter, fix t2i and referencenet",
    },
}
