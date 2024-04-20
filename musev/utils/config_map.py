from diffusers import DDPMScheduler, DDIMScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler, LCMScheduler

AniASchedulerMap = {
    "DDPM".lower(): DDPMScheduler,
    "DDIM".lower(): DDIMScheduler,
    "EulerA".lower(): EulerAncestralDiscreteScheduler,
    "Euler".lower(): EulerDiscreteScheduler,
    "DPM".lower(): DPMSolverMultistepScheduler,
    "LCM".lower(): LCMScheduler,
}

def get_ania_scheduler(scheduler_name = "DDIM"):
    if scheduler_name.lower() not in AniASchedulerMap:
        raise NotImplementedError("Scheduler not supported")
    print(f"*** AniA, use {scheduler_name} scheduler ***")
    return AniASchedulerMap[scheduler_name.lower()]
