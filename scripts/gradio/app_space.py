import os
import time
import pdb

import cuid
import gradio as gr
import spaces
import numpy as np

from huggingface_hub import snapshot_download

ProjectDir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CheckpointsDir = os.path.join(ProjectDir, "checkpoints")




def download_model():
    if not os.path.exists(CheckpointsDir):
        print("Checkpoint Not Downloaded, start downloading...")
        tic = time.time()
        snapshot_download(
            repo_id="TMElyralab/MuseV",
            local_dir=CheckpointsDir,
            max_workers=8,
        )
        toc = time.time()
        print(f"download cost {toc-tic} seconds")
    else:
        print("Already download the model.")


download_model()  # for huggingface deployment.

from gradio_video2video import online_v2v_inference
from gradio_text2video import online_t2v_inference

@spaces.GPU(duration=180)
def hf_online_t2v_inference(
    prompt,
    image_np,
    seed,
    fps,
    w,
    h,
    video_len,
    img_edge_ratio,
):
    return online_t2v_inference(prompt,image_np,seed,fps,w,h,video_len,img_edge_ratio)

@spaces.GPU(duration=180)

def hg_online_v2v_inference(
    prompt,
    image_np,
    video,
    processor,
    seed,
    fps,
    w,
    h,
    video_length,img_edge_ratio,
):
    return online_v2v_inference(prompt,image_np,video,processor,seed,fps,w,h,video_length,img_edge_ratio)


def update_shape(image):
    if isinstance(image, np.ndarray):
        h, w, _ = image.shape
    else:
        h, w = 768, 512
    return w, h,w,h

def limit_shape(image,input_w,input_h,img_edge_ratio):
    
    if isinstance(image, np.ndarray):
        h, w, _ = image.shape
    else:
        h, w = -1, -1
    if input_h!=-1:
        h=input_h
    if input_w!=-1:
        w=input_w
        
    h=img_edge_ratio*h
    w=img_edge_ratio*w

    max_dim = 960
    if w > max_dim or h > max_dim:
        scale_ratio = min(max_dim / w, max_dim / h)
        new_w = int(w * scale_ratio)
        new_h = int(h * scale_ratio)
    else:
        new_w=w
        new_h=h
    return new_w, new_h

def limit_length(length):
    if length>24*6:
        length=24*6
    return length


class ConcatenateBlock(gr.blocks.Block):
    def __init__(self, options):
        self.options = options
        self.current_string = ""

    def update_string(self, new_choice):
        if new_choice and new_choice not in self.current_string.split(", "):
            if self.current_string == "":
                self.current_string = new_choice
            else:
                self.current_string += ", " + new_choice
        return self.current_string


def process_input(new_choice):
    return concatenate_block.update_string(new_choice), ""


control_options = [
    "pose",
    "pose_body",
    "pose_hand",
    "pose_face",
    "pose_hand_body",
    "pose_hand_face",
    "dwpose",
    "dwpose_face",
    "dwpose_hand",
    "dwpose_body",
    "dwpose_body_hand",
    "canny",
    "tile",
    "hed",
    "hed_scribble",
    "depth",
    "pidi",
    "normal_bae",
    "lineart",
    "lineart_anime",
    "zoe",
    "sam",
    "mobile_sam",
    "leres",
    "content",
    "face_detector",
]
concatenate_block = ConcatenateBlock(control_options)


css = """#input_img {max-width: 1024px !important} #output_vid {max-width: 1024px; max-height: 576px}"""


with gr.Blocks(css=css) as demo:
    gr.Markdown(
        "<div align='center'> <h1> MuseV: Infinite-length and High Fidelity Virtual Human Video Generation with Visual Conditioned Parallel Denoising</span> </h1> \
                    <h2 style='font-weight: 450; font-size: 1rem; margin: 0rem'>\
                    </br>\
                    Zhiqiang Xia <sup>*</sup>,\
                    Zhaokang Chen<sup>*</sup>,\
                    Bin Wu<sup>†</sup>,\
                    Chao Li,\
                    Kwok-Wai Hung,\
                    Chao Zhan,\
                    Yingjie He,\
                    Wenjiang Zhou\
                    (<sup>*</sup>Equal Contribution,  <sup>†</sup>Corresponding Author, benbinwu@tencent.com)\
                    </br>\
                    Lyra Lab, Tencent Music Entertainment\
                </h2> \
                <a style='font-size:18px;color: #000000' href='https://github.com/TMElyralab/MuseV'>[Github Repo]</a>\
                <a style='font-size:18px;color: #000000'>, which is important to Open-Source projects. Thanks!</a>\
                <a style='font-size:18px;color: #000000' href=''> [ArXiv(Coming Soon)] </a>\
                <a style='font-size:18px;color: #000000' href=''> [Project Page(Coming Soon)] </a> \
                <a style='font-size:18px;color: #000000'>If MuseV is useful, please help star the repo~ </a> </div>"
    )
    with gr.Tab("Text to Video"):
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt")
                image = gr.Image(label="VisionCondImage")
                seed = gr.Number(label="Seed (seed=-1 means that the seeds run each time are different)", value=-1)
                video_length = gr.Number(label="Video Length(need smaller than 144,If you want to be able to generate longer videos, run it locally )", value=12)
                fps = gr.Number(label="Generate Video FPS", value=6)
                gr.Markdown(
                    (
                        "If W&H is -1, then use the Reference Image's Size. Size of target video is $(W, H)*img\_edge\_ratio$. \n"
                        "The shorter the image size, the larger the motion amplitude, and the lower video quality.\n"
                        "The longer the W&H, the smaller the motion amplitude, and the higher video quality.\n"
                        "Due to the GPU VRAM limits, the W&H need smaller than 960px"
                    )
                )
                with gr.Row():
                    w = gr.Number(label="Width", value=-1)
                    h = gr.Number(label="Height", value=-1)
                    img_edge_ratio = gr.Number(label="img_edge_ratio", value=1.0)
                with gr.Row():
                    final_w = gr.Number(label="Generate Width", value=-1)
                    final_h = gr.Number(label="Generate Height", value=-1)
                btn1 = gr.Button("Generate")
            out = gr.outputs.Video()
            # pdb.set_trace()
        # with gr.Row():
        #     board = gr.Dataframe(
        #         value=[["", "", ""]] * 3,
        #         interactive=False,
        #         type="array",
        #         label="Demo Video",
        #     )

        image.change(fn=update_shape, inputs=[image], outputs=[w, h,final_w,final_h])

        w.change(fn=limit_shape,inputs=[image,w,h,img_edge_ratio],outputs=[final_w,final_h])
        h.change(fn=limit_shape,inputs=[image,w,h,img_edge_ratio],outputs=[final_w,final_h])
        img_edge_ratio.change(fn=limit_shape,inputs=[image,w,h,img_edge_ratio],outputs=[final_w,final_h])
        
        video_length.change(fn=limit_length,inputs=[video_length],outputs=[video_length])

        btn1.click(
            fn=hf_online_t2v_inference,
            inputs=[prompt, image, seed, fps, w, h, video_length, img_edge_ratio],
            outputs=out,
        )

    with gr.Tab("Video to Video"):
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt")
                gr.Markdown(
                    (
                        "pose of VisionCondImage should be same as of the first frame of the video. "
                        "its better generate target first frame whose pose is same as of first frame of the video with text2image tool, sch as MJ, SDXL."
                    )
                )
                image = gr.Image(label="VisionCondImage")
                video = gr.Video(label="ReferVideo")
                # radio = gr.inputs.Radio(, label="Select an option")
                # ctr_button = gr.inputs.Button(label="Add ControlNet List")
                # output_text = gr.outputs.Textbox()
                processor = gr.Textbox(
                    label=f"Control Condition. gradio code now only support dwpose_body_hand, use command can support multi of {control_options}",
                    value="dwpose_body_hand",
                )
                gr.Markdown("seed=-1 means that seeds are different in every run")
                seed = gr.Number(label="Seed (seed=-1 means that the seeds run each time are different)", value=-1)
                video_length = gr.Number(label="Video Length", value=12)
                fps = gr.Number(label="Generate Video FPS", value=6)
                gr.Markdown(
                    (
                        "If W&H is -1, then use the Reference Image's Size. Size of target video is $(W, H)*img\_edge\_ratio$. \n"
                        "The shorter the image size, the larger the motion amplitude, and the lower video quality.\n"
                        "The longer the W&H, the smaller the motion amplitude, and the higher video quality.\n"
                        "Due to the GPU VRAM limits, the W&H need smaller than 2000px"
                    )
                )
                with gr.Row():
                    w = gr.Number(label="Width", value=-1)
                    h = gr.Number(label="Height", value=-1)
                    img_edge_ratio = gr.Number(label="img_edge_ratio", value=1.0)
                with gr.Row():
                    final_w = gr.Number(label="Generate Width", value=-1)
                    final_h = gr.Number(label="Generate Height", value=-1)
                btn2 = gr.Button("Generate")
            out1 = gr.outputs.Video()
        image.change(fn=update_shape, inputs=[image], outputs=[w, h])
        final_w.change(fn=limit_shape,inputs=[image,img_edge_ratio],outputs=[final_w,final_h])
        final_h.change(fn=limit_shape,inputs=[image,img_edge_ratio],outputs=[final_w,final_h])
        
        video_length.change(fn=limit_length,inputs=[video_length],outputs=[video_length])
        
        btn2.click(
            fn=hg_online_v2v_inference,
            inputs=[
                prompt,
                image,
                video,
                processor,
                seed,
                fps,
                w,
                h,
                video_length,
                img_edge_ratio,
            ],
            outputs=out1,
        )


# Set the IP and port
ip_address = "0.0.0.0"  # Replace with your desired IP address
port_number = 7860  # Replace with your desired port number


demo.queue().launch(
    share=False, debug=True, server_name=ip_address, server_port=port_number
)
