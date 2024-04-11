import os
import time
import pdb

import PIL.Image
import cuid
import gradio as gr
import spaces
import numpy as np

import PIL
from huggingface_hub import snapshot_download

ProjectDir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CheckpointsDir = os.path.join(ProjectDir, "checkpoints")
ignore_video2video = True
max_image_edge = 960


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
if not ignore_video2video:
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
    img_edge_ratio, _, _ = limit_shape(
        image_np, w, h, img_edge_ratio, max_image_edge=max_image_edge
    )
    if not isinstance(image_np, np.ndarray):  # None
        raise gr.Error("Need input reference image")
    return online_t2v_inference(
        prompt, image_np, seed, fps, w, h, video_len, img_edge_ratio
    )


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
    video_length,
    img_edge_ratio,
):
    img_edge_ratio, _, _ = limit_shape(
        image_np, w, h, img_edge_ratio, max_image_edge=max_image_edge
    )
    if not isinstance(image_np, np.ndarray):  # None
        raise gr.Error("Need input reference image")
    return online_v2v_inference(
        prompt,
        image_np,
        video,
        processor,
        seed,
        fps,
        w,
        h,
        video_length,
        img_edge_ratio,
    )


def limit_shape(image, input_w, input_h, img_edge_ratio, max_image_edge=max_image_edge):
    """limite generation video shape to avoid gpu memory overflow"""
    if input_h == -1 and input_w == -1:
        if isinstance(image, np.ndarray):
            input_h, input_w, _ = image.shape
        elif isinstance(image, PIL.Image.Image):
            input_w, input_h = image.size
        else:
            raise ValueError(
                f"image should be in [image, ndarray], but given {type(image)}"
            )
    if img_edge_ratio == 0:
        img_edge_ratio = 1
    img_edge_ratio_infact = min(max_image_edge / max(input_h, input_w), img_edge_ratio)
    # print(
    #     image.shape,
    #     input_w,
    #     input_h,
    #     img_edge_ratio,
    #     max_image_edge,
    #     img_edge_ratio_infact,
    # )
    if img_edge_ratio != 1:
        return (
            img_edge_ratio_infact,
            input_w * img_edge_ratio_infact,
            input_h * img_edge_ratio_infact,
        )
    else:
        return img_edge_ratio_infact, -1, -1


def limit_length(length):
    """limite generation video frames numer to avoid gpu memory overflow"""

    if length > 24 * 6:
        gr.Warning("Length need to smaller than 144, dute to gpu memory limit")
        length = 24 * 6
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
                seed = gr.Number(
                    label="Seed (seed=-1 means that the seeds run each time are different)",
                    value=-1,
                )
                video_length = gr.Number(
                    label="Video Length(need smaller than 144,If you want to be able to generate longer videos, run it locally )",
                    value=12,
                )
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
                    out_w = gr.Number(label="Output Width", value=0, interactive=False)
                    out_h = gr.Number(label="Output Height", value=0, interactive=False)
                    img_edge_ratio_infact = gr.Number(
                        label="img_edge_ratio in fact",
                        value=1.0,
                        interactive=False,
                    )
                btn1 = gr.Button("Generate")
            out = gr.Video()
            # pdb.set_trace()
        i2v_examples_256 = [
            [
                "(masterpiece, best quality, highres:1),(1boy, solo:1),(eye blinks:1.8),(head wave:1.3)",
                "../../data/images/yongen.jpeg",
            ],
            [
                "(masterpiece, best quality, highres:1), peaceful beautiful sea scene",
                "../../data/images/seaside4.jpeg",
            ],
        ]
        with gr.Row():
            gr.Examples(
                examples=i2v_examples_256,
                inputs=[prompt, image],
                outputs=[out],
                fn=hf_online_t2v_inference,
                cache_examples=False,
            )
        img_edge_ratio.change(
            fn=limit_shape,
            inputs=[image, w, h, img_edge_ratio],
            outputs=[img_edge_ratio_infact, out_w, out_h],
        )

        video_length.change(
            fn=limit_length, inputs=[video_length], outputs=[video_length]
        )

        btn1.click(
            fn=hf_online_t2v_inference,
            inputs=[
                prompt,
                image,
                seed,
                fps,
                w,
                h,
                video_length,
                img_edge_ratio_infact,
            ],
            outputs=out,
        )

    with gr.Tab("Video to Video"):
        if ignore_video2video:
            gr.Markdown(
                (
                    "Due to GPU limit, MuseVDemo now only support Text2Video. If you want to try Video2Video, please run it locally. \n"
                    "We are trying to support video2video in the future. Thanks for your understanding."
                )
            )
        else:
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
                    seed = gr.Number(
                        label="Seed (seed=-1 means that the seeds run each time are different)",
                        value=-1,
                    )
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
                        out_w = gr.Number(label="Width", value=0, interactive=False)
                        out_h = gr.Number(label="Height", value=0, interactive=False)
                        img_edge_ratio_infact = gr.Number(
                            label="img_edge_ratio in fact",
                            value=1.0,
                            interactive=False,
                        )
                    btn2 = gr.Button("Generate")
                out1 = gr.Video()

            v2v_examples_256 = [
                [
                    "(masterpiece, best quality, highres:1), harley quinn is dancing, animation, by joshua klein",
                    "../../data/demo/cyber_girl.png",
                    "../../data/demo/video1.mp4",
                ],
            ]
            with gr.Row():
                gr.Examples(
                    examples=v2v_examples_256,
                    inputs=[prompt, image, video],
                    outputs=[out],
                    fn=hg_online_v2v_inference,
                    cache_examples=False,
                )

            img_edge_ratio.change(
                fn=limit_shape,
                inputs=[image, w, h, img_edge_ratio],
                outputs=[img_edge_ratio_infact, out_w, out_h],
            )
            video_length.change(
                fn=limit_length, inputs=[video_length], outputs=[video_length]
            )
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
                    img_edge_ratio_infact,
                ],
                outputs=out1,
            )


# Set the IP and port
ip_address = "0.0.0.0"  # Replace with your desired IP address
port_number = 7860  # Replace with your desired port number


demo.queue().launch(
    share=True, debug=True, server_name=ip_address, server_port=port_number
)
