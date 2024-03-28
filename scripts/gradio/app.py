import os
import gradio as gr
import pdb
import cuid

from gradio_videocreation_text2video import online_t2v_inference
from gradio_videocreation_video2video import online_v2v_inference
from huggingface_hub import snapshot_download


def download_model():
    if not os.path.exists("../../checkpoints"):
        print("Checkpoint Not Downloaded, start downloading...")
        snapshot_download(
            repo_id="TMElyralab/MuseV",
            local_dir = "../../checkpoints",
            max_workers=8,
        )
    else:
        print("Already download the model.")
download_model()# for huggingface deployment.


def update_shape(image):
    h,w,_=image.shape
    return w,h

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

control_options = ['pose', 'pose_body', 'pose_hand', 'pose_face', 'pose_hand_body', 'pose_hand_face', 'dwpose', 'dwpose_face', 'dwpose_hand', 'dwpose_body', 'dwpose_body_hand', 'canny', 'tile', 'hed', 'hed_scribble', 'depth', 'pidi', 'normal_bae', 'lineart', 'lineart_anime', 'zoe', 'sam', 'mobile_sam', 'leres', 'content', 'face_detector']
concatenate_block = ConcatenateBlock(control_options)


css = """#input_img {max-width: 1024px !important} #output_vid {max-width: 1024px; max-height: 576px}"""


with gr.Blocks(css=css) as demo:
    gr.Markdown("<div align='center'> <h1> MuseV: Infinite-length and High Fidelity Virtual Human Video Generation with Visual Conditioned Parallel Denoising</span> </h1> \
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
                <a style='font-size:18px;color: #000000'>If MuseV is useful, please help star the repo~ </a> </div>" )
    with gr.Tab("Text to Video"):
        with gr.Row():
            with gr.Column():
                prompt=gr.Textbox(label='Prompt')
                image=gr.Image(label="Reference Image")
                seed=gr.Number(label='Seed')
                video_length=gr.Number(label='Video Length',value=12)
                fps=gr.Number(label='Generate Video FPS',value=12)
                gr.Markdown("Default use the Reference Image's Size.")
                with gr.Row():
                    w=gr.Number(label='Width',value=512)
                    h=gr.Number(label='Height',value=768)
                btn1 = gr.Button("Generate")
            out = gr.outputs.Video()
            # pdb.set_trace()
        with gr.Row():
            board = gr.Dataframe(value=[["", "", ""]] * 3, interactive=False, type="array",label='Demo Video')
            
        image.change(fn=update_shape,inputs=[image],outputs=[w,h])
        
        btn1.click(fn=online_t2v_inference, inputs=[prompt,image,seed,fps,w,h,video_length], outputs=out)

    with gr.Tab("Video to Video"):
        with gr.Row():
            with gr.Column():
                prompt=gr.Textbox(label='Prompt')
                image=gr.Image(label="Reference Image")
                video=gr.Video(label="Input Video")
                # radio = gr.inputs.Radio(, label="Select an option")
                # ctr_button = gr.inputs.Button(label="Add ControlNet List")
                # output_text = gr.outputs.Textbox()
                processor=gr.Textbox(label=f'Control Condition. Select from {control_options}', value='dwpose,')
                seed=gr.Number(label='Seed')
                video_length=gr.Number(label='Video Length',value=12)
                fps=gr.Number(label='Generate Video FPS',value=12)
                gr.Markdown("If W&H is None, then use the Reference Image's Size")
                with gr.Row():
                    w=gr.Number(label='Width',value=512)
                    h=gr.Number(label='Height',value=704)
                btn2 = gr.Button("Generate")
            out1 = gr.outputs.Video()
        image.change(fn=update_shape,inputs=[image],outputs=[w,h])

        btn2.click(fn=online_v2v_inference, inputs=[prompt,image,video,processor,seed,fps,w,h,video_length], outputs=out1)


# Set the IP and port
ip_address = "0.0.0.0"  # Replace with your desired IP address
port_number = 12345      # Replace with your desired port number


demo.queue().launch(share=False, debug=True, server_name=ip_address, server_port=port_number)