import os
import gradio as gr
import pdb
import cuid

from gradio_videocreation_text2video import online_t2v_inference
from gradio_videocreation_video2video import online_v2v_inference




def update_shape(image):
    h,w,_=image.shape

    
    return w,h

with gr.Blocks() as demo:
    gr.Markdown("V Demo")
    with gr.Tab("Text to Video"):
        with gr.Row():
            with gr.Column():
                prompt=gr.Textbox(label='Prompt')
                image=gr.Image(label="Input Image")
                seed=gr.Number(label='Seed')
                video_length=gr.Number(label='Video Length',value=12)
                fps=gr.Number(label='Generate Video FPS',value=12)
                gr.Markdown("If W&H is None, then use the Reference Image's Size")
                with gr.Row():
                    w=gr.Number(label='Width',value=512)
                    h=gr.Number(label='Height',value=704)
                btn1 = gr.Button("Run")
            out = gr.outputs.Video()
            # pdb.set_trace()
            
        image.change(fn=update_shape,inputs=[image],outputs=[w,h])
        
        btn1.click(fn=online_t2v_inference, inputs=[prompt,image,seed,fps,w,h,video_length], outputs=out)

    with gr.Tab("Video to Video"):
        with gr.Row():
            with gr.Column():
                prompt=gr.Textbox(label='Prompt')
                image=gr.Image(label="Input Image")
                video=gr.Video(label="Input Video")
                processor=gr.Dropdown(label="Condition Processor",choices=['pose', 'pose_body', 'pose_hand', 'pose_face', 'pose_hand_body', 'pose_hand_face', 'dwpose', 'dwpose_face', 'dwpose_hand', 'dwpose_body', 'dwpose_body_hand', 'canny', 'tile', 'hed', 'hed_scribble', 'depth', 'pidi', 'normal_bae', 'lineart', 'lineart_anime', 'zoe', 'sam', 'mobile_sam', 'leres', 'content', 'face_detector'],value="Pose")
                seed=gr.Number(label='Seed')
                fps=gr.Number(label='Generate Video FPS')
                gr.Markdown("If W&H is None, then use the Reference Image's Size")
                with gr.Row():
                    w=gr.Number(label='Width',value=512)
                    h=gr.Number(label='Height',value=704)
                btn2 = gr.Button("Run")
            out1 = gr.outputs.Video()
        image.change(fn=update_shape,inputs=[image],outputs=[w,h])

        btn2.click(fn=online_v2v_inference, inputs=[prompt,image,video,processor,seed,fps,w,h], outputs=out1)


# Set the IP and port
ip_address = "0.0.0.0"  # Replace with your desired IP address
port_number = 12345      # Replace with your desired port number


demo.queue().launch(share=False, debug=True, server_name=ip_address, server_port=port_number)
