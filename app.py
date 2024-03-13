import os
import argparse
import gradio as gr
import numpy as np
import torch
from ram_pipeline import RAM
from PIL import Image
import time


parser = argparse.ArgumentParser(prog=__file__)
parser.add_argument('--input', type=str, default='./test_imgs', help='path of input')
parser.add_argument('--dev_id', type=int, default=0, help='device id')
parser.add_argument('--conf_thresh', type=float, default=0.25, help='det confidence threshold')
parser.add_argument('--nms_thresh', type=float, default=0.7, help='det nms threshold')
parser.add_argument('--single_output', action='store_true', help='det confidence threshold')
args = parser.parse_args()

ram = RAM()

def recognize_anything(img_path, return_anno_img=False):
    if img_path is None or return_anno_img is None:
        return '[ INVALID INPUT ]', None
    ram_res = ram(img_path, return_bbox=return_anno_img)
    tags = [ram_res['tag_ch'][i]+ram_res['tag_en'][i] for i in range(len(ram_res['tag_ch']))]
    print(tags)
    return ' || '.join(tags), ram_res['img_res']


# Description
title = f"<center><strong><font size='8'>万物检测⭐powered by 1684x <font></strong></center>"

description_e = f"""### 这是在1684X上部署[Recognize Anything Model (RAM)](https://github.com/xinyu1205/recognize-anything)+GroundingDINO实现万物检测的示例。             
              """

description_p = """ # 使用方法

                1. 上传/拍摄需要检测的图像。
                2. 若需要框出物体位置，请勾选“标注位置”。

              """

default_example = ["./resources/image/demo1.jpg"]

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"


with gr.Blocks(css=css, title="万物检测") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            # Title
            gr.Markdown(title)
            gr.Markdown(description_e)
    
    with gr.Row():
        with gr.Column():
            img_inputs = gr.Image(label="选择图片", value=default_example[0], type='filepath')
            using_dino = gr.Checkbox(label="标注位置", value=True)
        with gr.Column():
            tags_area = gr.Textbox(label='标签', interactive=False)
            annotated_img = gr.Image(label="检测结果", interactive=False)
            

    # Submit & Clear
    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    det_btn_p = gr.Button(
                        "检测", variant="primary"
                    )
                    clear_btn_p = gr.Button("清空", variant="secondary")


        with gr.Column():
            # Description
            gr.Markdown(description_p)

    det_btn_p.click(
        recognize_anything, inputs=[img_inputs, using_dino], outputs=[tags_area, annotated_img]#, json_str]
    )
    def clear():
        return [None, None, None]

    clear_btn_p.click(clear, outputs=[img_inputs, tags_area, annotated_img])

demo.queue()
demo.launch(ssl_verify=False, server_name="0.0.0.0")
