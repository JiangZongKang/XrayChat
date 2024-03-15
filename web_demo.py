#!/usr/bin/env python

import gradio as gr
from PIL import Image
import os
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModel


chat_meta = """You are an AI assistant whose name is XrayChat (Xray多模态大模型).
- XrayChat (Xray多模态大模型) is a multi-modality conversational language model that is developed by XrayChat Team (Xray多模态大模型团队). 
- XrayChat (Xray多模态大模型) can understand and communicate fluently in the language chosen by the user such as English and 中文.
- XrayChat (Xray多模态大模型) is capable of comprehending and articulating responses effectively based on the provided image.
"""


def generate_text_with_image(input_text, image_prompt, history=[], request_data=dict(), is_zh=True):
    input_para = {
        "max_length": 2048,
        "min_length": 50,
        "temperature": 0.8,
        "top_p": 0.4,
        "top_k": 100,
        "repetition_penalty": 1.2
    }
    input_para.update(request_data)

    print(input_text)
    print(history)
    print(image_prompt)
    print(input_para["temperature"])
    print(input_para["top_p"])
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            answer, history = model.chat(tokenizer, query=input_text, history=history, image=image_prompt, \
                                        do_sample=False, meta_instruction=chat_meta)
                            # top_p=input_para['top_p'], \
                            # temperature=input_para['temperature'])
    return answer


def request_model(input_text, temperature, top_p, image_prompt, result_previous):
    result_text = [(ele[0], ele[1]) for ele in result_previous]
    for i in range(len(result_text)-1, -1, -1):
        if result_text[i][0] == "" or result_text[i][1] == "":
            del result_text[i]
    print(f"history {result_text}")

    if len(result_text) > 0:
        input_text = input_text
    else:
        input_text = '<ImageHere> ' + input_text

    is_zh = True
    if image_prompt is None:
        if is_zh:
            result_text.append((input_text, '图片为空！请上传图片并重试。'))
        else:
            result_text.append((input_text, 'Image empty! Please upload a image and retry.'))
        return input_text, result_text
    elif input_text == "":
        result_text.append((input_text, 'Text empty! Please enter text and retry.'))
        return "", result_text                

    request_para = {"temperature": temperature, "top_p": top_p}
    # image = Image.open(image_prompt)
    try:
        answer = generate_text_with_image(input_text, image_prompt, result_text.copy(), request_para, is_zh)
    except Exception as e:
        print(f"error: {e}")
        if is_zh:
            result_text.append((input_text, '超时！请稍等几分钟再重试。'))
        else:
            result_text.append((input_text, 'Timeout! Please wait a few minutes and retry.'))
        return "", result_text

    result_text.append((input_text, answer))
    print(result_text)
    return "", result_text


DESCRIPTION = '''# <a href="https://github.com/JiangZongKang/XrayChat">XrayChat</a>'''

MAINTENANCE_NOTICE1 = 'Hint 1: If the app report "Something went wrong, connection error out", please turn off your proxy and retry.\nHint 2: If you upload a large size of image like 10MB, it may take some time to upload and process. Please be patient and wait.'
MAINTENANCE_NOTICE2 = '提示1: 如果应用报了“Something went wrong, connection error out”的错误，请关闭代理并重试。\n提示2: 如果你上传了很大的图片，比如10MB大小，那将需要一些时间来上传和处理，请耐心等待。'

NOTES = 'This app is adapted from <a href="https://github.com/JiangZongKang/XrayChat">https://github.com/JiangZongKang/XrayChat</a>.'


def clear_fn(value):
    return "", [("", "Hi, What do you want to know about this image?")], None

def clear_fn2(value):
    return [("", "Hi, What do you want to know about this image?")]


def main(args):
    global model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.from_pretrained, trust_remote_code=True).eval()
    model.half().cuda()

    with gr.Blocks(css='style.css') as demo:
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column(scale=4.5):
                with gr.Group():
                    input_text = gr.Textbox(label='Input Text', placeholder='Please enter text prompt below and press ENTER.')
                    with gr.Row():
                        run_button = gr.Button('Generate')
                        clear_button = gr.Button('Clear')

                    image_prompt = gr.Image(type="filepath", label="Image Prompt", value=None, height=350)
                with gr.Row():
                    temperature = gr.Slider(maximum=1, value=0.8, minimum=0, label='Temperature')
                    top_p = gr.Slider(maximum=1, value=0.4, minimum=0, label='Top P')
                with gr.Group():
                    with gr.Row():
                        maintenance_notice = gr.Markdown(MAINTENANCE_NOTICE1)
            with gr.Column(scale=6): # 5.5
                result_text = gr.components.Chatbot(label='Multi-round conversation History', value=[("", "Hi, What do you want to know about this image?")]).style(height=850) #550

        gr.Markdown(NOTES)

        print(gr.__version__)
        run_button.click(fn=request_model,inputs=[input_text, temperature, top_p, image_prompt, result_text],
                         outputs=[input_text, result_text])
        input_text.submit(fn=request_model,inputs=[input_text, temperature, top_p, image_prompt, result_text],
                         outputs=[input_text, result_text])
        clear_button.click(fn=clear_fn, inputs=clear_button, outputs=[input_text, result_text, image_prompt])
        image_prompt.upload(fn=clear_fn2, inputs=clear_button, outputs=[result_text])
        image_prompt.clear(fn=clear_fn2, inputs=clear_button, outputs=[result_text])

        print(gr.__version__)

    demo.queue(concurrency_count=10)
    demo.launch(share=args.share)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--from_pretrained", type=str, default="/datastore2/jiangzongkang/code/InternLM-XComposer/finetune/output/finetune_v2/checkpoint-3750", help='pretrained ckpt')
    args = parser.parse_args()
    main(args)