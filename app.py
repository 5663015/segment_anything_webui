import os
import cv2
import sys
import numpy as np
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


models = {
    'vit_b': './checkpoints/sam_vit_b_01ec64.pth',
    'vit_l': './checkpoints/sam_vit_l_0b3195.pth',
    'vit_h': './checkpoints/sam_vit_h_4b8939.pth'
}

def inference(device, model_type, input_img, points_per_side, pred_iou_thresh, stability_score_thresh, min_mask_region_area,
              stability_score_offset, box_nms_thresh, crop_n_layers, crop_nms_thresh):
    sam = sam_model_registry[model_type](checkpoint=models[model_type]).to(device)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        stability_score_offset=stability_score_offset,
        box_nms_thresh=box_nms_thresh,
        crop_n_layers=crop_n_layers,
        crop_nms_thresh=crop_nms_thresh,
        crop_overlap_ratio=512 / 1500,
        crop_n_points_downscale_factor=1,
        point_grids=None,
        min_mask_region_area=min_mask_region_area,
        output_mode='binary_mask'
    )

    masks = mask_generator.generate(input_img)
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)

    img = np.ones((input_img.shape[0], input_img.shape[1], 3))
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[m==True, i] = color_mask[i]
    result = input_img / 255 * 0.3 + img * 0.7

    return result



with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown(
            '''# Segment Anything!🚀
            分割一切！CV的GPT-3时刻！
            [**官方网址**](https://segment-anything.com/)
            '''
        )
        with gr.Row():
            # 选择模型类型
            model_type = gr.Dropdown(["vit_b", "vit_l", "vit_h"], value='vit_b', label="选择模型")
            # 选择device
            device = gr.Dropdown(["cpu", "cuda"], value='cuda', label="选择你的硬件")

    # 参数
    with gr.Accordion(label='参数调整', open=False):
        with gr.Row():
            points_per_side = gr.Number(value=32, label="points_per_side", precision=0,
                                        info='''The number of points to be sampled along one side of the image. The total 
                                        number of points is points_per_side**2.''')
            pred_iou_thresh = gr.Slider(value=0.88, minimum=0, maximum=1.0, step=0.01, label="pred_iou_thresh",
                                        info='''A filtering threshold in [0,1], using the model's predicted mask quality.''')
            stability_score_thresh = gr.Slider(value=0.95, minimum=0, maximum=1.0, step=0.01, label="stability_score_thresh",
                                               info='''A filtering threshold in [0,1], using the stability of the mask under 
                                               changes to the cutoff used to binarize the model's mask predictions.''')
            min_mask_region_area = gr.Number(value=0, label="min_mask_region_area", precision=0,
                                             info='''If >0, postprocessing will be applied to remove disconnected regions 
                                             and holes in masks with area smaller than min_mask_region_area.''')
        with gr.Row():
            stability_score_offset = gr.Number(value=1, label="stability_score_offset",
                                               info='''The amount to shift the cutoff when calculated the stability score.''')
            box_nms_thresh = gr.Slider(value=0.7, minimum=0, maximum=1.0, step=0.01, label="box_nms_thresh",
                                       info='''The box IoU cutoff used by non-maximal ression to filter duplicate masks.''')
            crop_n_layers = gr.Number(value=0, label="crop_n_layers", precision=0,
                                      info='''If >0, mask prediction will be run again on crops of the image. 
                                      Sets the number of layers to run, where each layer has 2**i_layer number of image crops.''')
            crop_nms_thresh = gr.Slider(value=0.7, minimum=0, maximum=1.0, step=0.01, label="crop_nms_thresh",
                                        info='''The box IoU cutoff used by non-maximal suppression to filter duplicate 
                                        masks between different crops.''')

    # 显示图片
    with gr.Row().style(equal_height=True):
        with gr.Column():
            input_image = gr.Image(type="numpy")
            with gr.Row():
                button = gr.Button("Auto!")
        image_output = gr.Image(type='numpy')

    gr.Examples(
        examples=[os.path.join(os.path.dirname(__file__), "./images/53960-scaled.jpg"),
                  os.path.join(os.path.dirname(__file__), "./images/2388455-scaled.jpg"),
                  os.path.join(os.path.dirname(__file__), "./images/1.jpg"),
                  os.path.join(os.path.dirname(__file__), "./images/2.jpg"),
                  os.path.join(os.path.dirname(__file__), "./images/3.jpg"),
                  os.path.join(os.path.dirname(__file__), "./images/4.jpg"),
                  os.path.join(os.path.dirname(__file__), "./images/5.jpg"),
                  os.path.join(os.path.dirname(__file__), "./images/6.jpg"),
                  os.path.join(os.path.dirname(__file__), "./images/7.jpg"),
                  os.path.join(os.path.dirname(__file__), "./images/8.jpg"),
                  ],
        inputs=input_image,
        outputs=image_output,
    )


    # 按钮交互
    button.click(inference, inputs=[device, model_type, input_image, points_per_side, pred_iou_thresh,
                                stability_score_thresh, min_mask_region_area, stability_score_offset, box_nms_thresh,
                                crop_n_layers, crop_nms_thresh],
             outputs=image_output)



demo.launch(debug=True)



