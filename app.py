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


def segment_one(img, mask_generator, seed=None):
    if seed is not None:
        np.random.seed(seed)
    masks = mask_generator.generate(img)
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    mask_all = np.ones((img.shape[0], img.shape[1], 3))
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            mask_all[m == True, i] = color_mask[i]
    result = img / 255 * 0.3 + mask_all * 0.7
    return result, mask_all


def inference(device, model_type, points_per_side, pred_iou_thresh, stability_score_thresh, min_mask_region_area,
                  stability_score_offset, box_nms_thresh, crop_n_layers, crop_nms_thresh, input_x, progress=gr.Progress()):
    # sam model
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

    # input is image, type: numpy
    if type(input_x) == np.ndarray:
        result, mask_all = segment_one(input_x, mask_generator)
        return result, mask_all
    elif isinstance(input_x, str):  # input is video, type: path (str)
        cap = cv2.VideoCapture(input_x)     # read video
        frames_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc('x', '2', '6', '4'), fps, (W, H), isColor=True)
        for _ in progress.tqdm(range(int(frames_num)), desc='Processing video ({} frames, size {}x{})'.format(int(frames_num), W, H)):
            ret, frame = cap.read()     # read a frame
            result, mask_all = segment_one(frame, mask_generator, seed=2023)
            result = (result * 255).astype(np.uint8)
            out.write(result)
        out.release()
        cap.release()
        return 'output.mp4'


with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown(
            '''# Segment Anything!🚀
            The Segment Anything Model (SAM) produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image. It has been trained on a dataset of 11 million images and 1.1 billion masks, and has strong zero-shot performance on a variety of segmentation tasks.
            [**Official Project**](https://segment-anything.com/)
            '''
        )
        with gr.Row():
            # select model
            model_type = gr.Dropdown(["vit_b", "vit_l", "vit_h"], value='vit_b', label="Select Model")
            # select device
            device = gr.Dropdown(["cpu", "cuda"], value='cuda', label="Select Device")

    # 参数
    with gr.Accordion(label='Parameters', open=False):
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

    # Show image
    with gr.Tab(label='Image'):
        with gr.Row().style(equal_height=True):
            with gr.Column():
                input_image = gr.Image(type="numpy")
                with gr.Row():
                    button = gr.Button("Auto!")
            with gr.Tab(label='Image+Mask'):
                output_image = gr.Image(type='numpy')
            with gr.Tab(label='Mask'):
                output_mask = gr.Image(type='numpy')

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
            outputs=output_image,
        )
    # Show video
    with gr.Tab(label='Video'):
        with gr.Row().style(equal_height=True):
            with gr.Column():
                input_video = gr.Video()
                with gr.Row():
                    button_video = gr.Button("Auto!")
            output_video = gr.Video(format='mp4')
        gr.Markdown('''
        **Note:** processing video will take a long time, please upload a short video.
        ''')
        gr.Examples(
            examples=[os.path.join(os.path.dirname(__file__), "./images/video1.mp4"),
                      os.path.join(os.path.dirname(__file__), "./images/video2.mp4")
                      ],
            inputs=input_video,
            outputs=output_video
        )

    # button image
    button.click(inference, inputs=[device, model_type, points_per_side, pred_iou_thresh, stability_score_thresh,
                                    min_mask_region_area, stability_score_offset, box_nms_thresh, crop_n_layers,
                                    crop_nms_thresh, input_image],
                 outputs=[output_image, output_mask])
    # button video
    button_video.click(inference, inputs=[device, model_type, points_per_side, pred_iou_thresh, stability_score_thresh,
                                    min_mask_region_area, stability_score_offset, box_nms_thresh, crop_n_layers,
                                    crop_nms_thresh, input_video],
                       outputs=[output_video])


demo.queue().launch(debug=True, enable_queue=True)



