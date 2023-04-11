import os
import gradio as gr
from inference import run_inference




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
            device = gr.Dropdown(["cpu", "cuda"], value='cpu', label="Select Device")

    # parameters
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
                text = gr.Textbox(label='Text prompt(optional)', info=
                    'If you input a word, the OWL-ViT model will be run to detect the object in image, '
                    'and the box will be fed into SAM model to predict mask.')
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
    button.click(run_inference, inputs=[device, model_type, points_per_side, pred_iou_thresh, stability_score_thresh,
                                    min_mask_region_area, stability_score_offset, box_nms_thresh, crop_n_layers,
                                    crop_nms_thresh, input_image, text],
                 outputs=[output_image, output_mask])
    # button video
    button_video.click(run_inference, inputs=[device, model_type, points_per_side, pred_iou_thresh, stability_score_thresh,
                                    min_mask_region_area, stability_score_offset, box_nms_thresh, crop_n_layers,
                                    crop_nms_thresh, input_video, text],
                       outputs=[output_video])


demo.queue().launch(debug=True, enable_queue=True)



