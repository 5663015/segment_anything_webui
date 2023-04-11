import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from transformers import OwlViTProcessor, OwlViTForObjectDetection


models = {
    'vit_b': './checkpoints/sam_vit_b_01ec64.pth',
    'vit_l': './checkpoints/sam_vit_l_0b3195.pth',
    'vit_h': './checkpoints/sam_vit_h_4b8939.pth'
}


def plot_boxes(img, boxes):
	img_pil = Image.fromarray(np.uint8(img * 255)).convert('RGB')
	draw = ImageDraw.Draw(img_pil)
	for box in boxes:
		color = tuple(np.random.randint(0, 255, size=3).tolist())
		x0, y0, x1, y1 = box
		x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
		draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
	# img_pil = img_pil.point(lambda x: x / 255)
	return img_pil


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


def generator_inference(device, model_type, points_per_side, pred_iou_thresh, stability_score_thresh, min_mask_region_area,
                  stability_score_offset, box_nms_thresh, crop_n_layers, crop_nms_thresh, input_x, input_text, progress=gr.Progress()):
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


def predictor_inference(device, model_type, input_x, input_text):
	# sam model
	sam = sam_model_registry[model_type](checkpoint=models[model_type]).to(device)
	predictor = SamPredictor(sam)
	predictor.set_image(input_x)  # Process the image to produce an image embedding

	# get box using OWL-ViT
	processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
	owlvit_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
	input_text = processor(text=[input_text], images=input_x, return_tensors="pt")
	outputs = owlvit_model(**input_text)
	target_size = torch.Tensor([input_x.shape[1:]]).to(device)
	results = processor.post_process(outputs=outputs, target_sizes=target_size)
	# get the box with best score
	scores = torch.sigmoid(outputs.logits)
	best_scores, best_idxs = torch.topk(scores, k=1, dim=1)
	best_idxs = best_idxs.squeeze(1).tolist()

	i = 0
	# boxes, scores, labels = results[i]["boxes"][best_idxs], results[i]["scores"][best_idxs], results[i]["labels"][best_idxs]
	boxes = results[i]["boxes"][best_idxs]
	boxes = boxes.cpu().detach().numpy()
	transformed_boxes = predictor.transform.apply_boxes_torch(torch.Tensor(boxes).to(device), input_x.shape[:2])
	# transformed_boxes = np.squeeze(transformed_boxes.cpu().detach().numpy())
	# transformed_boxes = torch.Tensor(transformed_boxes).to(device)
	print(transformed_boxes)

	masks, scores, logits = predictor.predict_torch(
		point_coords=None,
		point_labels=None,
		boxes=transformed_boxes,  # only one box
		multimask_output=False,
	)
	print(masks.size())
	mask_all = np.ones((input_x.shape[0], input_x.shape[1], 3))
	for ann in masks:
		color_mask = np.random.random((1, 3)).tolist()[0]
		for i in range(3):
			mask_all[ann[0] == True, i] = color_mask[i]
	img = input_x / 255 * 0.3 + mask_all * 0.7
	img = plot_boxes(img, transformed_boxes)
	return img, mask_all

def run_inference(device, model_type, points_per_side, pred_iou_thresh, stability_score_thresh, min_mask_region_area,
                  stability_score_offset, box_nms_thresh, crop_n_layers, crop_nms_thresh, input_x, input_text):
	print('prompt text: ', input_text)
	if input_text != '':    # user input text
		return predictor_inference(device, model_type, input_x, input_text)
	else:
		return generator_inference(device, model_type, points_per_side, pred_iou_thresh, stability_score_thresh, min_mask_region_area,
                  stability_score_offset, box_nms_thresh, crop_n_layers, crop_nms_thresh, input_x, input_text)

