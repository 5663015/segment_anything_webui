# Segment Anything WebUI

This project is based on **[Segment Anything Model](https://segment-anything.com/) ** by Meta. The UI is based on [Gradio](https://gradio.app/). 

![](./images/20230408023615.png)

## **Usage**

Following usage is running on your computer. 

- Install Segment Anything（[more details about install Segment Anything](https://github.com/facebookresearch/segment-anything#installation)）：

```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

- `git clone` this repository：

```
git clone https://github.com/5663015/segment_anything_webui.git
```

- Make a new folder named `checkpoints` under this project，and put the downloaded weights files in `checkpoints`。You can download the weights using following URLs：

  - `vit_h`: [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

  - `vit_l`: [ViT-L SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)

  - `vit_b`: [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

- Run：

```
python app.py
```

**Note：** Default model is `vit_b`，the demo can run on CPU. Default device is `cuda`。

## TODO

- Add segmentation prompt 