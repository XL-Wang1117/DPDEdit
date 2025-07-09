# 🎉 DPDEdit: Detail-Preserved Diffusion Models for Multimodal Fashion Image Editing

<p align="center">
  <img src="assets/banner_poster.png" width="80%"><br>
  <em>Accepted as Poster at IEEE ICME 2025, Nantes, France 🇫🇷</em>
</p>

---

## ✨ Overview

**DPDEdit** is a novel diffusion-based framework designed for precise and controllable fashion image editing using multimodal inputs — including text, garment texture images, region masks, and human pose.  
By integrating **Grounded-SAM** for accurate region localization and a **Detail-Preserving U-Net (DP-UNet)** for fine texture restoration, our method enables photorealistic garment editing with high fidelity and controllability.

> 🔧 Paper Title: **DPDEdit: Detail-Preserved Diffusion Models for Multimodal Fashion Image Editing**  
> 📍 Conference: IEEE ICME 2025  
> 🧠 Authors: Xiaolong Wang\*, Zhigi Cheng, Jue Wang, Huizi Xue, Xiaojiang Peng\*  
> 🏫 Affiliations: Shenzhen Technology University, University of Washington, Chinese Academy of Sciences

---

## 📷 Results

<p align="center">
  <img src="assets/showcase_results.png" width="90%">
</p>

(*You may replace this image with your own showcase results.*)

---


---

## ⚙️ Requirements

```bash
git clone https://github.com/XL-Wang1117/DPDEdit.git
cd DPDEdit

conda create -n dpdedit python=3.10
conda activate dpdedit

pip install -r requirements.txt
📦 Dataset
We evaluate and train DPDEdit on the DPDEdit-Extension dataset — an extension of VITON-HD with annotated texture-image pairs and garment descriptions.

You can download the dataset from:
👉 [🔗 Insert your dataset download link here]

Expected folder structure:

DPDEdit-Extension/
├── train/
│   ├── image/
│   ├── image-densepose/
│   ├── agnostic-mask/
│   ├── cloth/
│   └── vitonhd_train_tagged.json
├── test/
│   ├── image/
│   ├── image-densepose/
│   ├── agnostic-mask/
│   ├── cloth/
│   └── vitonhd_test_tagged.json
🔮 Inference
Run inference on the test set:

accelerate launch Inference.py \
    --width 768 \
    --height 1024 \
    --num_inference_steps 30 \
    --output_dir "results" \
    --unpaired \
    --data_dir "PATH_TO_DPDEdit-Extension/test" \
    --seed 42 \
    --test_batch_size 2 \
    --guidance_scale 5.0
Replace PATH_TO_DPDEdit-Extension with your local dataset path.

🏋️ Training (optional)
To train the model from scratch or fine-tune:


accelerate launch train.py \
    --output_dir "checkpoints" \
    --train_batch_size 6 \
    --data_dir "PATH_TO_DPDEdit-Extension/train"
You can modify training configs via parser_args.py or configs/.


  title={DPDEdit: Detail-Preserved Diffusion Models for Multimodal Fashion Image Editing},
  author={Xiaolong Wang, Zhigi Cheng, Jue Wang, Huizi Xue, Xiaojiang Peng},
  booktitle={IEEE International Conference on Multimedia and Expo (ICME)},
  year={2025}
}
