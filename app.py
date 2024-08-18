import gradio as gr
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry
from torch.nn import functional as F
from skimage import transform
from PIL import Image, ImageDraw
import io
import os
import gdown

# Download MedSAM model if not exists
model_path = "medsam_vit_b.pth"
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1bxqHb4zC3D3Sdc7x1gxX14w7QjPa6bEe"
    gdown.download(url, model_path, quiet=False)

# Load MedSAM model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_type = "vit_b"
sam_checkpoint = model_path
model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
model.to(device=device)
model.eval()

def preprocess(img):
    img = transform.resize(img, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True)
    img = (img - img.min()) / np.clip(img.max() - img.min(), a_min=1e-8, a_max=None)
    img = img.astype(np.float32)
    img = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    return torch.from_numpy(img)

@torch.no_grad()
def process_image(image, bbox):
    # Preprocess image
    preprocessed_image = preprocess(image).to(device)
    
    # Get image embeddings
    image_embedding = model.image_encoder(preprocessed_image)
    
    # Prepare bbox
    H, W = image.shape[:2]
    box_1024 = np.array(bbox) / np.array([W, H, W, H]) * 1024
    box_torch = torch.from_numpy(box_1024).float().unsqueeze(0).to(device)

    # Get mask prediction
    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )
    pred_mask = (low_res_pred > 0.5).squeeze().cpu().numpy()
    
    return pred_mask

def segment_image(input_image, evt: gr.SelectData):
    if evt.index is None or len(evt.index) != 4:
        return input_image, None, "Please draw a bounding box."
    
    x1, y1, x2, y2 = evt.index
    bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    # Convert PIL Image to numpy array
    image_np = np.array(input_image)

    # Process the image
    pred_mask = process_image(image_np, bbox)

    # Create segmentation overlay
    overlay = image_np.copy()
    overlay[pred_mask] = (0, 255, 0)  # Green color for the segmentation
    result_image = cv2.addWeighted(image_np, 0.7, overlay, 0.3, 0)

    # Convert numpy array back to PIL Image
    result_pil = Image.fromarray(result_image)

    # Draw bounding box on the input image
    draw = ImageDraw.Draw(input_image)
    draw.rectangle(bbox, outline="red", width=2)

    return input_image, result_pil, f"Segmentation completed. Bounding box: {bbox}"

def evaluate_segmentation(input_image, result_image, gt_mask):
    if result_image is None:
        return "Please perform segmentation first."

    # Convert PIL Images to numpy arrays
    result_np = np.array(result_image)
    gt_mask_np = np.array(gt_mask.convert('L'))

    # Extract the green channel (where segmentation is marked) and binarize
    pred_mask = (result_np[:,:,1] > 200).astype(np.uint8)
    gt_mask_bin = (gt_mask_np > 128).astype(np.uint8)

    # Resize ground truth mask if necessary
    if gt_mask_bin.shape != pred_mask.shape:
        gt_mask_bin = cv2.resize(gt_mask_bin, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Calculate IoU and Dice score
    intersection = np.logical_and(pred_mask, gt_mask_bin)
    union = np.logical_or(pred_mask, gt_mask_bin)
    iou = np.sum(intersection) / np.sum(union)
    dice = 2. * np.sum(intersection) / (np.sum(pred_mask) + np.sum(gt_mask_bin))

    return f"IoU: {iou:.4f}, Dice Score: {dice:.4f}"

# Define Gradio interface
with gr.Blocks() as webapp:
    gr.Markdown("# MedSAM Segmentation Web App")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", tool="bbox", interactive=True)
            segment_button = gr.Button("Segment")
        with gr.Column():
            result_image = gr.Image(label="Segmentation Result")
    
    status_text = gr.Textbox(label="Status")
    
    with gr.Row():
        gt_mask = gr.Image(label="Ground Truth Mask")
        evaluate_button = gr.Button("Evaluate")
    
    eval_result = gr.Textbox(label="Evaluation Result")

    segment_button.click(
        segment_image, 
        inputs=[input_image],
        outputs=[input_image, result_image, status_text]
    )

    evaluate_button.click(
        evaluate_segmentation,
        inputs=[input_image, result_image, gt_mask],
        outputs=[eval_result]
    )

# Launch the web app
webapp.launch()
