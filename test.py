import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
print(dtype)

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
checkpoint_path = "./model.pt"
state_dict = torch.load(checkpoint_path)
model = VGGT()
model.load_state_dict(state_dict)
model = model.to(dtype=dtype)
model.eval()
model = model.to(device)

# Load and preprocess example images (replace with your own image paths)
image_names = ["/home/geneta/project/vggt/examples/single_cartoon/images/model_was_never_trained_on_single_image_or_cartoon.jpg"]  
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.

        predictions = model(images)

        # for key in predictions.keys():
        #     if isinstance(predictions[key], torch.Tensor):
        #         print(f"Key: {key}, Shape: {predictions[key].shape}")