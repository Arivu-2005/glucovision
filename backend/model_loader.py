import torch
import torch.nn as nn
from torchvision.models import vit_b_16
from torchvision import transforms
from PIL import Image

# -----------------------------------------
# Build model EXACTLY like trained version
# -----------------------------------------
def build_model():
    model = vit_b_16(weights=None)

    # Replace default head with your custom head
    model.heads = nn.Sequential(
        nn.Linear(768, 512),   # MATCH checkpoint
        nn.ReLU(),
        nn.Linear(512, 1)      # MATCH checkpoint
    )

    return model


# -----------------------------------------
# Load model from checkpoint
# -----------------------------------------
def load_glaucoma_model(model_path):
    model = build_model()
    state_dict = torch.load(model_path, map_location="cpu")

    # Load state dict EXACTLY
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    return model


# -----------------------------------------
# Transform
# -----------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# -----------------------------------------
# Prediction
# -----------------------------------------
def predict_image(model, image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)

    # output is a single value → sigmoid → 0/1
    prob = torch.sigmoid(output).item()

    predicted = 1 if prob >= 0.5 else 0

    return predicted, prob
