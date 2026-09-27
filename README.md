from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

print("Loading AI Image Captioning Model...")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# Enter image path
image_path = input("Enter image path: ")

try:
    image = Image.open(image_path).convert("RGB")

    print("\nGenerating caption...")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=50
        )

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    print("\nIMAGE CAPTION:")
    print(caption)

except Exception as e:
    print("Error:", e)
