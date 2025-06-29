from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np

np.random.seed(42)

# Create dataset/receipts folder one level above script
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
dataset_dir = os.path.join(parent_dir, "dataset", "receipts")
os.makedirs(dataset_dir, exist_ok=True)

merchant_names = [
    "Cafe Luna", "Quick Mart", "Book Haven", "Tech Store",
    "Green Grocery", "Urban Outfit", "Fresh Bites", "Gear Hub"
]

# Load Arial Black 
try:
    font = ImageFont.truetype("arialbd.ttf", 32)  # Arial Black bold
except IOError:
    print("Arial Black not found, using default PIL font ")
    font = ImageFont.load_default()
## 100 images
for i in range(100):
    img = Image.new('RGB', (1500, 700), color='white')  
    draw = ImageDraw.Draw(img)
    
    merchant = np.random.choice(merchant_names)
    total = round(np.random.uniform(1, 500), 2)
    
    #  Bold text
    draw.text((100, 200), merchant, fill='black', font=font)
    draw.text((100, 400), f"TOTAL: ${total}", fill='black', font=font)

    #Rotation to mimic camera
    angle = np.random.uniform(-1, 1)
    img = img.rotate(angle, expand=True, fillcolor='white')
    
    img_path = os.path.join(dataset_dir, f"receipt_{i:03d}.jpg")
    img.save(img_path)

print("Synthetic Data of Receipts Generated!!")
