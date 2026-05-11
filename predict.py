import torch
import timm
from torchvision import transforms
from PIL import Image

# CLASS NAMES
class_names = [
    'Siyah-tan',
    'akbas',
    'beagle',
    'bulldog',
    'cocker',
    'dalmatian',
    'german-shepherd',
    'husky',
    'kangal',
    'karadeniz dag',
    'labrador-retriever',
    'malakli',
    'poodle',
    'rottweiler'
]

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using:", device)

# MODEL
model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=len(class_names)
)

model.load_state_dict(torch.load("breed_model.pth"))

model.to(device)

model.eval()

# TRANSFORM
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# IMAGE
image = Image.open("cropped_dog.jpg").convert("RGB")

image = transform(image).unsqueeze(0).to(device)

# PREDICTION
with torch.no_grad():

    outputs = model(image)

    probabilities = torch.softmax(outputs[0], dim=0)

    top3_prob, top3_catid = torch.topk(probabilities, 3)

# RESULTS
print("\nTop 3 Predictions:\n")

for i in range(3):

    breed = class_names[top3_catid[i]]
    prob = top3_prob[i].item() * 100

    print(f"{i+1}. {breed} --> %{prob:.2f}")