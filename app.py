import streamlit as st
from PIL import Image
import torch
import timm
from torchvision import transforms
from ultralytics import YOLO
import cv2
import os
import random

from breed_info import breed_data

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Stray Dog Breed Classification",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""

""", unsafe_allow_html=True)

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1 {
    text-align: center;
    color: #ff4b4b;
    font-size: 50px !important;
}

h2, h3 {
    color: #222222;
}

.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 10px 25px;
    font-size: 18px;
}

.stFileUploader {
    background-color: transparent;
    border: none;
    padding: 0;
}

img {
    border-radius: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# CLASS NAMES
# =========================
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

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD MODEL
# =========================
classifier = timm.create_model(
    "efficientnet_b3",
    pretrained=False,
    num_classes=len(class_names)
)

classifier.load_state_dict(
    torch.load("breed_model.pth", map_location=device)
)

classifier.to(device)
classifier.eval()

# =========================
# LOAD YOLO
# =========================
yolo_model = YOLO("yolov8n.pt")

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================
# TITLE
# =========================
st.title("🐶 Stray Dog Breed Classification")

st.markdown("""
<div style='text-align:center; font-size:20px;'>

Upload a dog image and the system will:

✅ Detect the dog using YOLO  
✅ Predict the breed  
✅ Analyze possible hybrid structure  
✅ Show similar dataset examples  
✅ Display breed information  

</div>
""", unsafe_allow_html=True)

st.write("")

# =========================
# FILE UPLOADER
# =========================
uploaded_file = st.file_uploader(
    "📤 Upload Dog Image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# MAIN
# =========================
if uploaded_file:

    # LOAD IMAGE
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("📸 Uploaded Image")

    # CENTER IMAGE
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        st.image(
            image,
            caption="Uploaded Dog Image",
            width=500
        )

    # SAVE TEMP IMAGE
    image.save("temp.jpg")

    # YOLO DETECTION
    results = yolo_model("temp.jpg")

    # READ IMAGE
    img_cv = cv2.imread("temp.jpg")

    DOG_CLASS_ID = 16
    dog_found = False

    # DETECT DOG
    for result in results:

        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls[0])

            if cls == DOG_CLASS_ID:

                dog_found = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cropped = img_cv[y1:y2, x1:x2]

                cv2.imwrite("cropped.jpg", cropped)

    # =========================
    # DOG FOUND
    # =========================
    if dog_found:

        st.success("✅ Dog detected successfully!")

        crop_img = Image.open("cropped.jpg").convert("RGB")

        st.subheader("🐕 Detected Dog")

        # CENTER DETECTED IMAGE
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:

            st.image(
                crop_img,
                caption="Detected Dog",
                width=500
            )

        # MODEL INPUT
        input_tensor = transform(crop_img).unsqueeze(0).to(device)

        # PREDICTION
        with torch.no_grad():

            outputs = classifier(input_tensor)

            probs = torch.softmax(outputs[0], dim=0)

            top3_prob, top3_catid = torch.topk(probs, 3)

        # TOP BREEDS
        first_breed = class_names[top3_catid[0]]
        first_prob = top3_prob[0].item() * 100

        second_breed = class_names[top3_catid[1]]
        second_prob = top3_prob[1].item() * 100

        # =========================
        # RESULTS
        # =========================
        st.subheader("🏆 Breed Prediction")

        st.write(f"### 🐶 {first_breed} → %{first_prob:.2f}")

        st.progress(min(int(first_prob), 100))

        # HYBRID ONLY IF SECOND > 1%
        show_hybrid = second_prob > 1

        if show_hybrid:

            st.write(f"### 🐶 {second_breed} → %{second_prob:.2f}")

            st.progress(min(int(second_prob), 100))

            st.subheader("🧬 Hybrid Breed Analysis")

            st.info(
                f"""
                This stray dog may be a mix of:

                🐕 {first_breed}
                +
                🐕 {second_breed}
                """
            )

        # =========================
        # DATASET EXAMPLES
        # =========================
        st.subheader("📸 Similar Dogs From Dataset")

        try:

            # FIRST BREED IMAGE
            first_folder = os.path.join("dataset", first_breed)

            first_images = os.listdir(first_folder)

            first_random = random.choice(first_images)

            first_path = os.path.join(first_folder, first_random)

            img1 = Image.open(first_path).resize((350, 350))

            # IF HYBRID
            if show_hybrid:

                second_folder = os.path.join("dataset", second_breed)

                second_images = os.listdir(second_folder)

                second_random = random.choice(second_images)

                second_path = os.path.join(second_folder, second_random)

                img2 = Image.open(second_path).resize((350, 350))

                col1, col2 = st.columns(2)

                with col1:

                    st.image(
                        img1,
                        caption=f"{first_breed} Example"
                    )

                with col2:

                    st.image(
                        img2,
                        caption=f"{second_breed} Example"
                    )

            else:

                col1, col2, col3 = st.columns([1,2,1])

                with col2:

                    st.image(
                        img1,
                        caption=f"{first_breed} Example"
                    )

        except Exception as e:

            st.warning(f"Dataset images could not be loaded: {e}")

        # =========================
        # BREED INFO
        # =========================
        st.subheader("📚 Breed Information")

        if first_breed in breed_data:

            info = breed_data[first_breed]

            st.markdown(f"## 🐶 {first_breed}")

            st.write(f"**Lifespan:** {info['lifespan']}")
            st.write(f"**Personality:** {info['personality']}")
            st.write(f"**Aggressiveness:** {info['aggressiveness']}")
            st.write(f"**Feeding:** {info['feeding']}")
            st.write(f"**Exercise Needs:** {info['exercise']}")

    else:

        st.error("❌ No dog detected in the image.")