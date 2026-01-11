import streamlit as st
import os
from PIL import Image
import numpy as np

st.set_page_config(page_title="EcoTogether", page_icon="♻️")

st.title("♻️ EcoTogether – AI phân loại rác")
st.caption("Upload ảnh → AI nhận diện → tích điểm → đổi quà")
st.divider()

# ================== CHECK FILE ==================
model_ok = True

if not os.path.exists("keras_model.h5"):
    st.error("❌ Thiếu file keras_model.h5")
    model_ok = False

if not os.path.exists("labels.txt"):
    st.error("❌ Thiếu file labels.txt")
    model_ok = False

# ================== LOAD MODEL ==================
if model_ok:
    try:
        from tensorflow import keras
        model = keras.models.load_model("keras_model.h5", compile=False)
        with open("labels.txt", "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f.readlines()]
    except Exception as e:
        st.error("❌ Lỗi load AI model")
        st.code(str(e))
        model_ok = False

# ================== SESSION ==================
if "total_points" not in st.session_state:
    st.session_state.total_points = 0

# ================== UPLOAD / CAMERA ==================
image_file = st.camera_input("📷 Chụp ảnh rác")
if image_file is None:
    image_file = st.file_uploader(
        "Hoặc tải ảnh rác",
        type=["jpg", "png", "jpeg"]
    )

# ================== XỬ LÝ ==================
if image_file:
    image = Image.open(image_file).convert("RGB")
    st.image(image, use_column_width=True)

    if model_ok:
        img = image.resize((224, 224))
        arr = np.asarray(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        with st.spinner("🤖 AI đang nhận diện..."):
            pred = model.predict(arr)

        idx = np.argmax(pred)
        trash = labels[idx]
        conf = pred[0][idx] * 100

        st.success(f"🧠 AI nhận diện: **{trash}** ({conf:.1f}%)")

        weight = st.slider("⚖️ Trọng lượng (gram)", 0, 500, 50, 10)
        points = weight / 10

        if st.button("✅ Xác nhận bỏ rác"):
            st.session_state.total_points += points
            st.success(f"🎉 +{points:.1f} điểm")

    else:
        st.warning("⚠️ AI chưa sẵn sàng – chỉ demo giao diện")

# ================== ĐIỂM ==================
st.divider()
st.subheader("⭐ Tổng điểm")
st.write(f"🎯 {st.session_state.total_points:.1f}")

# ================== QUÀ ==================
st.subheader("🎁 Đổi quà")
if st.session_state.total_points >= 100:
    st.success("🌱 Cây xanh mini")
elif st.session_state.total_points >= 50:
    st.info("👜 Túi vải môi trường")
else:
    st.warning("❌ Chưa đủ điểm")


