from pathlib import Path
import PIL
import streamlit as st
import settings
import helper
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

# =============================
# ⚙️ Cấu hình giao diện
# =============================
st.set_page_config(
    page_title="Phân Loại Rác Thải bằng YOLO",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌍 Phân loại rác thải bằng YOLO")

# =============================
# 📌 Thanh bên
# =============================
st.sidebar.header("⚙️ Cấu hình ứng dụng")
page = st.sidebar.radio("📑 Chọn chức năng", ["Phát hiện"])

if page == "Phát hiện":
    st.sidebar.subheader("🧠 Cấu hình mô hình học máy")
    model_type = st.sidebar.radio("🔍 Chọn chế độ", ['Phát hiện'])
    confidence = float(st.sidebar.slider("📊 Chọn độ tin cậy (%)", 15, 100, 25))/100
    imgsz = st.sidebar.slider("🖼️ Kích thước ảnh (px)", 320, 1280, 640, step=32)

    model_path = Path(settings.DETECTION_MODEL)
    
    # Kiểm tra xem file có tồn tại không
    if not model_path.exists():
        st.error(f"❌ File mô hình không tồn tại: {model_path}")
        st.error(f"📍 Đường dẫn tuyệt đối: {model_path.absolute()}")
        st.info("💡 Hãy kiểm tra lại đường dẫn trong settings.py")
        st.stop()
    
    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        st.error(f"❌ Không thể tải mô hình: {model_path}")
        st.error(str(ex))
        st.stop()

    st.sidebar.subheader("📸 Chọn nguồn ảnh/Video")
    source_radio = st.sidebar.radio("🖼️ Nguồn", settings.SOURCES_LIST)
    source_img = None

    # --- Hàm hiển thị kết quả ảnh tĩnh ---
    def display_results(res):
        boxes = res[0].boxes
        data = []
        class_count = {}
        for box in boxes:
            cls_id = int(box.cls)
            cls_name = model.names[cls_id]
            conf_score = float(box.conf)
            coords = box.data.tolist()
            data.append([cls_name, conf_score, coords])
            class_count[cls_name] = class_count.get(cls_name, 0)+1
        df = pd.DataFrame(data, columns=["Class","Confidence","Box"])
        return df, class_count

    def plot_class_counts(class_count):
        if class_count:
            fig, ax = plt.subplots()
            ax.bar(class_count.keys(), class_count.values(), color='skyblue')
            ax.set_xlabel("Class")
            ax.set_ylabel("Số lượng")
            ax.set_title("Số lượng từng class")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # --- Ảnh tĩnh ---
    if source_radio == settings.IMAGE:
        source_img = st.sidebar.file_uploader("📂 Chọn ảnh...", type=("jpg","jpeg","png","bmp","webp"))
        col1, col2 = st.columns(2)
        with col1:
            if source_img is None:
                st.image(str(settings.DEFAULT_IMAGE), caption="📎 Ảnh Mặc Định", use_container_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(uploaded_image, caption="📎 Ảnh Đã Tải Lên", use_container_width=True)

        with col2:
            if source_img and st.sidebar.button("🚀 Phát hiện đối tượng"):
                res = model.predict(uploaded_image, conf=confidence, imgsz=imgsz, iou=0.45)
                frame_plot = res[0].plot()[:, :, ::-1]
                st.image(frame_plot, caption="📍 Ảnh sau phát hiện", use_container_width=True)

                df, class_count = display_results(res)
                with st.expander("📋 Kết quả phát hiện"):
                    st.dataframe(df)
                st.subheader("📊 Biểu đồ số lượng từng class")
                plot_class_counts(class_count)

    # --- Webcam ---
    elif source_radio == settings.WEBCAM:
        helper.play_webcam(confidence, model, imgsz=imgsz)

    else:
        st.error("⚠️ Vui lòng chọn loại nguồn hợp lệ!")
