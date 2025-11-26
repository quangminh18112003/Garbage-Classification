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
    confidence = float(st.sidebar.slider("📊 Chọn độ tin cậy (%)", 10, 100, 20))/100
    imgsz = st.sidebar.slider("🖼️ Kích thước ảnh (px)", 320, 1280, 800, step=32)
    max_det = st.sidebar.slider("🔢 Số lượng phát hiện tối đa", 10, 300, 100, step=10)

    model_path = Path(settings.DETECTION_MODEL)
    
    # Kiểm tra xem file có tồn tại không
    if not model_path.exists():
        st.error(f"❌ File mô hình không tồn tại: {model_path}")
        st.error(f"📍 Đường dẫn tuyệt đối: {model_path.absolute()}")
        st.info("💡 Hãy kiểm tra lại đường dẫn trong settings.py")
        st.stop()
    
    try:
        with st.spinner("🔄 Đang tải mô hình..."):
            model = helper.load_model(model_path)
        st.sidebar.success(f"✅ Model đã tải: {model_path.name}")
        # Hiển thị thông tin model
        with st.sidebar.expander("ℹ️ Thông tin Model"):
            st.write(f"**Đường dẫn:** {model_path}")
            st.write(f"**Classes:** {len(model.names)}")
            st.write(f"**Tên classes:** {', '.join(model.names.values())}")
    except Exception as ex:
        st.error(f"❌ Không thể tải mô hình: {model_path}")
        st.error(str(ex))
        st.info("💡 Kiểm tra xem file best.pt có tồn tại trong GiaoDien/weights/ không")
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
                with st.spinner("🔄 Đang xử lý ảnh..."):
                    try:
                        # Thử với nhiều cấu hình khác nhau để phát hiện tốt hơn
                        res = model.predict(
                            uploaded_image, 
                            conf=confidence, 
                            imgsz=imgsz, 
                            iou=0.45,
                            max_det=max_det,
                            agnostic_nms=False,
                            verbose=False
                        )
                        frame_plot = res[0].plot()[:, :, ::-1]
                        st.image(frame_plot, caption="📍 Ảnh sau phát hiện", use_container_width=True)
                        
                        # Hiển thị thông tin debug
                        num_detections = len(res[0].boxes)
                        if num_detections == 0:
                            st.warning(f"⚠️ Không phát hiện được đối tượng nào với cấu hình hiện tại!")
                            st.info(f"💡 Thử giảm confidence xuống 10-15% hoặc tăng image size lên 1024px")
                            st.info(f"📊 Cấu hình hiện tại: Confidence={confidence*100:.0f}%, Image Size={imgsz}px, Max Detections={max_det}")
                        else:
                            st.success(f"✅ Phát hiện được {num_detections} đối tượng!")
                            
                            df, class_count = display_results(res)
                            with st.expander("📋 Kết quả phát hiện"):
                                st.dataframe(df)
                            st.subheader("📊 Biểu đồ số lượng từng class")
                            plot_class_counts(class_count)
                    except Exception as e:
                        st.error(f"❌ Lỗi khi phát hiện: {str(e)}")
                        st.exception(e)

    # --- Webcam ---
    elif source_radio == settings.WEBCAM:
        helper.play_webcam(confidence, model, imgsz=imgsz)

    else:
        st.error("⚠️ Vui lòng chọn loại nguồn hợp lệ!")
