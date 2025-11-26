from ultralytics import YOLO
import streamlit as st
import cv2
import settings

def load_model(model_path):
    """Tải mô hình YOLO."""
    model = YOLO(str(model_path))
    return model

def display_tracker_options():
    display_tracker = st.radio("Hiển thị trình theo dõi", ('Có', 'Không'))
    is_display_tracker = display_tracker == 'Có'
    tracker_type = None
    if is_display_tracker:
        tracker_type = st.radio("Trình theo dõi", ("bytetrack.yaml", "botsort.yaml"))
    return is_display_tracker, tracker_type

def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None, imgsz=640):
    """Hiển thị frame detect + trả về boxes."""
    image = cv2.resize(image, (720, int(720*(9/16))))
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker, imgsz=imgsz, iou=0.45)
    else:
        res = model.predict(image, conf=conf, imgsz=imgsz, iou=0.45)

    frame_plot = res[0].plot()
    st_frame.image(frame_plot, caption="Video đã phát hiện", channels="BGR", use_container_width=True)
    return res[0].boxes

def play_webcam(conf, model, imgsz=640):
    """Webcam Start/Stop với bảng + biểu đồ realtime."""
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    st_frame = st.empty()
    data_placeholder = st.expander("📋 Kết quả phát hiện webcam", expanded=True)
    plot_placeholder = st.empty()

    # --- Nút Start/Stop ---
    if 'webcam_running' not in st.session_state:
        st.session_state.webcam_running = False

    if st.sidebar.button("Start Webcam") and not st.session_state.webcam_running:
        st.session_state.webcam_running = True
    if st.sidebar.button("Stop Webcam") and st.session_state.webcam_running:
        st.session_state.webcam_running = False

    # --- Chạy webcam nếu đang bật ---
    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(source_webcam)
        if not cap.isOpened():
            st.error("⚠️ Không mở được webcam!")
            st.session_state.webcam_running = False
            return

        while st.session_state.webcam_running:
            success, frame = cap.read()
            if not success:
                st.warning("⚠️ Không nhận được frame từ webcam")
                break

            boxes = _display_detected_frames(conf, model, st_frame, frame, is_display_tracker, tracker, imgsz=imgsz)

            # --- Bảng + biểu đồ ---
            data = []
            class_count = {}
            for box in boxes:
                cls_id = int(box.cls)
                cls_name = model.names[cls_id]
                conf_score = float(box.conf)
                coords = box.data.tolist()
                data.append([cls_name, conf_score, coords])
                class_count[cls_name] = class_count.get(cls_name, 0) + 1

            data_placeholder.empty()
            with data_placeholder:
                if data:
                    import pandas as pd
                    df = pd.DataFrame(data, columns=["Class", "Confidence", "Box"])
                    st.dataframe(df)
                else:
                    st.write("⚠️ Không phát hiện đối tượng nào!")

            plot_placeholder.empty()
            with plot_placeholder:
                if class_count:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(6,3))
                    ax.bar(class_count.keys(), class_count.values(), color='skyblue')
                    ax.set_xlabel("Class")
                    ax.set_ylabel("Số lượng")
                    ax.set_title("Số lượng từng class (Realtime)")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

        cap.release()
        cv2.destroyAllWindows()
