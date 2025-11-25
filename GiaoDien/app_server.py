import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image

st.set_page_config(page_title='Phân loại rác bằng Server', layout='wide')

st.title('Client for FastAPI Model Server')
server_url = st.sidebar.text_input('Server URL', 'http://localhost:8000')
conf = st.sidebar.slider('Confidence', 0.0, 1.0, 0.35)

st.write('Upload an image or take a picture with camera. The app will send it to the model server and display returned image/detections.')

col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader('Upload image', type=['jpg','jpeg','png'])
    camera_img = st.camera_input('Or take photo')

if uploaded is not None or camera_img is not None:
    img_data = None
    if uploaded is not None:
        img_data = uploaded.read()
    else:
        img_data = camera_img.getvalue()

    files = {'file': ('image.jpg', img_data, 'image/jpeg')}
    with st.spinner('Sending to server...'):
        try:
            resp = requests.post(f"{server_url}/predict?conf={conf}", files=files, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            detections = data.get('detections', [])
            image_b64 = data.get('image_b64')

            if image_b64:
                img = Image.open(BytesIO(base64.b64decode(image_b64)))
                st.image(img, caption='Result from server', use_column_width=True)

            st.subheader('Detections')
            if detections:
                st.write(detections)
            else:
                st.write('No detections')

        except Exception as e:
            st.error(f'Error calling server: {e}')
else:
    st.info('Upload or take a photo to test server inference')

st.markdown('---')
st.markdown('Server run example: `uvicorn server.api:app --host 0.0.0.0 --port 8000`')
