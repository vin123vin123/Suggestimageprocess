import base64
import math
import cv2
import numpy as np
import streamlit as st
from pymongo import MongoClient

# --- DATABASE SETUP ---
MONGO_URI = MONGO_URI = st.secrets["MONGO_URI"]
client = MongoClient(MONGO_URI)
db = client["StreamlitImageDB"]
collection = db["universal_chunks"]

# Create an index immediately on app start to guarantee fast lookups
collection.create_index("image_id")

st.title("🧩 Unordered Image Splitter & Database Assembler")


# --- CACHED FETCHING FOR MAXIMUM PERFORMANCE ---
@st.cache_data(show_spinner="Fetching fragments from Database...")
def fetch_and_stitch_image(image_id):
    """Fetches chunks out-of-order from MongoDB and stitches them into a canvas."""
    chunks = list(collection.find({"image_id": image_id}))
    if not chunks:
        return None

    meta = chunks[0]["metadata"]
    # Pre-allocate canvas array
    canvas = np.zeros((meta["total_height"], meta["total_width"], 3), dtype=np.uint8)

    for chunk in chunks:
        slice_info = chunk["slice_info"]
        x_off = slice_info["x_offset"]
        y_off = slice_info["y_offset"]

        # Decode base64 stream
        img_bytes = base64.b64decode(chunk["data_stream"].encode("utf-8"))
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        slice_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Drop block directly onto coordinates
        h, w, _ = slice_img.shape
        canvas[y_off : y_off + h, x_off : x_off + w] = slice_img

    # Convert BGR (OpenCV standard) to RGB (Streamlit standard)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


# --- UI COMPONENT: UPLOAD & PROCESS ---
uploaded_file = st.file_uploader("Upload an Image File", type=["jpg", "jpeg", "png"])
image_tag = st.text_input("Assign a Unique Image ID Tag", value="project_canvas_01")

if uploaded_file is not None and st.button("🚀 Slice & Push Scrambled to MongoDB"):
    # Convert uploaded stream to OpenCV matrix
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    height, width, _ = img.shape
    rows, cols = 3, 3  # 3x3 Grid
    slice_h = math.ceil(height / rows)
    slice_w = math.ceil(width / cols)

    documents = []

    with st.spinner("Slicing image grids into text payloads..."):
        for r in range(rows):
            for c in range(cols):
                y_start, y_end = r * slice_h, min((r + 1) * slice_h, height)
                x_start, x_end = c * slice_w, min((c + 1) * slice_w, width)

                crop = img[y_start:y_end, x_start:x_end]
                _, buffer = cv2.imencode(".png", crop)
                b64_string = base64.b64encode(buffer).decode("utf-8")

                documents.append(
                    {
                        "image_id": image_tag,
                        "metadata": {
                            "total_width": width,
                            "total_height": height,
                        },
                        "slice_info": {"x_offset": x_start, "y_offset": y_start},
                        "data_stream": b64_string,
                    }
                )

    # Push to DB
    collection.insert_many(documents)
    st.success(f"Successfully scrambled and saved {len(documents)} blocks into DB!")
    # Clear cache to ensure any fresh replacement loads cleanly
    st.cache_data.clear()

# --- UI COMPONENT: FETCH & VIEW ---
st.divider()
st.subheader("📥 Retrieve and Compile Canvas")

if st.button("Fetch and Re-Assemble Image"):
    result_canvas = fetch_and_stitch_image(image_tag)

    if result_canvas is not None:
        st.image(
            result_canvas,
            caption="Perfectly integrated image from unordered database chunks",
        )
    else:
        st.warning("No coordinate documents found matching that Image ID.")

