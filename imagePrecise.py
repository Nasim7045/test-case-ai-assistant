import os
import google.generativeai as genai
import fitz  # PyMuPDF
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import re
import pytesseract  # For OCR

# Set up Google Generative AI API
api_key = "Api-key"
genai.configure(api_key=api_key)

# Configure generation settings
generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize models
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)
chat_session = model.start_chat(history=[])

# Initialize BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
device = "cuda" if torch.cuda.is_available() else "cpu"
caption_model = caption_model.to(device)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        pdf_document = fitz.open(stream=pdf_file, filetype="pdf")
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text("text")
    except Exception as e:
        st.error(f"PDF extraction error: {e}")
    return text

# Text cleaning function
def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Image captioning function
def generate_image_caption(image):
    try:
        image = image.convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)
        outputs = caption_model.generate(**inputs, max_length=50, num_beams=5)
        return processor.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Caption error: {e}"

# OCR function
def extract_image_text(image):
    try:
        return pytesseract.image_to_string(image)
    except Exception as e:
        return f"OCR error: {e}"

# Streamlit UI
st.title("Enhanced Multi-functional AI Assistant")
tab1, tab2, tab3 = st.tabs(["Document Analysis", "Image Analysis", "Prompting"])

with tab1:
    st.header("Document Analysis")
    uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_pdf:
        pdf_text = clean_text(extract_text_from_pdf(uploaded_pdf.read()))
        st.session_state['pdf_text'] = pdf_text
        st.success("PDF processed!")
        
        if st.checkbox("Show extracted text"):
            st.text_area("PDF Text", pdf_text, height=300)
        
        question = st.text_input("Ask about the PDF:")
        if question and 'pdf_text' in st.session_state:
            input_text = f"PDF Content:\n{st.session_state['pdf_text']}\n\nQuestion: {question}"
            response = chat_session.send_message(input_text)
            st.write("**Response:**", response.text)

with tab2:
    st.header("Image Analysis")
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, use_column_width=True)
        
        # Define a function to analyze the image
        def analyze_image(img, prompt=""):
            with st.spinner("Analyzing image..."):
                caption = generate_image_caption(img)
                ocr_text = clean_text(extract_image_text(img))
                
                # Enhanced prompt integration for precision
                if prompt:
                    # Re-generate caption with the prompt in mind
                    enhanced_caption_prompt = f"Describe the image precisely, focusing on elements related to: {prompt}. Be concise."
                    inputs = processor(image, text=enhanced_caption_prompt, return_tensors="pt").to(device)
                    outputs = caption_model.generate(**inputs, max_length=75, num_beams=5)
                    caption = processor.decode(outputs[0], skip_special_tokens=True)

                return {
                    'caption': caption,
                    'ocr_text': ocr_text
                }

        # Initialize image data in session state only if it doesn't exist
        if 'image_data' not in st.session_state:
            st.session_state['image_data'] = {}

        # Continuous questioning
        question = st.text_input("Ask about the image:")
        if question:
            # Re-analyze the image with the question as a prompt
            image_data = analyze_image(image, prompt=question)
            st.session_state['image_data'][question] = image_data  # Store this specific analysis
            
            st.subheader("Image Insights")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Visual Analysis:**", image_data['caption'])
            with col2:
                st.write("**Extracted Text:**", image_data['ocr_text'] if image_data['ocr_text'] else "No text detected")

            context = f"Visual context: {image_data['caption']}. Text in image: {image_data['ocr_text']}"
            response = chat_session.send_message(f"{context}\n\nQuestion: {question}")
            st.write("**Response:**", response.text)
        else:
            # If no question is asked, perform a general analysis
            if not st.session_state['image_data']:  # Only do initial analysis once
                image_data = analyze_image(image)
                st.session_state['image_data']['general'] = image_data

            if 'general' in st.session_state['image_data']:
                data = st.session_state['image_data']['general']
                st.subheader("Image Insights")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Visual Analysis:**", data['caption'])
                with col2:
                    st.write("**Extracted Text:**", data['ocr_text'] if data['ocr_text'] else "No text detected")


with tab3:
    st.header("General Prompting")
    prompt = st.text_input("Ask anything:")
    if prompt:
        response = chat_session.send_message(prompt)
        st.write("**Response:**", response.text)
