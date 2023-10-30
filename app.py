# Import necessary libraries
import streamlit as st
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import io

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "segmind/SSD-1B",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    pipe.to("cuda")
    return pipe

# Function to convert PIL image to byte array
def get_image_download_link(img):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="image.jpg">Download Image</a>'
    return href

# Main function for Streamlit app
def main():
    st.title("Image Generation with Stable Diffusion")
    
    # Create text input boxes for prompts
    prompt = st.text_input("Enter your prompt")
    neg_prompt = st.text_input("Enter your negative prompt")
    
    # Create a button for generating the image
    if st.button("Generate Image"):
        model = load_model()
        image = model(prompt=prompt, negative_prompt=neg_prompt).images[0]
        
        # Display the image
        st.image(image, use_column_width=True)
        
        # Create a download link for the image
        st.markdown(get_image_download_link(image), unsafe_allow_html=True)

# Run the main function
if __name__ == "__main__":
    main()
