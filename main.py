import os
os.environ["STREAMLIT_WATCH_INSTALLER"] = "false"  # Prevent Streamlit watcher conflict with torch

import streamlit as st
from transformers import pipeline
import torch

torch.set_grad_enabled(False)  # Disable autograd for inference (safer, faster)

# Load text generation pipeline with caching
@st.cache_resource
def load_story_generator():
    """
    Load a fine-tuned Hugging Face text generation pipeline for storytelling.
    """
    return pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B", device=-1)  # CPU usage

# Streamlit app UI
st.title("📖 Story Generator: Bring Your Imagination to Life!")
st.write("#### Create captivating short stories in seconds! ✍️✨")
st.markdown(
    """
    Welcome to **Story Generator**, your AI-powered storytelling assistant!  
    Provide a character and a setting, and watch as the magic unfolds. 🌠  
    Unleash your creativity and let AI bring your imagination to life! 🚀
    """
)

# User input
st.subheader("📝 Enter Your Story Prompt")
character_and_setting = st.text_input(
    "Describe a character and a setting:",
    placeholder="E.g., Write a short story about a knight who protects a magical forest."
)

# Generate button
if st.button("✨ Generate Story"):
    if character_and_setting.strip():
        story_generator = load_story_generator()
        try:
            story = story_generator(
                character_and_setting,
                max_length=150,
                truncation=True,
                num_return_sequences=1,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=50256
            )
            generated_story = story[0]["generated_text"]
            st.success("✅ Story generated! Here's your masterpiece:")
            st.text_area("📖 Your Story", value=generated_story, height=200)
        except Exception as e:
            st.error(f"⚠️ An error occurred while generating the story: {e}")
    else:
        st.warning("⚠️ Please enter a character and a setting to generate a story.")

# Footer
st.markdown("---")
st.write("🌟 **Powered by [Hugging Face Transformers](https://huggingface.co/transformers) and Streamlit**. Let your imagination soar! 🌟")
