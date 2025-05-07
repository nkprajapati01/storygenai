import os
os.environ["STREAMLIT_WATCH_INSTALLER"] = "false"

import streamlit as st
from transformers import pipeline
import torch

torch.set_grad_enabled(False)

# Load a lightweight, safe model
@st.cache_resource
def load_story_generator():
    """
    Load GPT-2 pipeline for storytelling.
    """
    return pipeline("text-generation", model="gpt2", device=-1)

# UI
st.title("📖 Story Generator: Bring Your Imagination to Life!")
st.write("#### Create captivating short stories in seconds! ✍️✨")
st.markdown(
    """
    Welcome to **Story Generator**, your AI-powered storytelling assistant!  
    Provide a character and a setting, and watch as the magic unfolds. 🌠  
    Unleash your creativity and let AI bring your imagination to life! 🚀
    """
)

st.subheader("📝 Enter Your Story Prompt")
character_and_setting = st.text_input(
    "Describe a character and a setting:",
    placeholder="E.g., Write a short story about a knight who protects a magical forest."
)

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
                pad_token_id=50256  # GPT-2 uses this as eos_token_id
            )
            generated_story = story[0]["generated_text"]
            st.success("✅ Story generated! Here's your masterpiece:")
            st.text_area("📖 Your Story", value=generated_story, height=200)
        except Exception as e:
            st.error(f"⚠️ An error occurred while generating the story: {e}")
    else:
        st.warning("⚠️ Please enter a character and a setting to generate a story.")

st.markdown("---")
st.write("🌟 **Made by Vivek Chouhan** 🌟")
