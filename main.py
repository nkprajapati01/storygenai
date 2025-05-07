import streamlit as st
from transformers import pipeline

# Define function to load the text generation pipeline
@st.cache_resource
def load_story_generator():
    """
    Load the Hugging Face text generation pipeline with a pre-trained model.
    Device is set to CPU explicitly for compatibility.
    """
    return pipeline("text-generation", model="gpt2", device=-1)  # Use CPU

# Streamlit app
st.title("📖 Story Generator: Bring Your Imagination to Life! 🌟")
st.write("#### Create captivating short stories in seconds! ✍️✨")
st.markdown(
    """
    Welcome to **Story Generator**, your AI-powered storytelling assistant!  
    Provide a character and a setting, and watch as the magic unfolds. 🌠  
    Unleash your creativity and let AI bring your imagination to life! 🚀
    """
)

# Input: Character and Setting
st.subheader("📝 Enter Your Story Prompt")
character_and_setting = st.text_input("Describe a character and a setting:", placeholder="E.g., A robot named Max in a desert")

# Generate button
if st.button("✨ Generate Story"):
    if character_and_setting.strip():
        try:
            # Load story generator pipeline
            story_generator = load_story_generator()
            # Generate a short story
            story = story_generator(character_and_setting, max_length=100, num_return_sequences=1)
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
