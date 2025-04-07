import streamlit as st
import os

st.set_page_config(page_title="Chroma AI", layout="wide")

st.title("🚀 Chroma AI: The Future of Inclusive Learning is Here!")

# Directory to save outputs
os.makedirs("output", exist_ok=True)

st.markdown("""
### 📚 **Struggling with Dense PDFs? Blurry Text? Complex Jargon?**
Let **Chroma AI** handle it for you. **Upload** your files, and we'll transform them into **clear, simple, and audible knowledge!**

### ⚡ **Features That Change the Game:**
✅ **Instant PDF/Image-to-Audio** – Let AI **read** for you.  
✅ **Lecture-to-Text** – No more struggling with notes!  
✅ **Text Simplification** – Complex language? **We break it down.**  
✅ **Visual Aid Generation** – **See** what you learn.  
✅ **(WIP) Sign Language to Text** – Breaking barriers, one sign at a time.  

🔽 **Use the sidebar to explore features!**
""")
