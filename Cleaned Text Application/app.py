import streamlit as st
import string
import re
from io import StringIO, BytesIO
from nltk.corpus import stopwords
import emoji
import nltk

nltk.download("stopwords")

# Set the page configuration
st.set_page_config(page_title="Text Cleaner", layout="wide")

# Title and description
st.title("Text Cleaner Application")
st.markdown("<h5>This application allows you to upload a text file and clean the text based on your preferences.</h5>", unsafe_allow_html=True)

# Function to clean the text
def clean_text(text, language, keep_emojis, keep_stopwords, keep_punctuation):
    if not keep_emojis:
        text = emoji.replace_emoji(text, replace='')

    if not keep_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))

    if not keep_stopwords:
        stop_words = set(stopwords.words(language))
        text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
    
    
    return text

# text file uploader s
uploaded_file = st.file_uploader("Upload your text file...", type=["txt"])

if uploaded_file is not None:
    # Read the text file
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    text = stringio.read()

    # Sidebar options for cleaning
    st.sidebar.title("Cleaning Options")
    
    # Language selection
    language = st.sidebar.selectbox("Select the language of the text:", ["english", "french", "spanish", "german"])
    
    # Option to keep or remove emojis
    keep_emojis = st.sidebar.checkbox("Keep Emojis", value=True)
    
    # Option to keep or remove stopwords
    keep_stopwords = st.sidebar.checkbox("Keep Stopwords", value=True)
    
    # Option to keep or remove punctuation
    keep_punctuation = st.sidebar.checkbox("Keep Punctuation", value=True)

    # Clean the text based on user selections
    cleaned_text = clean_text(text, language, keep_emojis, keep_stopwords, keep_punctuation)
    
    # Display the cleaned text
    st.subheader("Cleaned Text")
    st.text_area("Here is your cleaned text:", cleaned_text, height=300)

    # Allow the user to download the cleaned text
    cleaned_text_bytes = BytesIO(cleaned_text.encode("utf-8"))
    st.download_button(
        label="Download Cleaned Text",
        data=cleaned_text_bytes,
        file_name="cleaned_text.txt",
        mime="text/plain"
    )

else:
    st.info("Please upload a text file to get started.")

st.markdown("<center><h6>made by yasmine baroud </h6></center>", unsafe_allow_html=True)
