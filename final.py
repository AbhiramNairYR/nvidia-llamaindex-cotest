import openai

# Initialize the OpenAI client
client = openai.OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key = "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"
    )

# Capture the user prompt from the terminal
user_prompt = input("Enter your prompt: ")

# Create the completion request with the user-provided prompt
completion = client.chat.completions.create(
    model="nvidia/mistral-nemo-minitron-8b-8k-instruct",
    messages=[{"role": "system", "content": "You are a helpful assistant that summarizes text, You create heading bullet points and mindmap of the summery."}
    ,{"role": "user", "content": user_prompt}],
    temperature=0.5,
    top_p=1,
    max_tokens=1024,
    stream=True
)

# Print the streamed response
print("\n chat :")
for chunk in completion:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")

import streamlit as st
# Heading
st.title("Summery Bot")

# Text boxes
link = st.text_input("Paste Link")
text = st.text_area("Add Text")

# File upload
uploaded_file = st.file_uploader("Upload Files from System")

# Select box
selection = st.selectbox("Selection", ["audio", "youtube link", "document", "text"])

# Summarization button
if st.button(f"{selection} Summary"):
    st.write(f"Summarizing {selection}...")
    # Add your summarization logic here

