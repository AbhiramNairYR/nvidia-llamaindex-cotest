import openai
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import streamlit as st
import yt_dlp
import subprocess

# Set up device and model details
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load model and processor
model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

# Streamlit UI
st.title("Summary Bot")
key = st.text_input("Enter your api key")
link = st.text_input("Paste YouTube Link")
text = st.text_area("Add Text")
uploaded_file = st.file_uploader("Upload Audio File from System (Limit: 500MB)")
selection = st.selectbox("Selection", ["youtube link", "audio", "text"])

# Function to download audio from YouTube
def download_audio(link):
    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': 'audio.%(ext)s'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([link])
            st.success("Download completed successfully!")
            return 'audio.wav'  # Expected output filename based on outtmpl
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None

# Function to transcribe audio
def transcription(audio_source):
    # Set up ASR pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=32,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Run transcription
    result = pipe(audio_source)
    st.write("Transcription:", result["text"])
    return result["text"]

def summery(content):
    # Initialize the OpenAI client
    client = openai.OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key = key
        )

    # Create the completion request with the content
    completion = client.chat.completions.create(
        model="nvidia/mistral-nemo-minitron-8b-8k-instruct",
        messages=[{"role": "system", "content": "You are a helpful assistant that summarizes text under 250 words."}
        ,{"role": "user", "content": content}],
        temperature=0.5,
        top_p=1,
        max_tokens=1024,
        stream=True
    )

    # Collect the streamed response
    summary_text = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            summary_text += chunk.choices[0].delta.content

    st.write("Summary:", summary_text)
    return summary_text

def speach(summary_text):
    command = [
        "python", "python-clients/scripts/tts/talk.py",
        "--server", "grpc.nvcf.nvidia.com:443",
        "--use-ssl",
        "--metadata", "function-id", "0149dedb-2be8-4195-b9a0-e57e0e14f972",
        "--metadata", "authorization", f"Bearer {key}",
        "--text", summary_text,
        "--voice", "English-US.Female-1",
        "--output", "audio.wav"
    ]
    
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print("Command output:", result.stdout)
        st.success("Speech synthesis command executed successfully.")
        
        # Display audio player in Streamlit
        st.audio("audio.wav", format="audio/wav")
    except subprocess.CalledProcessError as e:
        print("An error occurred:", e.stderr)
        st.error("An error occurred while running the speech synthesis command.")

# Process the selected input and trigger summary
if st.button(f"{selection} Summary"):
    if selection == "youtube link" and link:
        st.write(f"Summarizing {selection}...")
        audio_path = download_audio(link)
        if audio_path:
            transcribed_text = transcription(audio_path)
            summary_text = summery(transcribed_text)
            speach(summary_text)
    elif selection == "audio" and uploaded_file:
        # Save uploaded file to disk
        audio_path = f"{uploaded_file.name}"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        transcribed_text = transcription(audio_path)
        summary_text = summery(transcribed_text)
        speach(summary_text)
    elif selection == "text" and text:
        st.write(f"Summarizing text: {text[:100]}...")  # Displaying a preview of the text
        summary_text = summery(text)
        speach(summary_text)
    else:
        st.warning("Please provide valid input based on the selection.")


#https://www.youtube.com/watch?v=ExbeHrdsmuA