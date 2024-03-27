import streamlit as st
st.set_page_config(page_title="SpeechT5", page_icon = "static/images/PLAYGROUND_LOGO_REDESIGN_IMAGE.png")
hide_menu_style = """
<style>
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_menu_style, unsafe_allow_html= True)

st.write("Languages: Vietnamese, English")
import glob
from src.model import Model, dataset_dict

if "model_name" not in st.session_state:
    st.session_state.model_name = None
    st.session_state.audio = None
    st.session_state.wav_file = None
    st.session_state.speaker_url = ""
    
with st.sidebar.form("my_form"):

    text = st.text_input("Your input: ")
    model_name = st.selectbox(label="Model: ", options=["truong-xuan-linh/speecht5-vietnamese-voiceclone-lsvsc",
                                                        "truong-xuan-linh/speecht5-vietnamese-commonvoice", 
                                                        "truong-xuan-linh/speecht5-vietnamese-hlpcvoice",
                                                        "truong-xuan-linh/speecht5-vietnamese-vstnvoice",
                                                        "truong-xuan-linh/speecht5-vietnamese-kcbnvoice",
                                                        "truong-xuan-linh/speecht5-irmvivoice",
                                                        "truong-xuan-linh/speecht5-vietnamese-voiceclone",
                                                        "truong-xuan-linh/speecht5-multilingual-voiceclone-speechbrain",
                                                        "truong-xuan-linh/speecht5-vietnamese-voiceclone-v3",
                                                        "truong-xuan-linh/speecht5-multilingual-voiceclone-pynote",
                                                        "truong-xuan-linh/speecht5-multilingual-voiceclone-speechbrain-nonverbal"])
    
    speaker_id = st.selectbox("source voice", options= ["speech_dataset_denoised"] + list(dataset_dict.keys()))
    speaker_url = st.text_input("speaker url", value="")
    # speaker_id = st.selectbox("source voice", options= glob.glob("voices/*.wav"))
    if st.session_state.model_name != model_name or speaker_url != st.session_state.speaker_url :
        st.session_state.model_name = model_name
        st.session_state.model = Model(model_name=model_name, speaker_url=speaker_url)
        st.session_state.speaker_id = speaker_id
        st.session_state.speaker_url = speaker_url
        
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.session_state.audio = st.session_state.model.inference(text=text, speaker_id=speaker_id)
        
audio_holder = st.empty()
audio_holder.audio(st.session_state.audio, sample_rate=16000)