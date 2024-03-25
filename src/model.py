import re
import torch
import requests
import torchaudio
import numpy as np
# from src.reduce_noise import smooth_and_reduce_noise, model_remove_noise, model, df_state
import io
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from pydub import AudioSegment
import re
from uroman import uroman
# from src.pynote_speaker_embedding import create_speaker_embedding
from src.speechbrain_speaker_embedding import create_speaker_embedding

from datasets import load_dataset
dataset = load_dataset("truong-xuan-linh/vi-xvector-speechbrain", 
                       download_mode="force_redownload", 
                            verification_mode="no_checks", 
                            cache_dir="temp/",
                            revision="5ea5e4345258333cbc6d1dd2544f6c658e66a634")
dataset = dataset["train"].to_list()

dataset_dict = {}

for rc in dataset:
    dataset_dict[rc["speaker_id"]] = rc["embedding"]
    
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

def remove_special_characters(sentence):
    # Use regular expression to keep only letters, periods, and commas
    sentence_after_removal =  re.sub(r'[^a-zA-Z\s,.\u00C0-\u1EF9]', ' ', sentence)
    return sentence_after_removal

from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def korean_splitter(string):
    pattern = re.compile('[가-힣]+')
    matches = pattern.findall(string)
    return matches

def uroman_normalization(string):
    korean_inputs = korean_splitter(string)
    for korean_input in korean_inputs:
        korean_roman = uroman(korean_input)
        string = string.replace(korean_input, korean_roman)
    return string

class Model():
    
    def __init__(self, model_name, speaker_url=""):
        self.model_name = model_name
        self.processor = SpeechT5Processor.from_pretrained(model_name)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_name)
        # self.model.generate = partial(self.model.generate, use_cache=True)

        self.model.eval()
        
        self.speaker_url = speaker_url
        if speaker_url:
            
            print(f"download speaker_url")
            response = requests.get(speaker_url)
            audio_stream = io.BytesIO(response.content)
            audio_segment = AudioSegment.from_file(audio_stream, format="wav")
            audio_segment = audio_segment.set_channels(1)
            audio_segment = audio_segment.set_frame_rate(16000)
            audio_segment = audio_segment.set_sample_width(2)
            wavform, _ = torchaudio.load(audio_segment.export())
            self.speaker_embeddings = create_speaker_embedding(wavform)[0]
        else:
            self.speaker_embeddings = None
        
        if model_name == "truong-xuan-linh/speecht5-vietnamese-commonvoice" or model_name == "truong-xuan-linh/speecht5-irmvivoice":
            self.speaker_embeddings = torch.zeros((1, 512))  # or load xvectors from a file
            
    def inference(self, text, speaker_id=None):
        # if self.model_name == "truong-xuan-linh/speecht5-vietnamese-voiceclone-v2":
        #     # self.speaker_embeddings = torch.tensor(dataset_dict_v2[speaker_id])
        #     wavform, _ = torchaudio.load(speaker_id)
        #     self.speaker_embeddings = create_speaker_embedding(wavform)[0]
            
        if "voiceclone" in self.model_name:
            if not self.speaker_url:
                self.speaker_embeddings = torch.tensor(dataset_dict[speaker_id])
            # self.speaker_embeddings = create_speaker_embedding(speaker_id)[0]
            # wavform, _ = torchaudio.load("voices/kcbn1.wav")
            # self.speaker_embeddings = create_speaker_embedding(wavform)[0]
            # wavform, _ = torchaudio.load(wav_file)
            # self.speaker_embeddings = create_speaker_embedding(wavform)[0]
        
            
        with torch.no_grad():
            full_speech = []
            separators = r";|\.|!|\?|\n"
            text = uroman_normalization(text)
            text = text.replace(" ", "▁")
            split_texts = re.split(separators, text)
            
            for split_text in split_texts:
                
                if split_text != "▁":
                    # split_text = remove_special_characters(" ," + split_text) + " ,"
                    split_text = split_text.lower() + "▁"
                    print(split_text)
                    inputs = self.processor.tokenizer(text=split_text, return_tensors="pt")
                    speech = self.model.generate_speech(inputs["input_ids"], threshold=0.5, speaker_embeddings=self.speaker_embeddings, vocoder=vocoder)
                    full_speech.append(speech.numpy())
                    # full_speech.append(butter_bandpass_filter(speech.numpy(), lowcut=10, highcut=5000, fs=16000, order=2))
            # out_audio = model_remove_noise(model, df_state, np.concatenate(full_speech))
            return np.concatenate(full_speech)
    
    @staticmethod
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')
    
# woman: VIVOSSPK26, VIVOSSPK02, VIVOSSPK40

# man: VIVOSSPK28, VIVOSSPK36, VIVOSDEV09, VIVOSSPK33, VIVOSSPK23

