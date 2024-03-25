import scipy.signal
import librosa
from df.enhance import enhance, init_df, load_audio, save_audio
import torch
from torchaudio.functional import resample

# Load default model
model, df_state, _ = init_df()

def smooth_and_reduce_noise(audio_signal, sampling_rate):
    # Apply a low-pass filter for smoothing
    cutoff_frequency = 1700  # Adjust as needed
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = scipy.signal.butter(4, normal_cutoff, btype='low', analog=False)
    smoothed_signal = scipy.signal.filtfilt(b, a, audio_signal)

    # Reduce noise using librosa's denoiser
    denoised_signal = librosa.effects.preemphasis(smoothed_signal, coef=0.95)

    return denoised_signal

def model_remove_noise(model, df_state, np_audio):
    #Read audio
    audio = torch.tensor([np_audio])
    audio = resample(audio, 16000, df_state.sr())
    
    #Inference
    enhanced = enhance(model, df_state, audio).cpu().numpy()
    
    #Save
    dtype=torch.int16
    out_audio = torch.as_tensor(enhanced)
    if out_audio.ndim == 1:
        out_audio.unsqueeze_(0)
    if dtype == torch.int16 and out_audio.dtype != torch.int16:
        out_audio = (out_audio * (1 << 15)).to(torch.int16)
    if dtype == torch.float32 and out_audio.dtype != torch.float32:
        out_audio = out_audio.to(torch.float32) / (1 << 15)
        
    out_audio = resample(audio, df_state.sr(), 16000)
    
    return out_audio.cpu().numpy()