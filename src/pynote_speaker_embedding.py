# import torch
# from pyannote.audio import Model, Inference
# speaker_model = Model.from_pretrained("pyannote/embedding",
#                               use_auth_token="")
# inference = Inference(speaker_model, window="whole")

# def create_speaker_embedding(audio_dir):
#     with torch.no_grad():
#         embedding = inference(audio_dir)
#         embedding = torch.tensor([[embedding]])
#         speaker_embeddings = torch.nn.functional.normalize(embedding, dim=-1)
#     return speaker_embeddings