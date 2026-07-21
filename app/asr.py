import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
SAMPLE_RATE = 16_000
CHUNK_SECONDS = 30  # on découpe les longs audios en segments de 30 s


class ASRModel:
    """Transcription français -> texte avec Wav2Vec 2.0 (CTC)."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        self.model.eval()  # mode inférence : désactive dropout

    @torch.inference_mode()  # pas de gradients -> moins de RAM, plus rapide
    def transcribe(self, waveform: torch.Tensor) -> str:
        """waveform : tenseur 1D mono 16 kHz (sortie de load_and_preprocess)."""
        chunk_size = CHUNK_SECONDS * SAMPLE_RATE
        texts = []

        for start in range(0, waveform.shape[0], chunk_size):
            chunk = waveform[start : start + chunk_size]

            inputs = self.processor(
                chunk.numpy(),
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
            )
            logits = self.model(inputs.input_values.to(self.device)).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            texts.append(self.processor.batch_decode(predicted_ids)[0])

        return " ".join(t.strip() for t in texts if t.strip()).lower()