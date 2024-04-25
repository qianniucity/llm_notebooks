import whisper
import whisperx

from base.config import Config


class WhisperX(Config):
    """Transcription Model"""

    def __init__(self, model_name: str):
        """
        Init method
        Args:
            model (str): whisper or whisper x
        """
        super().__init__()
        self.model_name = model_name
        if self.model_name == "whisper":
            self.model = whisper.load_model(
                self.config["transcriptor"][model_name]["model"],
                self.config["transcriptor"]["device"],
            )
        else:
            self.model = whisperx.load_model(
                self.config["transcriptor"][model_name]["model"],
                self.config["transcriptor"]["device"],
                compute_type=self.config["transcriptor"][model_name]["compute_type"],
            )

    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribes the audio
        Args:
            audio_path (str): path to .wav file
        Returns:
            result_aligned (dict): dictionary with segments and metadata
        """
        if self.model_name == "whisperx":
            audio_path = whisperx.load_audio(audio_path)

        result = self.model.transcribe(audio_path)
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=self.config["transcriptor"]["device"],
        )
        result_aligned = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio_path,
            self.config["transcriptor"]["device"],
        )

        return result_aligned
