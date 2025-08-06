"""Text-to-Speech service using Chatterbox TTS."""

import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import numpy as np

logger = logging.getLogger(__name__)


class TTSService:
    """Service for text-to-speech conversion using Chatterbox TTS."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize TTS service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.tts_config = config.get("tts", {})
        self.model = None
        self.sample_rate = None
        self.device = self.tts_config.get("device", "cuda")
        self.loaded = False
        
        # Audio parameters
        self.exaggeration = self.tts_config.get("exaggeration", 0.5)
        self.cfg_weight = self.tts_config.get("cfg_weight", 0.5)
        self.audio_prompt_path = self.tts_config.get("audio_prompt_path", None)
        
        # Output settings
        self.output_dir = Path(self.tts_config.get("output_dir", "data/tts_output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_model(self) -> None:
        """Load the Chatterbox TTS model."""
        if self.loaded:
            logger.info("TTS model already loaded")
            return
            
        try:
            logger.info(f"Loading Chatterbox TTS model on {self.device}...")
            
            # Import here to avoid dependency issues if TTS is not used
            from chatterbox.tts import ChatterboxTTS
            
            # Load model with specified device
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            self.sample_rate = self.model.sr
            self.loaded = True
            
            logger.info(f"TTS model loaded successfully (sample rate: {self.sample_rate}Hz)")
            
        except ImportError as e:
            logger.error("Chatterbox TTS not installed. Run: pip install chatterbox-tts")
            raise ImportError("Please install chatterbox-tts: pip install chatterbox-tts") from e
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            raise
    
    def synthesize(
        self,
        text: str,
        output_path: Optional[Union[str, Path]] = None,
        audio_prompt_path: Optional[Union[str, Path]] = None,
        exaggeration: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        return_audio: bool = False
    ) -> Union[Path, np.ndarray, tuple]:
        """Synthesize speech from text.
        
        Args:
            text: Text to convert to speech
            output_path: Optional path to save audio file
            audio_prompt_path: Optional path to audio file for voice cloning
            exaggeration: Emotion exaggeration level (0.0 to 1.0)
            cfg_weight: Classifier-free guidance weight
            return_audio: If True, return audio array (and optionally path)
            
        Returns:
            Path to saved audio file, audio array, or both
        """
        if not self.loaded:
            self.load_model()
        
        try:
            # Use provided parameters or defaults
            exaggeration = exaggeration if exaggeration is not None else self.exaggeration
            cfg_weight = cfg_weight if cfg_weight is not None else self.cfg_weight
            audio_prompt = audio_prompt_path or self.audio_prompt_path
            
            logger.info(f"Synthesizing speech for text: '{text[:50]}...'")
            logger.debug(f"Parameters: exaggeration={exaggeration}, cfg_weight={cfg_weight}")
            
            # Generate audio
            wav = self.model.generate(
                text,
                audio_prompt_path=audio_prompt,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight
            )
            
            # Save audio if path provided or auto-generate path
            saved_path = None
            if output_path or not return_audio:
                if output_path is None:
                    # Auto-generate filename
                    import time
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_path = self.output_dir / f"tts_{timestamp}.wav"
                else:
                    output_path = Path(output_path)
                    
                # Save audio file
                import torchaudio
                torchaudio.save(str(output_path), wav, self.sample_rate)
                logger.info(f"Audio saved to: {output_path}")
                saved_path = output_path
            
            # Return based on requested format
            if return_audio and saved_path:
                return wav.numpy(), saved_path
            elif return_audio:
                return wav.numpy()
            else:
                return saved_path
                
        except Exception as e:
            logger.error(f"Failed to synthesize speech: {e}")
            raise
    
    def synthesize_batch(
        self,
        texts: List[str],
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> List[Path]:
        """Synthesize speech for multiple texts.
        
        Args:
            texts: List of texts to convert
            output_dir: Directory to save audio files
            **kwargs: Additional arguments for synthesize()
            
        Returns:
            List of paths to saved audio files
        """
        output_dir = Path(output_dir) if output_dir else self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = []
        for i, text in enumerate(texts, 1):
            logger.info(f"Processing text {i}/{len(texts)}")
            try:
                output_path = output_dir / f"tts_batch_{i:03d}.wav"
                path = self.synthesize(text, output_path=output_path, **kwargs)
                paths.append(path)
            except Exception as e:
                logger.error(f"Failed to synthesize text {i}: {e}")
                paths.append(None)
        
        return paths
    
    def convert_voice(
        self,
        text: str,
        voice_sample_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Path:
        """Convert text to speech using a voice sample.
        
        Args:
            text: Text to convert
            voice_sample_path: Path to voice sample for cloning
            output_path: Optional path to save output
            **kwargs: Additional synthesis parameters
            
        Returns:
            Path to saved audio file
        """
        logger.info(f"Converting text with voice from: {voice_sample_path}")
        return self.synthesize(
            text,
            output_path=output_path,
            audio_prompt_path=voice_sample_path,
            **kwargs
        )
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.loaded and self.model:
            # Free model resources
            del self.model
            self.model = None
            self.loaded = False
            
            # Clear GPU cache if using CUDA
            if self.device.startswith("cuda"):
                try:
                    import torch
                    torch.cuda.empty_cache()
                except:
                    pass
            
            logger.info("TTS model cleaned up")


class EmotionalTTSService(TTSService):
    """Extended TTS service with emotional presets."""
    
    # Emotional presets for different speaking styles
    EMOTION_PRESETS = {
        "neutral": {"exaggeration": 0.5, "cfg_weight": 0.5},
        "excited": {"exaggeration": 0.8, "cfg_weight": 0.3},
        "dramatic": {"exaggeration": 0.9, "cfg_weight": 0.3},
        "calm": {"exaggeration": 0.3, "cfg_weight": 0.6},
        "fast": {"exaggeration": 0.7, "cfg_weight": 0.3},
        "slow": {"exaggeration": 0.4, "cfg_weight": 0.7},
        "emphatic": {"exaggeration": 0.85, "cfg_weight": 0.35},
    }
    
    def synthesize_with_emotion(
        self,
        text: str,
        emotion: str = "neutral",
        output_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Union[Path, np.ndarray]:
        """Synthesize speech with emotional preset.
        
        Args:
            text: Text to convert
            emotion: Emotion preset name
            output_path: Optional output path
            **kwargs: Additional parameters
            
        Returns:
            Path to saved audio or audio array
        """
        if emotion not in self.EMOTION_PRESETS:
            logger.warning(f"Unknown emotion '{emotion}', using neutral")
            emotion = "neutral"
        
        preset = self.EMOTION_PRESETS[emotion]
        logger.info(f"Using emotion preset '{emotion}': {preset}")
        
        return self.synthesize(
            text,
            output_path=output_path,
            exaggeration=preset["exaggeration"],
            cfg_weight=preset["cfg_weight"],
            **kwargs
        )