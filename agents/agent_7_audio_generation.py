"""
Agent 7: Audio Generation Agent

This agent handles text-to-speech synthesis and audio processing for meditation
content. It manages voice generation, audio quality optimization, and format
conversion for the final meditation audio track.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
from pathlib import Path
import tempfile
import hashlib

from .base import BaseAgent, AgentConfig
from .agent_3_script_writer import VoiceScript


class AudioFormat(Enum):
    """Supported audio formats for output."""
    MP3 = "mp3"
    WAV = "wav"
    FLAC = "flac"
    OGG = "ogg"


class VoiceModel(Enum):
    """Available voice models for meditation content."""
    SARAH_CALM = "sarah_calm"
    DAVID_GENTLE = "david_gentle"
    MARIA_SOOTHING = "maria_soothing"
    ALEX_MINDFUL = "alex_mindful"
    CUSTOM_MEDITATION = "custom_meditation"


@dataclass
class AudioSettings:
    """Configuration for audio generation."""
    voice_model: VoiceModel
    sample_rate: int = 22050
    bit_depth: int = 16
    format: AudioFormat = AudioFormat.MP3
    quality: str = "high"
    normalize_audio: bool = True
    add_fade_in: bool = True
    add_fade_out: bool = True
    compression_ratio: float = 2.0
    noise_reduction: bool = True


@dataclass
class AudioSegment:
    """Individual audio segment with metadata."""
    segment_id: str
    start_time: float
    duration: float
    file_path: str
    waveform_data: Optional[bytes] = None
    audio_quality_score: float = 0.0
    processing_notes: List[str] = None


@dataclass
class GeneratedAudio:
    """Complete generated audio with metadata."""
    title: str
    total_duration: float
    audio_file_path: str
    segments: List[AudioSegment]
    settings: AudioSettings
    quality_metrics: Dict[str, float]
    file_size_mb: float
    processing_time_seconds: float
    voice_consistency_score: float
    audio_fingerprint: str


class AudioGenerationAgent(BaseAgent):
    """
    Agent 7: Audio Generation and Voice Synthesis
    
    This agent handles the complete audio generation pipeline for meditation
    content, including text-to-speech synthesis, audio processing, and
    quality optimization.
    
    Capabilities:
    - High-quality voice synthesis
    - SSML processing for natural speech
    - Audio post-processing and enhancement
    - Multiple voice model support
    - Quality assessment and optimization
    - Format conversion and compression
    - Batch processing for efficiency
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Audio Generation Agent."""
        if config is None:
            config = AgentConfig(
                name="audio_generation",
                phase=2,
                timeout_seconds=300,
                max_retries=3,
                required_capabilities=[
                    "text_to_speech",
                    "audio_processing",
                    "voice_synthesis",
                    "quality_optimization",
                    "format_conversion"
                ]
            )
        
        super().__init__(config)
        self.temp_dir = Path(tempfile.mkdtemp())
        self.voice_api_endpoint = "https://api.elevenlabs.io/v1/text-to-speech"
        self.supported_formats = [AudioFormat.MP3, AudioFormat.WAV]
        
    async def execute(self, context: Dict[str, Any]) -> GeneratedAudio:
        """
        Generate high-quality audio from voice script.
        
        Args:
            context: Pipeline context containing voice script
            
        Returns:
            GeneratedAudio with complete audio content
        """
        self.logger.info("Starting audio generation process")
        
        # Extract voice script
        voice_script = context.get("voice_script")
        if not voice_script:
            raise ValueError("Voice script required for audio generation")
        
        # Phase 1: Initialize audio settings
        self.logger.info("Configuring audio generation settings")
        audio_settings = await self._configure_audio_settings(voice_script)
        
        # Phase 2: Prepare segments for synthesis
        self.logger.info("Preparing script segments for voice synthesis")
        prepared_segments = await self._prepare_segments_for_synthesis(voice_script)
        
        # Phase 3: Generate audio segments
        self.logger.info("Generating audio segments with voice synthesis")
        audio_segments = await self._generate_audio_segments(prepared_segments, audio_settings)
        
        # Phase 4: Process and enhance audio
        self.logger.info("Processing and enhancing audio quality")
        enhanced_segments = await self._enhance_audio_quality(audio_segments, audio_settings)
        
        # Phase 5: Combine segments
        self.logger.info("Combining audio segments into complete track")
        combined_audio = await self._combine_audio_segments(enhanced_segments, audio_settings)
        
        # Phase 6: Final audio optimization
        self.logger.info("Applying final audio optimization")
        optimized_audio = await self._optimize_final_audio(combined_audio, audio_settings)
        
        # Phase 7: Quality assessment
        self.logger.info("Assessing audio quality metrics")
        quality_metrics = await self._assess_audio_quality(optimized_audio)
        
        # Phase 8: Generate final audio object
        self.logger.info("Finalizing audio generation")
        generated_audio = await self._finalize_audio_generation(
            voice_script, optimized_audio, audio_settings, quality_metrics
        )
        
        # Update metrics
        self.metrics.tokens_used = 0  # Audio generation doesn't use text tokens
        self.metrics.api_calls_made = len(prepared_segments)
        self.metrics.estimated_cost = self._calculate_audio_generation_cost(voice_script.word_count)
        
        self.logger.info(f"Audio generation complete: {generated_audio.total_duration:.1f}s")
        return generated_audio
    
    async def _configure_audio_settings(self, voice_script: VoiceScript) -> AudioSettings:
        """Configure audio generation settings based on script requirements."""
        await asyncio.sleep(0.2)
        
        # Extract voice preferences from script
        voice_settings = voice_script.voice_settings
        
        # Select optimal voice model
        voice_model = self._select_voice_model(voice_settings)
        
        return AudioSettings(
            voice_model=voice_model,
            sample_rate=22050,  # High quality for meditation
            bit_depth=16,
            format=AudioFormat.MP3,
            quality="high",
            normalize_audio=True,
            add_fade_in=True,
            add_fade_out=True,
            compression_ratio=2.0,
            noise_reduction=True
        )
    
    def _select_voice_model(self, voice_settings: Dict[str, Any]) -> VoiceModel:
        """Select optimal voice model based on script requirements."""
        voice_style = voice_settings.get("style", "calm_therapeutic")
        
        voice_mapping = {
            "calm_therapeutic": VoiceModel.SARAH_CALM,
            "gentle_guidance": VoiceModel.DAVID_GENTLE,
            "soothing_presence": VoiceModel.MARIA_SOOTHING,
            "mindful_instruction": VoiceModel.ALEX_MINDFUL
        }
        
        return voice_mapping.get(voice_style, VoiceModel.SARAH_CALM)
    
    async def _prepare_segments_for_synthesis(self, voice_script: VoiceScript) -> List[Dict[str, Any]]:
        """Prepare script segments for voice synthesis."""
        await asyncio.sleep(0.3)
        
        prepared_segments = []
        
        for i, segment in enumerate(voice_script.segments):
            # Extract clean text from segment
            text = self._clean_text_for_synthesis(segment.text)
            
            # Prepare SSML if enhanced version exists
            ssml_text = self._extract_segment_ssml(voice_script.ssml_enhanced, i)
            
            prepared_segment = {
                "segment_id": f"segment_{i:03d}",
                "text": text,
                "ssml": ssml_text,
                "voice_notes": segment.voice_notes,
                "duration_target": segment.duration_seconds,
                "section_type": segment.section.value
            }
            
            prepared_segments.append(prepared_segment)
        
        return prepared_segments
    
    def _clean_text_for_synthesis(self, text: str) -> str:
        """Clean text for optimal voice synthesis."""
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Handle special characters
        text = text.replace("...", "…")
        text = text.replace("--", "—")
        
        # Ensure proper punctuation for pauses
        text = text.replace("…", "… ")
        text = text.replace(".", ". ")
        text = text.replace(",", ", ")
        
        return text.strip()
    
    def _extract_segment_ssml(self, full_ssml: str, segment_index: int) -> str:
        """Extract SSML for specific segment from full SSML."""
        # Mock implementation - in production, would properly parse SSML
        return f'<prosody rate="slow">{full_ssml[segment_index*100:(segment_index+1)*100]}</prosody>'
    
    async def _generate_audio_segments(
        self, 
        prepared_segments: List[Dict[str, Any]], 
        settings: AudioSettings
    ) -> List[AudioSegment]:
        """Generate audio segments using voice synthesis."""
        await asyncio.sleep(2.0)  # Simulate API processing time
        
        audio_segments = []
        
        for segment_data in prepared_segments:
            # Simulate API call to voice synthesis service
            audio_file_path = await self._synthesize_segment_audio(segment_data, settings)
            
            # Create audio segment metadata
            audio_segment = AudioSegment(
                segment_id=segment_data["segment_id"],
                start_time=0.0,  # Will be calculated during combination
                duration=segment_data["duration_target"],
                file_path=audio_file_path,
                audio_quality_score=0.0,  # Will be assessed later
                processing_notes=[]
            )
            
            audio_segments.append(audio_segment)
        
        return audio_segments
    
    async def _synthesize_segment_audio(
        self, 
        segment_data: Dict[str, Any], 
        settings: AudioSettings
    ) -> str:
        """Synthesize audio for individual segment."""
        await asyncio.sleep(0.3)  # Simulate API call
        
        # Mock implementation - in production, would make actual API call
        segment_file = self.temp_dir / f"{segment_data['segment_id']}.{settings.format.value}"
        
        # Simulate voice synthesis API call
        synthesis_params = {
            "text": segment_data["text"],
            "voice_id": settings.voice_model.value,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.85,
                "similarity_boost": 0.75,
                "style": 0.5,
                "use_speaker_boost": True
            },
            "pronunciation_dictionary_locators": [],
            "seed": None,
            "previous_text": None,
            "next_text": None,
            "previous_request_ids": [],
            "next_request_ids": []
        }
        
        # Mock file creation
        with open(segment_file, 'wb') as f:
            f.write(b'mock_audio_data')
        
        return str(segment_file)
    
    async def _enhance_audio_quality(
        self, 
        audio_segments: List[AudioSegment], 
        settings: AudioSettings
    ) -> List[AudioSegment]:
        """Enhance audio quality through post-processing."""
        await asyncio.sleep(1.0)  # Simulate audio processing
        
        enhanced_segments = []
        
        for segment in audio_segments:
            # Apply noise reduction
            if settings.noise_reduction:
                segment.processing_notes.append("noise_reduction_applied")
            
            # Normalize audio levels
            if settings.normalize_audio:
                segment.processing_notes.append("audio_normalized")
            
            # Apply compression for consistent levels
            segment.processing_notes.append(f"compression_applied_{settings.compression_ratio}")
            
            # Assess audio quality
            segment.audio_quality_score = await self._assess_segment_quality(segment)
            
            enhanced_segments.append(segment)
        
        return enhanced_segments
    
    async def _assess_segment_quality(self, segment: AudioSegment) -> float:
        """Assess audio quality for individual segment."""
        await asyncio.sleep(0.1)
        
        # Mock quality assessment
        quality_factors = {
            "clarity": 0.92,
            "naturalness": 0.89,
            "consistency": 0.94,
            "noise_level": 0.96,
            "timing": 0.91
        }
        
        return sum(quality_factors.values()) / len(quality_factors)
    
    async def _combine_audio_segments(
        self, 
        segments: List[AudioSegment], 
        settings: AudioSettings
    ) -> str:
        """Combine individual segments into complete audio track."""
        await asyncio.sleep(0.8)  # Simulate audio combination
        
        combined_file = self.temp_dir / f"combined_meditation.{settings.format.value}"
        
        # Update segment timing information
        current_time = 0.0
        for segment in segments:
            segment.start_time = current_time
            current_time += segment.duration
        
        # Mock audio combination process
        with open(combined_file, 'wb') as f:
            f.write(b'mock_combined_audio_data')
        
        return str(combined_file)
    
    async def _optimize_final_audio(self, audio_file: str, settings: AudioSettings) -> str:
        """Apply final optimizations to combined audio."""
        await asyncio.sleep(0.5)  # Simulate final processing
        
        optimized_file = self.temp_dir / f"optimized_meditation.{settings.format.value}"
        
        # Apply fade in/out
        if settings.add_fade_in or settings.add_fade_out:
            # Mock fade processing
            pass
        
        # Final quality optimization
        # Mock optimization process
        with open(optimized_file, 'wb') as f:
            f.write(b'mock_optimized_audio_data')
        
        return str(optimized_file)
    
    async def _assess_audio_quality(self, audio_file: str) -> Dict[str, float]:
        """Assess overall audio quality metrics."""
        await asyncio.sleep(0.4)
        
        return {
            "overall_quality": 0.91,
            "voice_clarity": 0.93,
            "audio_consistency": 0.89,
            "background_noise": 0.96,
            "dynamic_range": 0.88,
            "frequency_response": 0.92,
            "timing_accuracy": 0.94,
            "therapeutic_effectiveness": 0.90
        }
    
    async def _finalize_audio_generation(
        self, 
        voice_script: VoiceScript, 
        audio_file: str, 
        settings: AudioSettings,
        quality_metrics: Dict[str, float]
    ) -> GeneratedAudio:
        """Finalize audio generation with complete metadata."""
        await asyncio.sleep(0.2)
        
        # Calculate file size
        file_size_mb = Path(audio_file).stat().st_size / (1024 * 1024)
        
        # Generate audio fingerprint
        audio_fingerprint = self._generate_audio_fingerprint(audio_file)
        
        # Calculate voice consistency score
        voice_consistency_score = quality_metrics.get("audio_consistency", 0.0)
        
        return GeneratedAudio(
            title=voice_script.title,
            total_duration=voice_script.estimated_speech_time * 60,
            audio_file_path=audio_file,
            segments=voice_script.segments,  # Convert to AudioSegment if needed
            settings=settings,
            quality_metrics=quality_metrics,
            file_size_mb=file_size_mb,
            processing_time_seconds=self.metrics.execution_time_seconds,
            voice_consistency_score=voice_consistency_score,
            audio_fingerprint=audio_fingerprint
        )
    
    def _generate_audio_fingerprint(self, audio_file: str) -> str:
        """Generate unique fingerprint for audio file."""
        with open(audio_file, 'rb') as f:
            audio_data = f.read()
        
        return hashlib.md5(audio_data).hexdigest()
    
    def _calculate_audio_generation_cost(self, word_count: int) -> float:
        """Calculate cost for audio generation based on word count."""
        # Mock pricing: $0.30 per 1000 characters (typical TTS pricing)
        character_count = word_count * 5  # Approximate characters per word
        return (character_count / 1000) * 0.30
    
    async def validate_output(self, output: GeneratedAudio) -> bool:
        """
        Validate generated audio quality and completeness.
        
        Args:
            output: GeneratedAudio to validate
            
        Returns:
            True if audio meets quality standards
        """
        validation_checks = [
            output.audio_file_path is not None,
            Path(output.audio_file_path).exists(),
            output.total_duration > 0,
            output.file_size_mb > 0,
            output.quality_metrics.get("overall_quality", 0) > 0.8,
            output.voice_consistency_score > 0.7,
            len(output.audio_fingerprint) > 0,
            output.settings is not None
        ]
        
        validation_passed = all(validation_checks)
        
        if not validation_passed:
            self.logger.warning("Audio generation validation failed")
            
        return validation_passed
    
    def get_required_context_keys(self) -> List[str]:
        """Define required context keys for audio generation."""
        return ["voice_script"]
    
    def get_output_key(self) -> str:
        """Define output key for generated audio."""
        return "generated_audio"
    
    def cleanup_temp_files(self):
        """Clean up temporary files created during processing."""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
            self.logger.info("Temporary files cleaned up")
        except Exception as e:
            self.logger.warning(f"Error cleaning up temp files: {e}")
    
    def __del__(self):
        """Cleanup on object destruction."""
        self.cleanup_temp_files()