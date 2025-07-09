"""
Agent 10: Video Assembly Agent

This agent handles the complete video assembly process, combining generated audio,
background visuals, and music into a polished meditation video ready for
distribution. It manages video encoding, quality optimization, and format conversion.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
from pathlib import Path
import tempfile
from datetime import datetime

from .base import BaseAgent, AgentConfig
from .agent_7_audio_generation import GeneratedAudio


class VideoFormat(Enum):
    """Supported video formats for output."""
    MP4 = "mp4"
    MOV = "mov"
    AVI = "avi"
    WEBM = "webm"


class VideoQuality(Enum):
    """Video quality presets."""
    LOW = "480p"
    MEDIUM = "720p"
    HIGH = "1080p"
    ULTRA = "4K"


class TransitionType(Enum):
    """Available transition types for video segments."""
    FADE = "fade"
    CROSSFADE = "crossfade"
    DISSOLVE = "dissolve"
    NONE = "none"


@dataclass
class VideoAsset:
    """Individual video asset for composition."""
    asset_id: str
    file_path: str
    asset_type: str  # "background", "overlay", "text", "animation"
    start_time: float
    duration: float
    layer_index: int
    opacity: float = 1.0
    effects: List[str] = field(default_factory=list)
    position: Tuple[int, int] = (0, 0)
    scale: float = 1.0


@dataclass
class VideoSettings:
    """Configuration for video assembly."""
    format: VideoFormat = VideoFormat.MP4
    quality: VideoQuality = VideoQuality.HIGH
    frame_rate: int = 30
    bitrate: str = "2000k"
    codec: str = "h264"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    background_color: str = "#000000"
    add_metadata: bool = True
    optimize_for_web: bool = True


@dataclass
class VideoComposition:
    """Complete video composition structure."""
    title: str
    duration: float
    dimensions: Tuple[int, int]
    assets: List[VideoAsset]
    audio_tracks: List[Dict[str, Any]]
    transitions: List[Dict[str, Any]]
    effects: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class AssembledVideo:
    """Final assembled video with metadata."""
    title: str
    file_path: str
    duration: float
    file_size_mb: float
    format: VideoFormat
    quality: VideoQuality
    dimensions: Tuple[int, int]
    composition: VideoComposition
    rendering_time: float
    quality_metrics: Dict[str, float]
    thumbnail_path: Optional[str] = None
    preview_path: Optional[str] = None


class VideoAssemblyAgent(BaseAgent):
    """
    Agent 10: Video Assembly and Composition
    
    This agent handles the complete video assembly pipeline, combining audio,
    visuals, and effects into a polished meditation video. It manages video
    encoding, quality optimization, and format conversion.
    
    Capabilities:
    - Multi-layer video composition
    - Audio-visual synchronization
    - Background image/video processing
    - Text overlay and animation
    - Quality optimization and encoding
    - Format conversion and compression
    - Thumbnail generation
    - Preview creation
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Video Assembly Agent."""
        if config is None:
            config = AgentConfig(
                name="video_assembly",
                phase=4,
                timeout_seconds=600,  # Video processing can take longer
                max_retries=2,
                required_capabilities=[
                    "video_composition",
                    "audio_video_sync",
                    "video_encoding",
                    "quality_optimization",
                    "format_conversion"
                ]
            )
        
        super().__init__(config)
        self.temp_dir = Path(tempfile.mkdtemp())
        self.ffmpeg_path = "ffmpeg"  # Assuming ffmpeg is in PATH
        self.supported_formats = [VideoFormat.MP4, VideoFormat.MOV, VideoFormat.WEBM]
        
    async def execute(self, context: Dict[str, Any]) -> AssembledVideo:
        """
        Assemble complete meditation video from components.
        
        Args:
            context: Pipeline context containing audio and visual assets
            
        Returns:
            AssembledVideo with complete video file and metadata
        """
        self.logger.info("Starting video assembly process")
        
        # Extract required components
        generated_audio = context.get("generated_audio")
        background_assets = context.get("background_assets", {})
        
        if not generated_audio:
            raise ValueError("Generated audio required for video assembly")
        
        # Phase 1: Configure video settings
        self.logger.info("Configuring video assembly settings")
        video_settings = await self._configure_video_settings(generated_audio, background_assets)
        
        # Phase 2: Prepare video composition
        self.logger.info("Preparing video composition structure")
        composition = await self._prepare_video_composition(generated_audio, background_assets, video_settings)
        
        # Phase 3: Process background assets
        self.logger.info("Processing background visual assets")
        processed_assets = await self._process_background_assets(composition, video_settings)
        
        # Phase 4: Synchronize audio and video
        self.logger.info("Synchronizing audio and video components")
        synchronized_composition = await self._synchronize_audio_video(composition, generated_audio)
        
        # Phase 5: Apply effects and transitions
        self.logger.info("Applying visual effects and transitions")
        enhanced_composition = await self._apply_effects_and_transitions(synchronized_composition)
        
        # Phase 6: Render video
        self.logger.info("Rendering final video composition")
        rendered_video = await self._render_video(enhanced_composition, video_settings)
        
        # Phase 7: Optimize and encode
        self.logger.info("Optimizing and encoding video")
        optimized_video = await self._optimize_and_encode(rendered_video, video_settings)
        
        # Phase 8: Generate supplementary assets
        self.logger.info("Generating thumbnail and preview")
        thumbnail_path = await self._generate_thumbnail(optimized_video)
        preview_path = await self._generate_preview(optimized_video)
        
        # Phase 9: Quality assessment
        self.logger.info("Assessing video quality")
        quality_metrics = await self._assess_video_quality(optimized_video)
        
        # Phase 10: Finalize video assembly
        self.logger.info("Finalizing video assembly")
        assembled_video = await self._finalize_video_assembly(
            optimized_video, composition, video_settings, quality_metrics,
            thumbnail_path, preview_path
        )
        
        # Update metrics
        self.metrics.tokens_used = 0  # Video assembly doesn't use text tokens
        self.metrics.api_calls_made = 0
        self.metrics.estimated_cost = self._calculate_video_processing_cost(generated_audio.total_duration)
        
        self.logger.info(f"Video assembly complete: {assembled_video.duration:.1f}s, {assembled_video.file_size_mb:.1f}MB")
        return assembled_video
    
    async def _configure_video_settings(
        self, 
        audio: GeneratedAudio, 
        background_assets: Dict[str, Any]
    ) -> VideoSettings:
        """Configure video settings based on audio and available assets."""
        await asyncio.sleep(0.2)
        
        # Determine optimal settings based on content
        quality = VideoQuality.HIGH  # Default to high quality
        
        # Adjust based on audio duration (longer videos might need compression)
        if audio.total_duration > 1800:  # 30 minutes
            quality = VideoQuality.MEDIUM
        
        return VideoSettings(
            format=VideoFormat.MP4,
            quality=quality,
            frame_rate=30,
            bitrate="2000k",
            codec="h264",
            audio_codec="aac",
            audio_bitrate="192k",
            background_color="#000000",
            add_metadata=True,
            optimize_for_web=True
        )
    
    async def _prepare_video_composition(
        self, 
        audio: GeneratedAudio, 
        background_assets: Dict[str, Any],
        settings: VideoSettings
    ) -> VideoComposition:
        """Prepare complete video composition structure."""
        await asyncio.sleep(0.5)
        
        # Determine video dimensions based on quality
        dimensions = self._get_dimensions_for_quality(settings.quality)
        
        # Create base composition
        composition = VideoComposition(
            title=audio.title,
            duration=audio.total_duration,
            dimensions=dimensions,
            assets=[],
            audio_tracks=[],
            transitions=[],
            effects=[],
            metadata={}
        )
        
        # Add background asset
        background_asset = await self._create_background_asset(background_assets, composition)
        composition.assets.append(background_asset)
        
        # Add audio track
        audio_track = {
            "track_id": "main_audio",
            "file_path": audio.audio_file_path,
            "start_time": 0.0,
            "duration": audio.total_duration,
            "volume": 1.0,
            "fade_in": 2.0,
            "fade_out": 3.0
        }
        composition.audio_tracks.append(audio_track)
        
        # Add background music if available
        if "background_music" in background_assets:
            music_track = await self._create_background_music_track(background_assets["background_music"], composition)
            composition.audio_tracks.append(music_track)
        
        # Add text overlays if needed
        text_overlays = await self._create_text_overlays(audio, composition)
        composition.assets.extend(text_overlays)
        
        return composition
    
    def _get_dimensions_for_quality(self, quality: VideoQuality) -> Tuple[int, int]:
        """Get video dimensions based on quality setting."""
        quality_dimensions = {
            VideoQuality.LOW: (854, 480),
            VideoQuality.MEDIUM: (1280, 720),
            VideoQuality.HIGH: (1920, 1080),
            VideoQuality.ULTRA: (3840, 2160)
        }
        return quality_dimensions.get(quality, (1920, 1080))
    
    async def _create_background_asset(
        self, 
        background_assets: Dict[str, Any], 
        composition: VideoComposition
    ) -> VideoAsset:
        """Create background video asset."""
        await asyncio.sleep(0.3)
        
        # Check if we have a background image or video
        if "background_image" in background_assets:
            background_path = background_assets["background_image"]["file_path"]
            asset_type = "background_image"
        elif "background_video" in background_assets:
            background_path = background_assets["background_video"]["file_path"]
            asset_type = "background_video"
        else:
            # Create a simple gradient background
            background_path = await self._create_gradient_background(composition)
            asset_type = "generated_background"
        
        return VideoAsset(
            asset_id="background_01",
            file_path=background_path,
            asset_type=asset_type,
            start_time=0.0,
            duration=composition.duration,
            layer_index=0,
            opacity=1.0,
            effects=["slow_zoom", "subtle_movement"],
            position=(0, 0),
            scale=1.0
        )
    
    async def _create_gradient_background(self, composition: VideoComposition) -> str:
        """Create a gradient background for meditation video."""
        await asyncio.sleep(0.2)
        
        gradient_file = self.temp_dir / "gradient_background.png"
        
        # Mock gradient generation - in production, would use actual image processing
        # Create a soothing gradient (deep blue to purple)
        gradient_data = b'mock_gradient_image_data'
        
        with open(gradient_file, 'wb') as f:
            f.write(gradient_data)
        
        return str(gradient_file)
    
    async def _create_background_music_track(
        self, 
        music_asset: Dict[str, Any], 
        composition: VideoComposition
    ) -> Dict[str, Any]:
        """Create background music track for video."""
        await asyncio.sleep(0.1)
        
        return {
            "track_id": "background_music",
            "file_path": music_asset["file_path"],
            "start_time": 0.0,
            "duration": composition.duration,
            "volume": 0.3,  # Lower volume for background
            "fade_in": 5.0,
            "fade_out": 5.0,
            "loop": True if music_asset.get("duration", 0) < composition.duration else False
        }
    
    async def _create_text_overlays(
        self, 
        audio: GeneratedAudio, 
        composition: VideoComposition
    ) -> List[VideoAsset]:
        """Create text overlays for video."""
        await asyncio.sleep(0.3)
        
        overlays = []
        
        # Title overlay at beginning
        title_overlay = VideoAsset(
            asset_id="title_overlay",
            file_path=await self._create_title_text(audio.title, composition),
            asset_type="text_overlay",
            start_time=0.0,
            duration=5.0,
            layer_index=10,
            opacity=0.8,
            effects=["fade_in", "fade_out"],
            position=(composition.dimensions[0]//2, composition.dimensions[1]//4),
            scale=1.0
        )
        overlays.append(title_overlay)
        
        # Subtle progress indicator
        progress_overlay = VideoAsset(
            asset_id="progress_indicator",
            file_path=await self._create_progress_indicator(composition),
            asset_type="progress_overlay",
            start_time=0.0,
            duration=composition.duration,
            layer_index=5,
            opacity=0.3,
            effects=["progress_animation"],
            position=(composition.dimensions[0]//2, composition.dimensions[1] - 50),
            scale=1.0
        )
        overlays.append(progress_overlay)
        
        return overlays
    
    async def _create_title_text(self, title: str, composition: VideoComposition) -> str:
        """Create title text overlay."""
        await asyncio.sleep(0.1)
        
        title_file = self.temp_dir / "title_overlay.png"
        
        # Mock title generation - in production, would use actual text rendering
        with open(title_file, 'wb') as f:
            f.write(b'mock_title_overlay_data')
        
        return str(title_file)
    
    async def _create_progress_indicator(self, composition: VideoComposition) -> str:
        """Create progress indicator overlay."""
        await asyncio.sleep(0.1)
        
        progress_file = self.temp_dir / "progress_indicator.png"
        
        # Mock progress indicator - in production, would create actual indicator
        with open(progress_file, 'wb') as f:
            f.write(b'mock_progress_indicator_data')
        
        return str(progress_file)
    
    async def _process_background_assets(
        self, 
        composition: VideoComposition, 
        settings: VideoSettings
    ) -> List[VideoAsset]:
        """Process background assets for optimal video quality."""
        await asyncio.sleep(0.8)
        
        processed_assets = []
        
        for asset in composition.assets:
            if asset.asset_type in ["background_image", "background_video"]:
                # Resize and optimize for target dimensions
                processed_path = await self._resize_and_optimize_asset(asset, settings)
                asset.file_path = processed_path
                
                # Add subtle animation effects
                if asset.asset_type == "background_image":
                    asset.effects.extend(["ken_burns", "breathing_effect"])
                
            processed_assets.append(asset)
        
        return processed_assets
    
    async def _resize_and_optimize_asset(self, asset: VideoAsset, settings: VideoSettings) -> str:
        """Resize and optimize individual asset."""
        await asyncio.sleep(0.2)
        
        optimized_file = self.temp_dir / f"optimized_{asset.asset_id}.png"
        
        # Mock asset optimization - in production, would use actual image/video processing
        with open(optimized_file, 'wb') as f:
            f.write(b'mock_optimized_asset_data')
        
        return str(optimized_file)
    
    async def _synchronize_audio_video(
        self, 
        composition: VideoComposition, 
        audio: GeneratedAudio
    ) -> VideoComposition:
        """Synchronize audio and video components."""
        await asyncio.sleep(0.4)
        
        # Ensure video duration matches audio duration
        composition.duration = audio.total_duration
        
        # Adjust background assets to match duration
        for asset in composition.assets:
            if asset.asset_type.startswith("background"):
                asset.duration = composition.duration
        
        # Sync audio tracks
        for track in composition.audio_tracks:
            if track["track_id"] == "main_audio":
                track["duration"] = audio.total_duration
        
        return composition
    
    async def _apply_effects_and_transitions(self, composition: VideoComposition) -> VideoComposition:
        """Apply visual effects and transitions."""
        await asyncio.sleep(0.6)
        
        # Add smooth transitions between segments
        transitions = [
            {
                "type": TransitionType.FADE,
                "start_time": 0.0,
                "duration": 2.0,
                "properties": {"direction": "in"}
            },
            {
                "type": TransitionType.FADE,
                "start_time": composition.duration - 3.0,
                "duration": 3.0,
                "properties": {"direction": "out"}
            }
        ]
        
        composition.transitions = transitions
        
        # Add global effects
        effects = [
            {
                "type": "color_grading",
                "properties": {"warmth": 0.1, "saturation": 0.8, "contrast": 0.05}
            },
            {
                "type": "subtle_vignette",
                "properties": {"strength": 0.2, "feather": 0.8}
            }
        ]
        
        composition.effects = effects
        
        return composition
    
    async def _render_video(self, composition: VideoComposition, settings: VideoSettings) -> str:
        """Render the video composition."""
        await asyncio.sleep(3.0)  # Simulate rendering time
        
        rendered_file = self.temp_dir / f"rendered_{composition.title.replace(' ', '_')}.{settings.format.value}"
        
        # Mock video rendering - in production, would use FFmpeg
        render_command = self._build_ffmpeg_command(composition, settings, str(rendered_file))
        
        # Mock execution
        self.logger.info(f"Rendering with command: {render_command}")
        
        # Create mock rendered file
        with open(rendered_file, 'wb') as f:
            f.write(b'mock_rendered_video_data')
        
        return str(rendered_file)
    
    def _build_ffmpeg_command(
        self, 
        composition: VideoComposition, 
        settings: VideoSettings, 
        output_path: str
    ) -> str:
        """Build FFmpeg command for video rendering."""
        cmd_parts = [
            self.ffmpeg_path,
            "-y",  # Overwrite output file
        ]
        
        # Add input files
        for asset in composition.assets:
            cmd_parts.extend(["-i", asset.file_path])
        
        for track in composition.audio_tracks:
            cmd_parts.extend(["-i", track["file_path"]])
        
        # Add filtering and composition
        cmd_parts.extend([
            "-filter_complex",
            self._build_filter_complex(composition, settings),
            "-map", "[vout]",
            "-map", "[aout]",
            "-c:v", settings.codec,
            "-c:a", settings.audio_codec,
            "-b:v", settings.bitrate,
            "-b:a", settings.audio_bitrate,
            "-r", str(settings.frame_rate),
            output_path
        ])
        
        return " ".join(cmd_parts)
    
    def _build_filter_complex(self, composition: VideoComposition, settings: VideoSettings) -> str:
        """Build FFmpeg filter complex for video composition."""
        # Mock filter complex - in production, would build actual FFmpeg filters
        return f"[0:v]scale={composition.dimensions[0]}:{composition.dimensions[1]}[vout];[1:a][2:a]amix=inputs=2[aout]"
    
    async def _optimize_and_encode(self, video_path: str, settings: VideoSettings) -> str:
        """Optimize and encode final video."""
        await asyncio.sleep(1.0)
        
        optimized_file = self.temp_dir / f"optimized_{Path(video_path).stem}.{settings.format.value}"
        
        # Mock optimization - in production, would use actual video optimization
        with open(optimized_file, 'wb') as f:
            f.write(b'mock_optimized_video_data')
        
        return str(optimized_file)
    
    async def _generate_thumbnail(self, video_path: str) -> str:
        """Generate thumbnail from video."""
        await asyncio.sleep(0.3)
        
        thumbnail_file = self.temp_dir / f"thumbnail_{Path(video_path).stem}.jpg"
        
        # Mock thumbnail generation - in production, would extract frame from video
        with open(thumbnail_file, 'wb') as f:
            f.write(b'mock_thumbnail_data')
        
        return str(thumbnail_file)
    
    async def _generate_preview(self, video_path: str) -> str:
        """Generate preview clip from video."""
        await asyncio.sleep(0.5)
        
        preview_file = self.temp_dir / f"preview_{Path(video_path).stem}.mp4"
        
        # Mock preview generation - in production, would create short preview
        with open(preview_file, 'wb') as f:
            f.write(b'mock_preview_data')
        
        return str(preview_file)
    
    async def _assess_video_quality(self, video_path: str) -> Dict[str, float]:
        """Assess video quality metrics."""
        await asyncio.sleep(0.5)
        
        return {
            "visual_quality": 0.89,
            "audio_quality": 0.92,
            "sync_accuracy": 0.96,
            "compression_efficiency": 0.85,
            "color_accuracy": 0.90,
            "motion_smoothness": 0.88,
            "overall_production_value": 0.91
        }
    
    async def _finalize_video_assembly(
        self, 
        video_path: str, 
        composition: VideoComposition,
        settings: VideoSettings,
        quality_metrics: Dict[str, float],
        thumbnail_path: str,
        preview_path: str
    ) -> AssembledVideo:
        """Finalize video assembly with complete metadata."""
        await asyncio.sleep(0.2)
        
        # Calculate file size
        file_size_mb = Path(video_path).stat().st_size / (1024 * 1024)
        
        # Add metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "generator": "Enhanced Archibald v5.1",
            "composition_layers": len(composition.assets),
            "audio_tracks": len(composition.audio_tracks),
            "effects_applied": len(composition.effects),
            "rendering_settings": settings.__dict__
        }
        
        return AssembledVideo(
            title=composition.title,
            file_path=video_path,
            duration=composition.duration,
            file_size_mb=file_size_mb,
            format=settings.format,
            quality=settings.quality,
            dimensions=composition.dimensions,
            composition=composition,
            rendering_time=self.metrics.execution_time_seconds,
            quality_metrics=quality_metrics,
            thumbnail_path=thumbnail_path,
            preview_path=preview_path
        )
    
    def _calculate_video_processing_cost(self, duration: float) -> float:
        """Calculate cost for video processing based on duration."""
        # Mock pricing: $0.05 per minute of video processing
        return (duration / 60) * 0.05
    
    async def validate_output(self, output: AssembledVideo) -> bool:
        """
        Validate assembled video quality and completeness.
        
        Args:
            output: AssembledVideo to validate
            
        Returns:
            True if video meets quality standards
        """
        validation_checks = [
            output.file_path is not None,
            Path(output.file_path).exists(),
            output.duration > 0,
            output.file_size_mb > 0,
            output.quality_metrics.get("overall_production_value", 0) > 0.8,
            output.dimensions[0] > 0 and output.dimensions[1] > 0,
            output.thumbnail_path is not None,
            Path(output.thumbnail_path).exists() if output.thumbnail_path else True,
            len(output.composition.assets) > 0,
            len(output.composition.audio_tracks) > 0
        ]
        
        validation_passed = all(validation_checks)
        
        if not validation_passed:
            self.logger.warning("Video assembly validation failed")
            
        return validation_passed
    
    def get_required_context_keys(self) -> List[str]:
        """Define required context keys for video assembly."""
        return ["generated_audio"]
    
    def get_output_key(self) -> str:
        """Define output key for assembled video."""
        return "assembled_video"
    
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