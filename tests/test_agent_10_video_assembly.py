"""
Unit tests for VideoAssemblyAgent.

Tests the video assembly and FFmpeg processing functionality
of the tenth agent in the assembly & optimization phase.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, mock_open
from datetime import datetime, timedelta
from typing import Dict, Any
import os
import tempfile

from agents.agent_10_video_assembly import VideoAssemblyAgent
from orchestrator.pipeline_context import PipelineContext


class TestVideoAssemblyAgent:
    """Test cases for VideoAssemblyAgent."""
    
    @pytest.fixture
    def agent(self):
        """Create VideoAssemblyAgent instance."""
        return VideoAssemblyAgent()
    
    @pytest.fixture
    def context(self, mock_context):
        """Provide test context with media files."""
        mock_context.data.update({
            "audio_file": {
                "file_path": "/tmp/test_audio.mp3",
                "duration": 300.0,
                "format": "mp3",
                "sample_rate": 44100
            },
            "background_music": {
                "file_path": "/tmp/test_music.mp3",
                "duration": 320.0,
                "format": "mp3",
                "volume": 0.3
            },
            "background_image": {
                "file_path": "/tmp/test_image.jpg",
                "resolution": "1920x1080",
                "format": "jpg"
            },
            "video_settings": {
                "resolution": "1920x1080",
                "fps": 30,
                "bitrate": "2M",
                "codec": "h264"
            }
        })
        return mock_context
    
    @pytest.fixture
    def mock_video_file(self, tmp_path):
        """Create mock video file."""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(b"mock_video_data_content")
        return str(video_file)
    
    @pytest.fixture
    def mock_ffmpeg_result(self):
        """Mock FFmpeg execution result."""
        return {
            "returncode": 0,
            "stdout": "Video processing completed successfully",
            "stderr": ""
        }
    
    @pytest.mark.unit
    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.agent_id == "video_assembly"
        assert agent.cost_per_call == 0.0  # Local processing
        assert agent.timeout == 300.0  # 5 minutes for video processing
        assert agent.max_retries == 3
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_execute_success(self, agent, context, mock_video_file, mock_ffmpeg_result):
        """Test successful execution of video assembly."""
        with patch('agents.agent_10_video_assembly.run_ffmpeg') as mock_ffmpeg:
            with patch('agents.agent_10_video_assembly.validate_video_file') as mock_validate:
                with patch('agents.agent_10_video_assembly.get_video_metadata') as mock_metadata:
                    mock_ffmpeg.return_value = mock_ffmpeg_result
                    mock_validate.return_value = True
                    mock_metadata.return_value = {
                        "duration": 300.0,
                        "resolution": "1920x1080",
                        "fps": 30,
                        "bitrate": "2M",
                        "codec": "h264"
                    }
                    
                    with patch('agents.agent_10_video_assembly.generate_output_path') as mock_path:
                        mock_path.return_value = mock_video_file
                        
                        result = await agent.execute(context)
                        
                        assert result["status"] == "success"
                        assert "video_file" in result
                        assert "file_path" in result["video_file"]
                        assert "duration" in result["video_file"]
                        assert "resolution" in result["video_file"]
                        assert "quality_metrics" in result["video_file"]
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_execute_with_ffmpeg_error(self, agent, context):
        """Test execution with FFmpeg error."""
        with patch('agents.agent_10_video_assembly.run_ffmpeg') as mock_ffmpeg:
            mock_ffmpeg.return_value = {
                "returncode": 1,
                "stdout": "",
                "stderr": "FFmpeg error: Invalid input"
            }
            
            with pytest.raises(Exception, match="FFmpeg error"):
                await agent.execute(context)
    
    @pytest.mark.unit
    def test_validate_output_success(self, agent, mock_video_file):
        """Test output validation with valid data."""
        valid_output = {
            "status": "success",
            "video_file": {
                "file_path": mock_video_file,
                "duration": 300.0,
                "resolution": "1920x1080",
                "fps": 30,
                "file_size": 10485760,
                "format": "mp4",
                "quality_metrics": {
                    "video_quality": 0.92,
                    "audio_quality": 0.88,
                    "sync_quality": 0.95
                }
            }
        }
        
        assert agent.validate_output(valid_output)
    
    @pytest.mark.unit
    def test_validate_output_missing_file(self, agent):
        """Test output validation with missing file."""
        invalid_output = {
            "status": "success",
            "video_file": {
                "file_path": "/nonexistent/video.mp4",
                "duration": 300.0,
                "resolution": "1920x1080",
                "fps": 30,
                "file_size": 10485760,
                "format": "mp4",
                "quality_metrics": {
                    "video_quality": 0.92,
                    "audio_quality": 0.88,
                    "sync_quality": 0.95
                }
            }
        }
        
        assert not agent.validate_output(invalid_output)
    
    @pytest.mark.unit
    def test_validate_output_invalid_duration(self, agent, mock_video_file):
        """Test output validation with invalid duration."""
        invalid_output = {
            "status": "success",
            "video_file": {
                "file_path": mock_video_file,
                "duration": -1.0,  # Invalid duration
                "resolution": "1920x1080",
                "fps": 30,
                "file_size": 10485760,
                "format": "mp4",
                "quality_metrics": {
                    "video_quality": 0.92,
                    "audio_quality": 0.88,
                    "sync_quality": 0.95
                }
            }
        }
        
        assert not agent.validate_output(invalid_output)
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_ffmpeg_command_generation(self, agent, context, mock_video_file, mock_ffmpeg_result):
        """Test FFmpeg command generation."""
        with patch('agents.agent_10_video_assembly.run_ffmpeg') as mock_ffmpeg:
            with patch('agents.agent_10_video_assembly.validate_video_file') as mock_validate:
                with patch('agents.agent_10_video_assembly.get_video_metadata') as mock_metadata:
                    with patch('agents.agent_10_video_assembly.generate_output_path') as mock_path:
                        mock_ffmpeg.return_value = mock_ffmpeg_result
                        mock_validate.return_value = True
                        mock_metadata.return_value = {"duration": 300.0, "resolution": "1920x1080"}
                        mock_path.return_value = mock_video_file
                        
                        await agent.execute(context)
                        
                        # Check FFmpeg was called with correct parameters
                        assert mock_ffmpeg.called
                        call_args = mock_ffmpeg.call_args[0][0]
                        
                        # Should contain input files
                        assert any("-i" in arg for arg in call_args)
                        
                        # Should contain output settings
                        assert any("-c:v" in arg for arg in call_args)
                        assert any("-c:a" in arg for arg in call_args)
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_audio_mixing(self, agent, context, mock_video_file, mock_ffmpeg_result):
        """Test audio mixing functionality."""
        with patch('agents.agent_10_video_assembly.run_ffmpeg') as mock_ffmpeg:
            with patch('agents.agent_10_video_assembly.validate_video_file') as mock_validate:
                with patch('agents.agent_10_video_assembly.get_video_metadata') as mock_metadata:
                    with patch('agents.agent_10_video_assembly.generate_output_path') as mock_path:
                        mock_ffmpeg.return_value = mock_ffmpeg_result
                        mock_validate.return_value = True
                        mock_metadata.return_value = {"duration": 300.0, "resolution": "1920x1080"}
                        mock_path.return_value = mock_video_file
                        
                        await agent.execute(context)
                        
                        # Check audio mixing parameters
                        call_args = mock_ffmpeg.call_args[0][0]
                        command_str = " ".join(call_args)
                        
                        # Should include audio mixing filters
                        assert "-filter_complex" in command_str or "-af" in command_str
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_video_quality_settings(self, agent, context, mock_video_file, mock_ffmpeg_result):
        """Test video quality settings."""
        quality_settings = [
            {"resolution": "1920x1080", "bitrate": "2M", "fps": 30},
            {"resolution": "1280x720", "bitrate": "1M", "fps": 24},
            {"resolution": "854x480", "bitrate": "500k", "fps": 30}
        ]
        
        for settings in quality_settings:
            context.data["video_settings"].update(settings)
            
            with patch('agents.agent_10_video_assembly.run_ffmpeg') as mock_ffmpeg:
                with patch('agents.agent_10_video_assembly.validate_video_file') as mock_validate:
                    with patch('agents.agent_10_video_assembly.get_video_metadata') as mock_metadata:
                        with patch('agents.agent_10_video_assembly.generate_output_path') as mock_path:
                            mock_ffmpeg.return_value = mock_ffmpeg_result
                            mock_validate.return_value = True
                            mock_metadata.return_value = {"duration": 300.0, "resolution": settings["resolution"]}
                            mock_path.return_value = mock_video_file
                            
                            result = await agent.execute(context)
                            
                            assert result["status"] == "success"
                            assert result["video_file"]["resolution"] == settings["resolution"]
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_background_music_integration(self, agent, context, mock_video_file, mock_ffmpeg_result):
        """Test background music integration."""
        with patch('agents.agent_10_video_assembly.run_ffmpeg') as mock_ffmpeg:
            with patch('agents.agent_10_video_assembly.validate_video_file') as mock_validate:
                with patch('agents.agent_10_video_assembly.get_video_metadata') as mock_metadata:
                    with patch('agents.agent_10_video_assembly.generate_output_path') as mock_path:
                        mock_ffmpeg.return_value = mock_ffmpeg_result
                        mock_validate.return_value = True
                        mock_metadata.return_value = {"duration": 300.0, "resolution": "1920x1080"}
                        mock_path.return_value = mock_video_file
                        
                        await agent.execute(context)
                        
                        # Check background music integration
                        call_args = mock_ffmpeg.call_args[0][0]
                        command_str = " ".join(call_args)
                        
                        # Should include music file and volume settings
                        assert context.data["background_music"]["file_path"] in command_str
                        assert "volume" in command_str or "loudness" in command_str
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_image_to_video_conversion(self, agent, context, mock_video_file, mock_ffmpeg_result):
        """Test image to video conversion."""
        with patch('agents.agent_10_video_assembly.run_ffmpeg') as mock_ffmpeg:
            with patch('agents.agent_10_video_assembly.validate_video_file') as mock_validate:
                with patch('agents.agent_10_video_assembly.get_video_metadata') as mock_metadata:
                    with patch('agents.agent_10_video_assembly.generate_output_path') as mock_path:
                        mock_ffmpeg.return_value = mock_ffmpeg_result
                        mock_validate.return_value = True
                        mock_metadata.return_value = {"duration": 300.0, "resolution": "1920x1080"}
                        mock_path.return_value = mock_video_file
                        
                        await agent.execute(context)
                        
                        # Check image to video conversion
                        call_args = mock_ffmpeg.call_args[0][0]
                        command_str = " ".join(call_args)
                        
                        # Should include image file and loop settings
                        assert context.data["background_image"]["file_path"] in command_str
                        assert "-loop" in command_str
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_duration_synchronization(self, agent, context, mock_video_file, mock_ffmpeg_result):
        """Test duration synchronization."""
        # Set different durations for audio and music
        context.data["audio_file"]["duration"] = 300.0
        context.data["background_music"]["duration"] = 320.0
        
        with patch('agents.agent_10_video_assembly.run_ffmpeg') as mock_ffmpeg:
            with patch('agents.agent_10_video_assembly.validate_video_file') as mock_validate:
                with patch('agents.agent_10_video_assembly.get_video_metadata') as mock_metadata:
                    with patch('agents.agent_10_video_assembly.generate_output_path') as mock_path:
                        mock_ffmpeg.return_value = mock_ffmpeg_result
                        mock_validate.return_value = True
                        mock_metadata.return_value = {"duration": 300.0, "resolution": "1920x1080"}
                        mock_path.return_value = mock_video_file
                        
                        result = await agent.execute(context)
                        
                        # Should sync to audio duration
                        assert result["video_file"]["duration"] == 300.0
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_quality_metrics_calculation(self, agent, context, mock_video_file, mock_ffmpeg_result):
        """Test quality metrics calculation."""
        with patch('agents.agent_10_video_assembly.run_ffmpeg') as mock_ffmpeg:
            with patch('agents.agent_10_video_assembly.validate_video_file') as mock_validate:
                with patch('agents.agent_10_video_assembly.get_video_metadata') as mock_metadata:
                    with patch('agents.agent_10_video_assembly.calculate_video_quality') as mock_quality:
                        with patch('agents.agent_10_video_assembly.generate_output_path') as mock_path:
                            mock_ffmpeg.return_value = mock_ffmpeg_result
                            mock_validate.return_value = True
                            mock_metadata.return_value = {"duration": 300.0, "resolution": "1920x1080"}
                            mock_quality.return_value = {
                                "video_quality": 0.92,
                                "audio_quality": 0.88,
                                "sync_quality": 0.95
                            }
                            mock_path.return_value = mock_video_file
                            
                            result = await agent.execute(context)
                            
                            # Check quality metrics
                            quality_metrics = result["video_file"]["quality_metrics"]
                            assert "video_quality" in quality_metrics
                            assert "audio_quality" in quality_metrics
                            assert "sync_quality" in quality_metrics
                            
                            for score in quality_metrics.values():
                                assert isinstance(score, float)
                                assert 0.0 <= score <= 1.0
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_file_size_calculation(self, agent, context, mock_video_file, mock_ffmpeg_result):
        """Test file size calculation."""
        with patch('agents.agent_10_video_assembly.run_ffmpeg') as mock_ffmpeg:
            with patch('agents.agent_10_video_assembly.validate_video_file') as mock_validate:
                with patch('agents.agent_10_video_assembly.get_video_metadata') as mock_metadata:
                    with patch('os.path.getsize') as mock_size:
                        with patch('agents.agent_10_video_assembly.generate_output_path') as mock_path:
                            mock_ffmpeg.return_value = mock_ffmpeg_result
                            mock_validate.return_value = True
                            mock_metadata.return_value = {"duration": 300.0, "resolution": "1920x1080"}
                            mock_size.return_value = 10485760  # 10MB
                            mock_path.return_value = mock_video_file
                            
                            result = await agent.execute(context)
                            
                            # Check file size
                            assert result["video_file"]["file_size"] == 10485760
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_codec_selection(self, agent, context, mock_video_file, mock_ffmpeg_result):
        """Test codec selection."""
        codecs = ["h264", "h265", "vp9"]
        
        for codec in codecs:
            context.data["video_settings"]["codec"] = codec
            
            with patch('agents.agent_10_video_assembly.run_ffmpeg') as mock_ffmpeg:
                with patch('agents.agent_10_video_assembly.validate_video_file') as mock_validate:
                    with patch('agents.agent_10_video_assembly.get_video_metadata') as mock_metadata:
                        with patch('agents.agent_10_video_assembly.generate_output_path') as mock_path:
                            mock_ffmpeg.return_value = mock_ffmpeg_result
                            mock_validate.return_value = True
                            mock_metadata.return_value = {"duration": 300.0, "resolution": "1920x1080"}
                            mock_path.return_value = mock_video_file
                            
                            result = await agent.execute(context)
                            
                            assert result["status"] == "success"
                            
                            # Check codec was used
                            call_args = mock_ffmpeg.call_args[0][0]
                            command_str = " ".join(call_args)
                            assert codec in command_str
    
    @pytest.mark.integration
    @pytest.mark.async_test
    async def test_full_workflow(self, agent, context, mock_video_file, mock_ffmpeg_result):
        """Test complete video assembly workflow."""
        with patch('agents.agent_10_video_assembly.run_ffmpeg') as mock_ffmpeg:
            with patch('agents.agent_10_video_assembly.validate_video_file') as mock_validate:
                with patch('agents.agent_10_video_assembly.get_video_metadata') as mock_metadata:
                    with patch('agents.agent_10_video_assembly.calculate_video_quality') as mock_quality:
                        with patch('os.path.getsize') as mock_size:
                            with patch('agents.agent_10_video_assembly.generate_output_path') as mock_path:
                                mock_ffmpeg.return_value = mock_ffmpeg_result
                                mock_validate.return_value = True
                                mock_metadata.return_value = {
                                    "duration": 300.0,
                                    "resolution": "1920x1080",
                                    "fps": 30,
                                    "bitrate": "2M",
                                    "codec": "h264"
                                }
                                mock_quality.return_value = {
                                    "video_quality": 0.92,
                                    "audio_quality": 0.88,
                                    "sync_quality": 0.95
                                }
                                mock_size.return_value = 10485760
                                mock_path.return_value = mock_video_file
                                
                                # Execute the agent
                                result = await agent.run_with_retry(context)
                                
                                # Verify complete workflow
                                assert result["status"] == "success"
                                assert agent.validate_output(result)
                                
                                # Check all required data is present
                                video_file = result["video_file"]
                                required_fields = [
                                    "file_path", "duration", "resolution", 
                                    "fps", "file_size", "format", "quality_metrics"
                                ]
                                
                                for field in required_fields:
                                    assert field in video_file
                                    assert video_file[field] is not None
    
    @pytest.mark.performance
    @pytest.mark.async_test
    async def test_performance_benchmark(self, agent, context, mock_video_file, mock_ffmpeg_result, performance_benchmark):
        """Test agent performance benchmarking."""
        with patch('agents.agent_10_video_assembly.run_ffmpeg') as mock_ffmpeg:
            with patch('agents.agent_10_video_assembly.validate_video_file') as mock_validate:
                with patch('agents.agent_10_video_assembly.get_video_metadata') as mock_metadata:
                    with patch('agents.agent_10_video_assembly.generate_output_path') as mock_path:
                        mock_ffmpeg.return_value = mock_ffmpeg_result
                        mock_validate.return_value = True
                        mock_metadata.return_value = {"duration": 300.0, "resolution": "1920x1080"}
                        mock_path.return_value = mock_video_file
                        
                        benchmark = performance_benchmark
                        benchmark.start()
                        
                        await agent.run_with_retry(context)
                        
                        benchmark.end()
                        metrics = benchmark.get_metrics()
                        
                        # Performance assertions
                        assert metrics["duration"] < 300.0  # Should complete within timeout
                        assert metrics["duration"] > 0
                        
                        # Check agent-specific metrics
                        agent_metrics = agent.get_metrics()
                        assert agent_metrics["execution_count"] == 1
                        assert agent_metrics["error_count"] == 0
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_missing_input_files(self, agent, context):
        """Test handling of missing input files."""
        # Remove audio file
        context.data["audio_file"]["file_path"] = "/nonexistent/audio.mp3"
        
        with patch('agents.agent_10_video_assembly.validate_input_files') as mock_validate:
            mock_validate.return_value = False
            
            with pytest.raises(Exception, match="Missing input files"):
                await agent.execute(context)
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_ffmpeg_timeout(self, agent, context):
        """Test FFmpeg timeout handling."""
        with patch('agents.agent_10_video_assembly.run_ffmpeg') as mock_ffmpeg:
            # Simulate timeout
            mock_ffmpeg.side_effect = asyncio.TimeoutError("FFmpeg timeout")
            
            with pytest.raises(asyncio.TimeoutError, match="FFmpeg timeout"):
                await agent.execute(context)
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_output_path_generation(self, agent, context, mock_video_file, mock_ffmpeg_result):
        """Test output path generation."""
        with patch('agents.agent_10_video_assembly.run_ffmpeg') as mock_ffmpeg:
            with patch('agents.agent_10_video_assembly.validate_video_file') as mock_validate:
                with patch('agents.agent_10_video_assembly.get_video_metadata') as mock_metadata:
                    with patch('agents.agent_10_video_assembly.generate_output_path') as mock_path:
                        mock_ffmpeg.return_value = mock_ffmpeg_result
                        mock_validate.return_value = True
                        mock_metadata.return_value = {"duration": 300.0, "resolution": "1920x1080"}
                        mock_path.return_value = mock_video_file
                        
                        result = await agent.execute(context)
                        
                        # Check output path generation
                        assert mock_path.called
                        assert result["video_file"]["file_path"] == mock_video_file