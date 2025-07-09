"""
Unit tests for AudioGenerationAgent.

Tests the voice generation and audio processing functionality
of the seventh agent in the audio generation phase.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, mock_open
from datetime import datetime, timedelta
from typing import Dict, Any
import os
import tempfile

from agents.agent_7_audio_generation import AudioGenerationAgent
from orchestrator.pipeline_context import PipelineContext


class TestAudioGenerationAgent:
    """Test cases for AudioGenerationAgent."""
    
    @pytest.fixture
    def agent(self):
        """Create AudioGenerationAgent instance."""
        return AudioGenerationAgent()
    
    @pytest.fixture
    def context(self, mock_context):
        """Provide test context with script data."""
        mock_context.data.update({
            "meditation_script": {
                "title": "5-Minute Office Stress Reset",
                "script_content": "Welcome to this 5-minute meditation...",
                "duration": "5 minutes",
                "word_count": 150,
                "estimated_duration": 4.8
            },
            "voice_settings": {
                "voice_id": "test_voice_id",
                "stability": 0.8,
                "similarity_boost": 0.7,
                "style": 0.3
            },
            "audio_settings": {
                "format": "mp3",
                "quality": "high",
                "sample_rate": 44100
            }
        })
        return mock_context
    
    @pytest.fixture
    def mock_elevenlabs_response(self):
        """Mock ElevenLabs API response."""
        return {
            "audio_data": b"mock_audio_data_content",
            "metadata": {
                "voice_id": "test_voice_id",
                "duration": 4.8,
                "format": "mp3",
                "sample_rate": 44100
            }
        }
    
    @pytest.fixture
    def mock_audio_file(self, tmp_path):
        """Create mock audio file."""
        audio_file = tmp_path / "test_audio.mp3"
        audio_file.write_bytes(b"mock_audio_data_content")
        return str(audio_file)
    
    @pytest.mark.unit
    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.agent_id == "audio_generation"
        assert agent.cost_per_call == 0.02
        assert agent.timeout == 120.0
        assert agent.max_retries == 3
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_execute_success(self, agent, context, mock_elevenlabs_response, mock_audio_file):
        """Test successful execution of audio generation."""
        with patch('agents.agent_7_audio_generation.elevenlabs_client') as mock_client:
            with patch('agents.agent_7_audio_generation.save_audio_file') as mock_save:
                mock_client.generate = AsyncMock(
                    return_value=mock_elevenlabs_response["audio_data"]
                )
                mock_save.return_value = mock_audio_file
                
                result = await agent.execute(context)
                
                assert result["status"] == "success"
                assert "audio_file" in result
                assert "file_path" in result["audio_file"]
                assert "duration" in result["audio_file"]
                assert "format" in result["audio_file"]
                assert "quality_metrics" in result["audio_file"]
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_execute_with_api_error(self, agent, context):
        """Test execution with API error."""
        with patch('agents.agent_7_audio_generation.elevenlabs_client') as mock_client:
            mock_client.generate = AsyncMock(
                side_effect=Exception("API Error")
            )
            
            with pytest.raises(Exception, match="API Error"):
                await agent.execute(context)
    
    @pytest.mark.unit
    def test_validate_output_success(self, agent, mock_audio_file):
        """Test output validation with valid data."""
        valid_output = {
            "status": "success",
            "audio_file": {
                "file_path": mock_audio_file,
                "duration": 4.8,
                "format": "mp3",
                "sample_rate": 44100,
                "file_size": 1024,
                "quality_metrics": {
                    "clarity_score": 0.92,
                    "naturalness_score": 0.88,
                    "pacing_score": 0.90
                }
            }
        }
        
        assert agent.validate_output(valid_output)
    
    @pytest.mark.unit
    def test_validate_output_missing_file(self, agent):
        """Test output validation with missing file."""
        invalid_output = {
            "status": "success",
            "audio_file": {
                "file_path": "/nonexistent/path.mp3",
                "duration": 4.8,
                "format": "mp3",
                "sample_rate": 44100,
                "file_size": 1024,
                "quality_metrics": {
                    "clarity_score": 0.92,
                    "naturalness_score": 0.88,
                    "pacing_score": 0.90
                }
            }
        }
        
        assert not agent.validate_output(invalid_output)
    
    @pytest.mark.unit
    def test_validate_output_invalid_duration(self, agent, mock_audio_file):
        """Test output validation with invalid duration."""
        invalid_output = {
            "status": "success",
            "audio_file": {
                "file_path": mock_audio_file,
                "duration": -1.0,  # Invalid duration
                "format": "mp3",
                "sample_rate": 44100,
                "file_size": 1024,
                "quality_metrics": {
                    "clarity_score": 0.92,
                    "naturalness_score": 0.88,
                    "pacing_score": 0.90
                }
            }
        }
        
        assert not agent.validate_output(invalid_output)
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_voice_generation(self, agent, context, mock_elevenlabs_response, mock_audio_file):
        """Test voice generation functionality."""
        with patch('agents.agent_7_audio_generation.elevenlabs_client') as mock_client:
            with patch('agents.agent_7_audio_generation.save_audio_file') as mock_save:
                mock_client.generate = AsyncMock(
                    return_value=mock_elevenlabs_response["audio_data"]
                )
                mock_save.return_value = mock_audio_file
                
                result = await agent.execute(context)
                
                # Check voice generation
                assert result["status"] == "success"
                assert mock_client.generate.called
                
                # Check generation parameters
                call_args = mock_client.generate.call_args
                assert "text" in call_args[1]
                assert "voice_id" in call_args[1]
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_audio_quality_metrics(self, agent, context, mock_elevenlabs_response, mock_audio_file):
        """Test audio quality metrics calculation."""
        with patch('agents.agent_7_audio_generation.elevenlabs_client') as mock_client:
            with patch('agents.agent_7_audio_generation.save_audio_file') as mock_save:
                with patch('agents.agent_7_audio_generation.analyze_audio_quality') as mock_quality:
                    mock_client.generate = AsyncMock(
                        return_value=mock_elevenlabs_response["audio_data"]
                    )
                    mock_save.return_value = mock_audio_file
                    mock_quality.return_value = {
                        "clarity_score": 0.92,
                        "naturalness_score": 0.88,
                        "pacing_score": 0.90
                    }
                    
                    result = await agent.execute(context)
                    
                    # Check quality metrics
                    quality_metrics = result["audio_file"]["quality_metrics"]
                    assert "clarity_score" in quality_metrics
                    assert "naturalness_score" in quality_metrics
                    assert "pacing_score" in quality_metrics
                    
                    for score in quality_metrics.values():
                        assert isinstance(score, float)
                        assert 0.0 <= score <= 1.0
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_different_voice_settings(self, agent, context, mock_elevenlabs_response, mock_audio_file):
        """Test different voice settings."""
        voice_settings = [
            {"voice_id": "voice1", "stability": 0.8, "similarity_boost": 0.7},
            {"voice_id": "voice2", "stability": 0.6, "similarity_boost": 0.9},
            {"voice_id": "voice3", "stability": 0.9, "similarity_boost": 0.5}
        ]
        
        for settings in voice_settings:
            context.data["voice_settings"].update(settings)
            
            with patch('agents.agent_7_audio_generation.elevenlabs_client') as mock_client:
                with patch('agents.agent_7_audio_generation.save_audio_file') as mock_save:
                    mock_client.generate = AsyncMock(
                        return_value=mock_elevenlabs_response["audio_data"]
                    )
                    mock_save.return_value = mock_audio_file
                    
                    result = await agent.execute(context)
                    
                    assert result["status"] == "success"
                    
                    # Check voice settings were applied
                    call_args = mock_client.generate.call_args
                    assert call_args[1]["voice_id"] == settings["voice_id"]
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_audio_format_options(self, agent, context, mock_elevenlabs_response, mock_audio_file):
        """Test different audio format options."""
        formats = ["mp3", "wav", "pcm"]
        
        for audio_format in formats:
            context.data["audio_settings"]["format"] = audio_format
            
            with patch('agents.agent_7_audio_generation.elevenlabs_client') as mock_client:
                with patch('agents.agent_7_audio_generation.save_audio_file') as mock_save:
                    mock_client.generate = AsyncMock(
                        return_value=mock_elevenlabs_response["audio_data"]
                    )
                    mock_save.return_value = mock_audio_file
                    
                    result = await agent.execute(context)
                    
                    assert result["status"] == "success"
                    assert result["audio_file"]["format"] == audio_format
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_file_saving(self, agent, context, mock_elevenlabs_response, mock_audio_file):
        """Test file saving functionality."""
        with patch('agents.agent_7_audio_generation.elevenlabs_client') as mock_client:
            with patch('agents.agent_7_audio_generation.save_audio_file') as mock_save:
                mock_client.generate = AsyncMock(
                    return_value=mock_elevenlabs_response["audio_data"]
                )
                mock_save.return_value = mock_audio_file
                
                result = await agent.execute(context)
                
                # Check file saving
                assert mock_save.called
                call_args = mock_save.call_args
                assert call_args[0][0] == mock_elevenlabs_response["audio_data"]
                assert call_args[0][1].endswith(".mp3")
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_duration_calculation(self, agent, context, mock_elevenlabs_response, mock_audio_file):
        """Test duration calculation."""
        with patch('agents.agent_7_audio_generation.elevenlabs_client') as mock_client:
            with patch('agents.agent_7_audio_generation.save_audio_file') as mock_save:
                with patch('agents.agent_7_audio_generation.get_audio_duration') as mock_duration:
                    mock_client.generate = AsyncMock(
                        return_value=mock_elevenlabs_response["audio_data"]
                    )
                    mock_save.return_value = mock_audio_file
                    mock_duration.return_value = 4.8
                    
                    result = await agent.execute(context)
                    
                    # Check duration calculation
                    assert result["audio_file"]["duration"] == 4.8
                    assert mock_duration.called
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_file_size_calculation(self, agent, context, mock_elevenlabs_response, mock_audio_file):
        """Test file size calculation."""
        with patch('agents.agent_7_audio_generation.elevenlabs_client') as mock_client:
            with patch('agents.agent_7_audio_generation.save_audio_file') as mock_save:
                with patch('os.path.getsize') as mock_size:
                    mock_client.generate = AsyncMock(
                        return_value=mock_elevenlabs_response["audio_data"]
                    )
                    mock_save.return_value = mock_audio_file
                    mock_size.return_value = 1024
                    
                    result = await agent.execute(context)
                    
                    # Check file size calculation
                    assert result["audio_file"]["file_size"] == 1024
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_script_preprocessing(self, agent, context, mock_elevenlabs_response, mock_audio_file):
        """Test script preprocessing for TTS."""
        # Test with script containing special characters
        context.data["meditation_script"]["script_content"] = """
        Welcome... Take a deep breath (pause) and relax.
        
        Feel the stress [melt away] as you breathe.
        
        "This is your moment of peace."
        """
        
        with patch('agents.agent_7_audio_generation.elevenlabs_client') as mock_client:
            with patch('agents.agent_7_audio_generation.save_audio_file') as mock_save:
                mock_client.generate = AsyncMock(
                    return_value=mock_elevenlabs_response["audio_data"]
                )
                mock_save.return_value = mock_audio_file
                
                result = await agent.execute(context)
                
                # Check preprocessing
                assert result["status"] == "success"
                
                # Check that text was cleaned for TTS
                call_args = mock_client.generate.call_args
                processed_text = call_args[1]["text"]
                assert "[" not in processed_text
                assert "]" not in processed_text
    
    @pytest.mark.integration
    @pytest.mark.async_test
    async def test_full_workflow(self, agent, context, mock_elevenlabs_response, mock_audio_file):
        """Test complete audio generation workflow."""
        with patch('agents.agent_7_audio_generation.elevenlabs_client') as mock_client:
            with patch('agents.agent_7_audio_generation.save_audio_file') as mock_save:
                with patch('agents.agent_7_audio_generation.get_audio_duration') as mock_duration:
                    with patch('agents.agent_7_audio_generation.analyze_audio_quality') as mock_quality:
                        with patch('os.path.getsize') as mock_size:
                            mock_client.generate = AsyncMock(
                                return_value=mock_elevenlabs_response["audio_data"]
                            )
                            mock_save.return_value = mock_audio_file
                            mock_duration.return_value = 4.8
                            mock_quality.return_value = {
                                "clarity_score": 0.92,
                                "naturalness_score": 0.88,
                                "pacing_score": 0.90
                            }
                            mock_size.return_value = 1024
                            
                            # Execute the agent
                            result = await agent.run_with_retry(context)
                            
                            # Verify complete workflow
                            assert result["status"] == "success"
                            assert agent.validate_output(result)
                            assert context.cost_tracking["total_cost"] > 0
                            
                            # Check all required data is present
                            audio_file = result["audio_file"]
                            required_fields = [
                                "file_path", "duration", "format", 
                                "sample_rate", "file_size", "quality_metrics"
                            ]
                            
                            for field in required_fields:
                                assert field in audio_file
                                assert audio_file[field] is not None
    
    @pytest.mark.performance
    @pytest.mark.async_test
    async def test_performance_benchmark(self, agent, context, mock_elevenlabs_response, mock_audio_file, performance_benchmark):
        """Test agent performance benchmarking."""
        with patch('agents.agent_7_audio_generation.elevenlabs_client') as mock_client:
            with patch('agents.agent_7_audio_generation.save_audio_file') as mock_save:
                mock_client.generate = AsyncMock(
                    return_value=mock_elevenlabs_response["audio_data"]
                )
                mock_save.return_value = mock_audio_file
                
                benchmark = performance_benchmark
                benchmark.start()
                
                await agent.run_with_retry(context)
                
                benchmark.end()
                metrics = benchmark.get_metrics()
                
                # Performance assertions
                assert metrics["duration"] < 120.0  # Should complete within timeout
                assert metrics["duration"] > 0
                
                # Check agent-specific metrics
                agent_metrics = agent.get_metrics()
                assert agent_metrics["execution_count"] == 1
                assert agent_metrics["error_count"] == 0
                assert agent_metrics["total_cost"] > 0
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_error_handling_api_limit(self, agent, context):
        """Test error handling with API limits."""
        with patch('agents.agent_7_audio_generation.elevenlabs_client') as mock_client:
            # Simulate API limit error
            mock_client.generate = AsyncMock(
                side_effect=Exception("Character limit exceeded")
            )
            
            with pytest.raises(Exception, match="Character limit exceeded"):
                await agent.execute(context)
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_voice_validation(self, agent, context, mock_elevenlabs_response, mock_audio_file):
        """Test voice validation."""
        with patch('agents.agent_7_audio_generation.elevenlabs_client') as mock_client:
            with patch('agents.agent_7_audio_generation.save_audio_file') as mock_save:
                with patch('agents.agent_7_audio_generation.validate_voice_id') as mock_validate:
                    mock_client.generate = AsyncMock(
                        return_value=mock_elevenlabs_response["audio_data"]
                    )
                    mock_save.return_value = mock_audio_file
                    mock_validate.return_value = True
                    
                    result = await agent.execute(context)
                    
                    assert result["status"] == "success"
                    assert mock_validate.called
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_context_updates(self, agent, context, mock_elevenlabs_response, mock_audio_file):
        """Test that context is properly updated."""
        with patch('agents.agent_7_audio_generation.elevenlabs_client') as mock_client:
            with patch('agents.agent_7_audio_generation.save_audio_file') as mock_save:
                mock_client.generate = AsyncMock(
                    return_value=mock_elevenlabs_response["audio_data"]
                )
                mock_save.return_value = mock_audio_file
                
                await agent.run_with_retry(context)
                
                # Check context updates
                assert context.cost_tracking["total_cost"] > 0
                assert context.cost_tracking["api_calls"] > 0
    
    @pytest.mark.slow
    @pytest.mark.async_test
    async def test_large_script_handling(self, agent, context, mock_elevenlabs_response, mock_audio_file):
        """Test handling of large scripts."""
        # Create a large script
        large_script = "Welcome to this meditation. " * 1000
        context.data["meditation_script"]["script_content"] = large_script
        
        with patch('agents.agent_7_audio_generation.elevenlabs_client') as mock_client:
            with patch('agents.agent_7_audio_generation.save_audio_file') as mock_save:
                mock_client.generate = AsyncMock(
                    return_value=mock_elevenlabs_response["audio_data"]
                )
                mock_save.return_value = mock_audio_file
                
                result = await agent.execute(context)
                
                assert result["status"] == "success"
                # Should handle large scripts appropriately