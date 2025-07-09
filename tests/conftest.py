"""
pytest configuration and fixtures for Enhanced Archibald v5.1 test suite.

This module provides common test fixtures, mock implementations,
and configuration for the entire test suite.
"""
import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from datetime import datetime, timedelta
import json
import os
from typing import Dict, Any, Optional

# Import project modules
from orchestrator.pipeline_context import PipelineContext
from orchestrator.enhanced_orchestrator import EnhancedOrchestrator
from agents.base_agent import BaseAgent


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_context():
    """Create a mock PipelineContext for testing."""
    context = PipelineContext()
    context.session_id = "test_session_123"
    context.created_at = datetime.now()
    context.status = "initialized"
    context.phase = "content_creation"
    context.data = {
        "meditation_concept": "stress relief",
        "target_audience": "professionals",
        "duration_minutes": 10,
        "voice_id": "test_voice_id",
        "music_style": "ambient"
    }
    context.cost_tracking = {
        "total_cost": 0.0,
        "api_calls": 0,
        "tokens_used": 0
    }
    context.metrics = {
        "start_time": datetime.now(),
        "phase_times": {},
        "quality_scores": {}
    }
    return context


@pytest.fixture
def mock_api_response():
    """Create mock API response data."""
    return {
        "openai_response": {
            "id": "test_id",
            "object": "text_completion",
            "created": 1234567890,
            "choices": [{
                "text": "This is a test meditation script about stress relief...",
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300
            }
        },
        "elevenlabs_response": {
            "audio_url": "https://test.com/audio.mp3",
            "duration": 600,
            "voice_id": "test_voice_id"
        },
        "fal_response": {
            "music_url": "https://test.com/music.mp3",
            "image_url": "https://test.com/image.jpg"
        }
    }


@pytest.fixture
def mock_agent_output():
    """Create mock agent output for testing."""
    return {
        "market_intelligence": {
            "trending_topics": ["stress relief", "sleep meditation", "mindfulness"],
            "target_keywords": ["meditation", "relaxation", "stress"],
            "competitor_analysis": {
                "top_performers": ["Channel A", "Channel B"],
                "content_gaps": ["quick office meditation"]
            },
            "recommended_concept": "5-minute office stress relief"
        },
        "meditation_script": {
            "title": "Quick Office Stress Relief",
            "duration": "5 minutes",
            "script": "Welcome to this brief meditation...",
            "key_points": ["breathing", "tension release", "focus"],
            "quality_score": 0.92
        },
        "audio_file": {
            "file_path": "/tmp/test_audio.mp3",
            "duration": 300,
            "format": "mp3",
            "quality": "high"
        },
        "video_file": {
            "file_path": "/tmp/test_video.mp4",
            "duration": 300,
            "resolution": "1920x1080",
            "format": "mp4"
        }
    }


@pytest.fixture
def mock_file_system(tmp_path):
    """Create mock file system for testing."""
    # Create test directories
    audio_dir = tmp_path / "audio"
    video_dir = tmp_path / "video"
    image_dir = tmp_path / "images"
    
    audio_dir.mkdir()
    video_dir.mkdir()
    image_dir.mkdir()
    
    # Create test files
    test_audio = audio_dir / "test.mp3"
    test_video = video_dir / "test.mp4"
    test_image = image_dir / "test.jpg"
    
    test_audio.write_text("mock audio content")
    test_video.write_text("mock video content")
    test_image.write_text("mock image content")
    
    return {
        "audio_dir": str(audio_dir),
        "video_dir": str(video_dir),
        "image_dir": str(image_dir),
        "test_audio": str(test_audio),
        "test_video": str(test_video),
        "test_image": str(test_image)
    }


@pytest.fixture
def mock_external_apis():
    """Create mock external API clients."""
    class MockOpenAI:
        def __init__(self):
            self.chat = Mock()
            self.chat.completions = Mock()
            self.chat.completions.create = AsyncMock(return_value=Mock(
                choices=[Mock(message=Mock(content="Test response"))],
                usage=Mock(total_tokens=100)
            ))
    
    class MockElevenLabs:
        def __init__(self):
            self.generate = AsyncMock(return_value=b"mock audio data")
            self.get_voices = AsyncMock(return_value=[{"voice_id": "test_voice"}])
    
    class MockFAL:
        def __init__(self):
            self.subscribe = AsyncMock(return_value={
                "audio_url": "https://test.com/music.mp3",
                "image_url": "https://test.com/image.jpg"
            })
    
    class MockYouTube:
        def __init__(self):
            self.upload = AsyncMock(return_value={
                "video_id": "test_video_id",
                "url": "https://youtube.com/watch?v=test_video_id"
            })
    
    return {
        "openai": MockOpenAI(),
        "elevenlabs": MockElevenLabs(),
        "fal": MockFAL(),
        "youtube": MockYouTube()
    }


@pytest.fixture
def mock_base_agent():
    """Create a mock base agent for testing."""
    class MockAgent(BaseAgent):
        def __init__(self, agent_id: str = "test_agent"):
            super().__init__(agent_id)
            self.execute_called = False
            self.validate_called = False
        
        async def execute(self, context: PipelineContext) -> Dict[str, Any]:
            self.execute_called = True
            return {"status": "success", "result": "test_result"}
        
        def validate_output(self, output: Dict[str, Any]) -> bool:
            self.validate_called = True
            return True
    
    return MockAgent


@pytest.fixture
def performance_benchmark():
    """Create performance benchmark utilities."""
    class PerformanceBenchmark:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.metrics = {}
        
        def start(self):
            self.start_time = datetime.now()
        
        def end(self):
            self.end_time = datetime.now()
        
        def duration(self) -> float:
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time).total_seconds()
            return 0.0
        
        def add_metric(self, name: str, value: Any):
            self.metrics[name] = value
        
        def get_metrics(self) -> Dict[str, Any]:
            return {
                "duration": self.duration(),
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                **self.metrics
            }
    
    return PerformanceBenchmark


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        "test_timeout": 30,
        "max_retries": 3,
        "mock_delays": {
            "api_call": 0.1,
            "file_processing": 0.2,
            "video_assembly": 0.5
        },
        "performance_thresholds": {
            "agent_execution": 5.0,  # seconds
            "phase_completion": 30.0,  # seconds
            "total_pipeline": 300.0  # seconds
        }
    }


@pytest.fixture
def mock_environment(monkeypatch):
    """Mock environment variables for testing."""
    env_vars = {
        "OPENAI_API_KEY": "test_openai_key",
        "ANTHROPIC_API_KEY": "test_anthropic_key",
        "ELEVENLABS_API_KEY": "test_elevenlabs_key",
        "FAL_API_KEY": "test_fal_key",
        "PERPLEXITY_API_KEY": "test_perplexity_key",
        "YOUTUBE_API_KEY": "test_youtube_key"
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    return env_vars


# Custom pytest markers for test organization
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "async_test: mark test as async"
    )


# Async test utilities
class AsyncTestUtils:
    """Utilities for async testing."""
    
    @staticmethod
    async def run_with_timeout(coro, timeout: float = 10.0):
        """Run coroutine with timeout."""
        return await asyncio.wait_for(coro, timeout=timeout)
    
    @staticmethod
    def create_async_mock(return_value=None):
        """Create an async mock."""
        mock = AsyncMock()
        if return_value is not None:
            mock.return_value = return_value
        return mock


@pytest.fixture
def async_utils():
    """Provide async testing utilities."""
    return AsyncTestUtils


# Quality assurance helpers
class QualityAssurance:
    """Quality assurance utilities for testing."""
    
    @staticmethod
    def validate_agent_output(output: Dict[str, Any], required_keys: list) -> bool:
        """Validate agent output has required keys."""
        return all(key in output for key in required_keys)
    
    @staticmethod
    def validate_context_state(context: PipelineContext) -> bool:
        """Validate context is in valid state."""
        return (
            context.session_id is not None and
            context.status in ["initialized", "running", "completed", "failed"] and
            context.phase is not None
        )
    
    @staticmethod
    def calculate_quality_score(metrics: Dict[str, Any]) -> float:
        """Calculate quality score from metrics."""
        if not metrics:
            return 0.0
        
        # Simple quality score calculation
        total_score = 0.0
        count = 0
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and 0 <= value <= 1:
                total_score += value
                count += 1
        
        return total_score / count if count > 0 else 0.0


@pytest.fixture
def quality_assurance():
    """Provide quality assurance utilities."""
    return QualityAssurance