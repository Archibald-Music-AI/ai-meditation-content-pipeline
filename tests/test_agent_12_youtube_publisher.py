"""
Unit tests for YouTubePublisherAgent.

Tests the YouTube upload and publishing functionality
of the twelfth agent in the publishing phase.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, mock_open
from datetime import datetime, timedelta
from typing import Dict, Any
import os

from agents.agent_12_youtube_publisher import YouTubePublisherAgent
from orchestrator.pipeline_context import PipelineContext


class TestYouTubePublisherAgent:
    """Test cases for YouTubePublisherAgent."""
    
    @pytest.fixture
    def agent(self):
        """Create YouTubePublisherAgent instance."""
        return YouTubePublisherAgent()
    
    @pytest.fixture
    def context(self, mock_context):
        """Provide test context with video and metadata."""
        mock_context.data.update({
            "video_file": {
                "file_path": "/tmp/test_video.mp4",
                "duration": 300.0,
                "resolution": "1920x1080",
                "file_size": 10485760
            },
            "thumbnail": {
                "file_path": "/tmp/test_thumbnail.jpg",
                "resolution": "1280x720",
                "format": "jpg"
            },
            "meditation_script": {
                "title": "5-Minute Office Stress Reset",
                "script_content": "Welcome to this meditation...",
                "word_count": 150
            },
            "market_data": {
                "target_keywords": ["stress relief", "office meditation", "5 minute"],
                "trending_topics": ["workplace wellness", "quick meditation"]
            },
            "youtube_settings": {
                "privacy_status": "public",
                "category_id": "22",  # People & Blogs
                "made_for_kids": False,
                "monetization_enabled": True
            }
        })
        return mock_context
    
    @pytest.fixture
    def mock_youtube_response(self):
        """Mock YouTube API response."""
        return {
            "id": "test_video_id_12345",
            "snippet": {
                "title": "5-Minute Office Stress Reset",
                "description": "A quick meditation for workplace stress relief...",
                "channelId": "test_channel_id",
                "publishedAt": "2024-01-01T12:00:00Z",
                "thumbnails": {
                    "default": {"url": "https://i.ytimg.com/vi/test_video_id/default.jpg"},
                    "medium": {"url": "https://i.ytimg.com/vi/test_video_id/mqdefault.jpg"},
                    "high": {"url": "https://i.ytimg.com/vi/test_video_id/hqdefault.jpg"}
                },
                "tags": ["meditation", "stress relief", "office", "5 minute"],
                "categoryId": "22"
            },
            "status": {
                "uploadStatus": "processed",
                "privacyStatus": "public",
                "license": "youtube",
                "embeddable": True,
                "publicStatsViewable": True
            },
            "statistics": {
                "viewCount": "0",
                "likeCount": "0",
                "commentCount": "0"
            },
            "contentDetails": {
                "duration": "PT5M0S",
                "dimension": "2d",
                "definition": "hd",
                "caption": "false"
            }
        }
    
    @pytest.mark.unit
    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.agent_id == "youtube_publisher"
        assert agent.cost_per_call == 0.0  # No API costs for YouTube upload
        assert agent.timeout == 600.0  # 10 minutes for upload
        assert agent.max_retries == 3
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_execute_success(self, agent, context, mock_youtube_response):
        """Test successful execution of YouTube publishing."""
        with patch('agents.agent_12_youtube_publisher.youtube_service') as mock_youtube:
            with patch('agents.agent_12_youtube_publisher.generate_video_metadata') as mock_metadata:
                with patch('agents.agent_12_youtube_publisher.upload_thumbnail') as mock_upload_thumb:
                    with patch('agents.agent_12_youtube_publisher.track_upload_progress') as mock_progress:
                        mock_youtube.videos().insert().execute.return_value = mock_youtube_response
                        mock_metadata.return_value = {
                            "title": "5-Minute Office Stress Reset",
                            "description": "A quick meditation for workplace stress relief...",
                            "tags": ["meditation", "stress relief", "office", "5 minute"],
                            "category_id": "22"
                        }
                        mock_upload_thumb.return_value = True
                        mock_progress.return_value = {"status": "completed", "progress": 100}
                        
                        result = await agent.execute(context)
                        
                        assert result["status"] == "success"
                        assert "youtube_video" in result
                        assert "video_id" in result["youtube_video"]
                        assert "url" in result["youtube_video"]
                        assert "metadata" in result["youtube_video"]
                        assert "upload_metrics" in result["youtube_video"]
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_execute_with_upload_error(self, agent, context):
        """Test execution with upload error."""
        with patch('agents.agent_12_youtube_publisher.youtube_service') as mock_youtube:
            mock_youtube.videos().insert().execute.side_effect = Exception("Upload failed")
            
            with pytest.raises(Exception, match="Upload failed"):
                await agent.execute(context)
    
    @pytest.mark.unit
    def test_validate_output_success(self, agent):
        """Test output validation with valid data."""
        valid_output = {
            "status": "success",
            "youtube_video": {
                "video_id": "test_video_id_12345",
                "url": "https://www.youtube.com/watch?v=test_video_id_12345",
                "title": "5-Minute Office Stress Reset",
                "description": "A quick meditation...",
                "privacy_status": "public",
                "upload_status": "processed",
                "metadata": {
                    "duration": "PT5M0S",
                    "definition": "hd",
                    "tags": ["meditation", "stress relief"]
                },
                "upload_metrics": {
                    "upload_duration": 120.5,
                    "file_size": 10485760,
                    "processing_time": 45.2
                }
            }
        }
        
        assert agent.validate_output(valid_output)
    
    @pytest.mark.unit
    def test_validate_output_missing_video_id(self, agent):
        """Test output validation with missing video ID."""
        invalid_output = {
            "status": "success",
            "youtube_video": {
                "url": "https://www.youtube.com/watch?v=test_video_id",
                "title": "Test Video",
                "description": "Test description",
                "privacy_status": "public",
                "upload_status": "processed",
                "metadata": {},
                "upload_metrics": {}
            }
        }
        
        assert not agent.validate_output(invalid_output)
    
    @pytest.mark.unit
    def test_validate_output_invalid_url(self, agent):
        """Test output validation with invalid URL."""
        invalid_output = {
            "status": "success",
            "youtube_video": {
                "video_id": "test_video_id",
                "url": "invalid_url",
                "title": "Test Video",
                "description": "Test description",
                "privacy_status": "public",
                "upload_status": "processed",
                "metadata": {},
                "upload_metrics": {}
            }
        }
        
        assert not agent.validate_output(invalid_output)
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_video_metadata_generation(self, agent, context, mock_youtube_response):
        """Test video metadata generation."""
        with patch('agents.agent_12_youtube_publisher.youtube_service') as mock_youtube:
            with patch('agents.agent_12_youtube_publisher.generate_video_metadata') as mock_metadata:
                with patch('agents.agent_12_youtube_publisher.upload_thumbnail') as mock_upload_thumb:
                    with patch('agents.agent_12_youtube_publisher.track_upload_progress') as mock_progress:
                        mock_youtube.videos().insert().execute.return_value = mock_youtube_response
                        mock_metadata.return_value = {
                            "title": "5-Minute Office Stress Reset",
                            "description": "A quick meditation for workplace stress relief...",
                            "tags": ["meditation", "stress relief", "office", "5 minute"],
                            "category_id": "22"
                        }
                        mock_upload_thumb.return_value = True
                        mock_progress.return_value = {"status": "completed", "progress": 100}
                        
                        result = await agent.execute(context)
                        
                        # Check metadata generation
                        assert mock_metadata.called
                        metadata = result["youtube_video"]["metadata"]
                        assert "tags" in metadata or "duration" in metadata
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_title_optimization(self, agent, context, mock_youtube_response):
        """Test title optimization for YouTube."""
        with patch('agents.agent_12_youtube_publisher.youtube_service') as mock_youtube:
            with patch('agents.agent_12_youtube_publisher.generate_video_metadata') as mock_metadata:
                with patch('agents.agent_12_youtube_publisher.upload_thumbnail') as mock_upload_thumb:
                    with patch('agents.agent_12_youtube_publisher.track_upload_progress') as mock_progress:
                        mock_youtube.videos().insert().execute.return_value = mock_youtube_response
                        mock_metadata.return_value = {
                            "title": "5-Minute Office Stress Reset | Quick Meditation",
                            "description": "A quick meditation for workplace stress relief...",
                            "tags": ["meditation", "stress relief", "office", "5 minute"],
                            "category_id": "22"
                        }
                        mock_upload_thumb.return_value = True
                        mock_progress.return_value = {"status": "completed", "progress": 100}
                        
                        result = await agent.execute(context)
                        
                        # Check title optimization
                        title = result["youtube_video"]["title"]
                        assert len(title) <= 100  # YouTube title limit
                        assert any(keyword in title.lower() for keyword in ["stress", "meditation", "office"])
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_description_generation(self, agent, context, mock_youtube_response):
        """Test description generation for YouTube."""
        with patch('agents.agent_12_youtube_publisher.youtube_service') as mock_youtube:
            with patch('agents.agent_12_youtube_publisher.generate_video_metadata') as mock_metadata:
                with patch('agents.agent_12_youtube_publisher.upload_thumbnail') as mock_upload_thumb:
                    with patch('agents.agent_12_youtube_publisher.track_upload_progress') as mock_progress:
                        mock_youtube.videos().insert().execute.return_value = mock_youtube_response
                        mock_metadata.return_value = {
                            "title": "5-Minute Office Stress Reset",
                            "description": """A quick meditation for workplace stress relief.

Perfect for busy professionals who need a moment of calm during their workday.

What you'll learn:
- Quick stress relief techniques
- Mindful breathing exercises
- How to reset your mental state

Duration: 5 minutes

#meditation #stressrelief #office #mindfulness""",
                            "tags": ["meditation", "stress relief", "office", "5 minute"],
                            "category_id": "22"
                        }
                        mock_upload_thumb.return_value = True
                        mock_progress.return_value = {"status": "completed", "progress": 100}
                        
                        result = await agent.execute(context)
                        
                        # Check description quality
                        description = result["youtube_video"]["description"]
                        assert len(description) > 100  # Should be substantial
                        assert "meditation" in description.lower()
                        assert "#" in description  # Should include hashtags
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_tags_generation(self, agent, context, mock_youtube_response):
        """Test tags generation for YouTube."""
        with patch('agents.agent_12_youtube_publisher.youtube_service') as mock_youtube:
            with patch('agents.agent_12_youtube_publisher.generate_video_metadata') as mock_metadata:
                with patch('agents.agent_12_youtube_publisher.upload_thumbnail') as mock_upload_thumb:
                    with patch('agents.agent_12_youtube_publisher.track_upload_progress') as mock_progress:
                        mock_youtube.videos().insert().execute.return_value = mock_youtube_response
                        mock_metadata.return_value = {
                            "title": "5-Minute Office Stress Reset",
                            "description": "A quick meditation for workplace stress relief...",
                            "tags": ["meditation", "stress relief", "office meditation", "5 minute meditation", "workplace wellness", "mindfulness", "quick meditation", "professional", "busy", "relaxation"],
                            "category_id": "22"
                        }
                        mock_upload_thumb.return_value = True
                        mock_progress.return_value = {"status": "completed", "progress": 100}
                        
                        result = await agent.execute(context)
                        
                        # Check tags generation
                        tags = result["youtube_video"]["metadata"].get("tags", [])
                        assert len(tags) <= 15  # YouTube tag limit
                        assert any("meditation" in tag.lower() for tag in tags)
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_thumbnail_upload(self, agent, context, mock_youtube_response):
        """Test thumbnail upload functionality."""
        with patch('agents.agent_12_youtube_publisher.youtube_service') as mock_youtube:
            with patch('agents.agent_12_youtube_publisher.generate_video_metadata') as mock_metadata:
                with patch('agents.agent_12_youtube_publisher.upload_thumbnail') as mock_upload_thumb:
                    with patch('agents.agent_12_youtube_publisher.track_upload_progress') as mock_progress:
                        mock_youtube.videos().insert().execute.return_value = mock_youtube_response
                        mock_metadata.return_value = {
                            "title": "5-Minute Office Stress Reset",
                            "description": "A quick meditation...",
                            "tags": ["meditation", "stress relief"],
                            "category_id": "22"
                        }
                        mock_upload_thumb.return_value = True
                        mock_progress.return_value = {"status": "completed", "progress": 100}
                        
                        result = await agent.execute(context)
                        
                        # Check thumbnail upload
                        assert mock_upload_thumb.called
                        call_args = mock_upload_thumb.call_args
                        assert call_args[0][0] == "test_video_id_12345"  # video_id
                        assert call_args[0][1] == context.data["thumbnail"]["file_path"]
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_privacy_settings(self, agent, context, mock_youtube_response):
        """Test privacy settings configuration."""
        privacy_options = ["public", "unlisted", "private"]
        
        for privacy_status in privacy_options:
            context.data["youtube_settings"]["privacy_status"] = privacy_status
            
            with patch('agents.agent_12_youtube_publisher.youtube_service') as mock_youtube:
                with patch('agents.agent_12_youtube_publisher.generate_video_metadata') as mock_metadata:
                    with patch('agents.agent_12_youtube_publisher.upload_thumbnail') as mock_upload_thumb:
                        with patch('agents.agent_12_youtube_publisher.track_upload_progress') as mock_progress:
                            mock_response = mock_youtube_response.copy()
                            mock_response["status"]["privacyStatus"] = privacy_status
                            mock_youtube.videos().insert().execute.return_value = mock_response
                            mock_metadata.return_value = {
                                "title": "Test Video",
                                "description": "Test description",
                                "tags": ["test"],
                                "category_id": "22"
                            }
                            mock_upload_thumb.return_value = True
                            mock_progress.return_value = {"status": "completed", "progress": 100}
                            
                            result = await agent.execute(context)
                            
                            assert result["status"] == "success"
                            assert result["youtube_video"]["privacy_status"] == privacy_status
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_upload_progress_tracking(self, agent, context, mock_youtube_response):
        """Test upload progress tracking."""
        with patch('agents.agent_12_youtube_publisher.youtube_service') as mock_youtube:
            with patch('agents.agent_12_youtube_publisher.generate_video_metadata') as mock_metadata:
                with patch('agents.agent_12_youtube_publisher.upload_thumbnail') as mock_upload_thumb:
                    with patch('agents.agent_12_youtube_publisher.track_upload_progress') as mock_progress:
                        mock_youtube.videos().insert().execute.return_value = mock_youtube_response
                        mock_metadata.return_value = {
                            "title": "Test Video",
                            "description": "Test description",
                            "tags": ["test"],
                            "category_id": "22"
                        }
                        mock_upload_thumb.return_value = True
                        mock_progress.return_value = {
                            "status": "completed",
                            "progress": 100,
                            "upload_duration": 120.5,
                            "processing_time": 45.2
                        }
                        
                        result = await agent.execute(context)
                        
                        # Check progress tracking
                        assert mock_progress.called
                        upload_metrics = result["youtube_video"]["upload_metrics"]
                        assert "upload_duration" in upload_metrics
                        assert "processing_time" in upload_metrics
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_category_selection(self, agent, context, mock_youtube_response):
        """Test category selection for YouTube."""
        categories = ["22", "27", "28"]  # People & Blogs, Education, Science & Technology
        
        for category_id in categories:
            context.data["youtube_settings"]["category_id"] = category_id
            
            with patch('agents.agent_12_youtube_publisher.youtube_service') as mock_youtube:
                with patch('agents.agent_12_youtube_publisher.generate_video_metadata') as mock_metadata:
                    with patch('agents.agent_12_youtube_publisher.upload_thumbnail') as mock_upload_thumb:
                        with patch('agents.agent_12_youtube_publisher.track_upload_progress') as mock_progress:
                            mock_response = mock_youtube_response.copy()
                            mock_response["snippet"]["categoryId"] = category_id
                            mock_youtube.videos().insert().execute.return_value = mock_response
                            mock_metadata.return_value = {
                                "title": "Test Video",
                                "description": "Test description",
                                "tags": ["test"],
                                "category_id": category_id
                            }
                            mock_upload_thumb.return_value = True
                            mock_progress.return_value = {"status": "completed", "progress": 100}
                            
                            result = await agent.execute(context)
                            
                            assert result["status"] == "success"
                            # Category should be set in metadata
                            assert result["youtube_video"]["metadata"]["category_id"] == category_id
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_monetization_settings(self, agent, context, mock_youtube_response):
        """Test monetization settings."""
        with patch('agents.agent_12_youtube_publisher.youtube_service') as mock_youtube:
            with patch('agents.agent_12_youtube_publisher.generate_video_metadata') as mock_metadata:
                with patch('agents.agent_12_youtube_publisher.upload_thumbnail') as mock_upload_thumb:
                    with patch('agents.agent_12_youtube_publisher.track_upload_progress') as mock_progress:
                        with patch('agents.agent_12_youtube_publisher.set_monetization_settings') as mock_monetization:
                            mock_youtube.videos().insert().execute.return_value = mock_youtube_response
                            mock_metadata.return_value = {
                                "title": "Test Video",
                                "description": "Test description",
                                "tags": ["test"],
                                "category_id": "22"
                            }
                            mock_upload_thumb.return_value = True
                            mock_progress.return_value = {"status": "completed", "progress": 100}
                            mock_monetization.return_value = True
                            
                            result = await agent.execute(context)
                            
                            # Check monetization settings
                            if context.data["youtube_settings"]["monetization_enabled"]:
                                assert mock_monetization.called
    
    @pytest.mark.integration
    @pytest.mark.async_test
    async def test_full_workflow(self, agent, context, mock_youtube_response):
        """Test complete YouTube publishing workflow."""
        with patch('agents.agent_12_youtube_publisher.youtube_service') as mock_youtube:
            with patch('agents.agent_12_youtube_publisher.generate_video_metadata') as mock_metadata:
                with patch('agents.agent_12_youtube_publisher.upload_thumbnail') as mock_upload_thumb:
                    with patch('agents.agent_12_youtube_publisher.track_upload_progress') as mock_progress:
                        mock_youtube.videos().insert().execute.return_value = mock_youtube_response
                        mock_metadata.return_value = {
                            "title": "5-Minute Office Stress Reset | Quick Meditation",
                            "description": "A comprehensive meditation guide...",
                            "tags": ["meditation", "stress relief", "office", "5 minute"],
                            "category_id": "22"
                        }
                        mock_upload_thumb.return_value = True
                        mock_progress.return_value = {
                            "status": "completed",
                            "progress": 100,
                            "upload_duration": 120.5,
                            "processing_time": 45.2
                        }
                        
                        # Execute the agent
                        result = await agent.run_with_retry(context)
                        
                        # Verify complete workflow
                        assert result["status"] == "success"
                        assert agent.validate_output(result)
                        
                        # Check all required data is present
                        youtube_video = result["youtube_video"]
                        required_fields = [
                            "video_id", "url", "title", "description",
                            "privacy_status", "upload_status", "metadata", "upload_metrics"
                        ]
                        
                        for field in required_fields:
                            assert field in youtube_video
                            assert youtube_video[field] is not None
    
    @pytest.mark.performance
    @pytest.mark.async_test
    async def test_performance_benchmark(self, agent, context, mock_youtube_response, performance_benchmark):
        """Test agent performance benchmarking."""
        with patch('agents.agent_12_youtube_publisher.youtube_service') as mock_youtube:
            with patch('agents.agent_12_youtube_publisher.generate_video_metadata') as mock_metadata:
                with patch('agents.agent_12_youtube_publisher.upload_thumbnail') as mock_upload_thumb:
                    with patch('agents.agent_12_youtube_publisher.track_upload_progress') as mock_progress:
                        mock_youtube.videos().insert().execute.return_value = mock_youtube_response
                        mock_metadata.return_value = {
                            "title": "Test Video",
                            "description": "Test description",
                            "tags": ["test"],
                            "category_id": "22"
                        }
                        mock_upload_thumb.return_value = True
                        mock_progress.return_value = {"status": "completed", "progress": 100}
                        
                        benchmark = performance_benchmark
                        benchmark.start()
                        
                        await agent.run_with_retry(context)
                        
                        benchmark.end()
                        metrics = benchmark.get_metrics()
                        
                        # Performance assertions
                        assert metrics["duration"] < 600.0  # Should complete within timeout
                        assert metrics["duration"] > 0
                        
                        # Check agent-specific metrics
                        agent_metrics = agent.get_metrics()
                        assert agent_metrics["execution_count"] == 1
                        assert agent_metrics["error_count"] == 0
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_upload_failure_handling(self, agent, context):
        """Test upload failure handling."""
        with patch('agents.agent_12_youtube_publisher.youtube_service') as mock_youtube:
            # Simulate upload failure
            mock_youtube.videos().insert().execute.side_effect = Exception("Quota exceeded")
            
            with pytest.raises(Exception, match="Quota exceeded"):
                await agent.execute(context)
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_video_url_generation(self, agent, context, mock_youtube_response):
        """Test video URL generation."""
        with patch('agents.agent_12_youtube_publisher.youtube_service') as mock_youtube:
            with patch('agents.agent_12_youtube_publisher.generate_video_metadata') as mock_metadata:
                with patch('agents.agent_12_youtube_publisher.upload_thumbnail') as mock_upload_thumb:
                    with patch('agents.agent_12_youtube_publisher.track_upload_progress') as mock_progress:
                        mock_youtube.videos().insert().execute.return_value = mock_youtube_response
                        mock_metadata.return_value = {
                            "title": "Test Video",
                            "description": "Test description",
                            "tags": ["test"],
                            "category_id": "22"
                        }
                        mock_upload_thumb.return_value = True
                        mock_progress.return_value = {"status": "completed", "progress": 100}
                        
                        result = await agent.execute(context)
                        
                        # Check URL generation
                        video_url = result["youtube_video"]["url"]
                        video_id = result["youtube_video"]["video_id"]
                        
                        expected_url = f"https://www.youtube.com/watch?v={video_id}"
                        assert video_url == expected_url
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_context_updates(self, agent, context, mock_youtube_response):
        """Test that context is properly updated."""
        with patch('agents.agent_12_youtube_publisher.youtube_service') as mock_youtube:
            with patch('agents.agent_12_youtube_publisher.generate_video_metadata') as mock_metadata:
                with patch('agents.agent_12_youtube_publisher.upload_thumbnail') as mock_upload_thumb:
                    with patch('agents.agent_12_youtube_publisher.track_upload_progress') as mock_progress:
                        mock_youtube.videos().insert().execute.return_value = mock_youtube_response
                        mock_metadata.return_value = {
                            "title": "Test Video",
                            "description": "Test description",
                            "tags": ["test"],
                            "category_id": "22"
                        }
                        mock_upload_thumb.return_value = True
                        mock_progress.return_value = {"status": "completed", "progress": 100}
                        
                        await agent.run_with_retry(context)
                        
                        # Check context updates
                        assert context.cost_tracking["api_calls"] > 0
                        # YouTube publishing doesn't add monetary cost
                        assert context.cost_tracking["total_cost"] == 0.0