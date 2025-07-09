"""
Unit tests for ThumbnailOptimizerAgent.

Tests the thumbnail creation and optimization functionality
of the eleventh agent in the assembly & optimization phase.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, mock_open
from datetime import datetime, timedelta
from typing import Dict, Any
import os
import tempfile

from agents.agent_11_thumbnail_optimizer import ThumbnailOptimizerAgent
from orchestrator.pipeline_context import PipelineContext


class TestThumbnailOptimizerAgent:
    """Test cases for ThumbnailOptimizerAgent."""
    
    @pytest.fixture
    def agent(self):
        """Create ThumbnailOptimizerAgent instance."""
        return ThumbnailOptimizerAgent()
    
    @pytest.fixture
    def context(self, mock_context):
        """Provide test context with video and metadata."""
        mock_context.data.update({
            "video_file": {
                "file_path": "/tmp/test_video.mp4",
                "duration": 300.0,
                "resolution": "1920x1080"
            },
            "meditation_script": {
                "title": "5-Minute Office Stress Reset",
                "script_content": "Welcome to this meditation..."
            },
            "market_data": {
                "target_keywords": ["stress relief", "office meditation", "5 minute"],
                "trending_topics": ["workplace wellness", "quick meditation"]
            },
            "thumbnail_settings": {
                "resolution": "1280x720",
                "format": "jpg",
                "quality": 85,
                "style": "calming"
            }
        })
        return mock_context
    
    @pytest.fixture
    def mock_thumbnail_file(self, tmp_path):
        """Create mock thumbnail file."""
        thumbnail_file = tmp_path / "test_thumbnail.jpg"
        thumbnail_file.write_bytes(b"mock_thumbnail_data_content")
        return str(thumbnail_file)
    
    @pytest.fixture
    def mock_design_response(self):
        """Mock design generation response."""
        return {
            "design_elements": {
                "title_text": "5-Minute Office Stress Reset",
                "subtitle_text": "Quick Meditation Break",
                "color_scheme": ["#4A90E2", "#7ED321", "#FFFFFF"],
                "font_choices": ["Roboto", "Open Sans"],
                "layout_style": "clean_minimal"
            },
            "thumbnail_variations": [
                {
                    "style": "modern",
                    "background": "gradient_blue",
                    "text_placement": "center_left"
                },
                {
                    "style": "nature",
                    "background": "forest_scene",
                    "text_placement": "bottom_third"
                }
            ]
        }
    
    @pytest.mark.unit
    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.agent_id == "thumbnail_optimizer"
        assert agent.cost_per_call == 0.01
        assert agent.timeout == 120.0
        assert agent.max_retries == 3
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_execute_success(self, agent, context, mock_thumbnail_file, mock_design_response):
        """Test successful execution of thumbnail optimization."""
        with patch('agents.agent_11_thumbnail_optimizer.generate_thumbnail_design') as mock_design:
            with patch('agents.agent_11_thumbnail_optimizer.create_thumbnail_image') as mock_create:
                with patch('agents.agent_11_thumbnail_optimizer.optimize_thumbnail') as mock_optimize:
                    with patch('agents.agent_11_thumbnail_optimizer.analyze_thumbnail_quality') as mock_analyze:
                        mock_design.return_value = mock_design_response
                        mock_create.return_value = mock_thumbnail_file
                        mock_optimize.return_value = mock_thumbnail_file
                        mock_analyze.return_value = {
                            "visual_appeal": 0.88,
                            "text_readability": 0.92,
                            "brand_consistency": 0.85,
                            "engagement_potential": 0.90
                        }
                        
                        result = await agent.execute(context)
                        
                        assert result["status"] == "success"
                        assert "thumbnail" in result
                        assert "file_path" in result["thumbnail"]
                        assert "design_elements" in result["thumbnail"]
                        assert "quality_metrics" in result["thumbnail"]
                        assert "variations" in result["thumbnail"]
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_execute_with_design_error(self, agent, context):
        """Test execution with design generation error."""
        with patch('agents.agent_11_thumbnail_optimizer.generate_thumbnail_design') as mock_design:
            mock_design.side_effect = Exception("Design generation failed")
            
            with pytest.raises(Exception, match="Design generation failed"):
                await agent.execute(context)
    
    @pytest.mark.unit
    def test_validate_output_success(self, agent, mock_thumbnail_file):
        """Test output validation with valid data."""
        valid_output = {
            "status": "success",
            "thumbnail": {
                "file_path": mock_thumbnail_file,
                "resolution": "1280x720",
                "format": "jpg",
                "file_size": 102400,
                "design_elements": {
                    "title_text": "Test Title",
                    "color_scheme": ["#FF0000", "#00FF00"],
                    "font_choices": ["Arial"]
                },
                "quality_metrics": {
                    "visual_appeal": 0.88,
                    "text_readability": 0.92,
                    "brand_consistency": 0.85,
                    "engagement_potential": 0.90
                },
                "variations": [
                    {"style": "modern", "file_path": mock_thumbnail_file}
                ]
            }
        }
        
        assert agent.validate_output(valid_output)
    
    @pytest.mark.unit
    def test_validate_output_missing_file(self, agent):
        """Test output validation with missing file."""
        invalid_output = {
            "status": "success",
            "thumbnail": {
                "file_path": "/nonexistent/thumbnail.jpg",
                "resolution": "1280x720",
                "format": "jpg",
                "file_size": 102400,
                "design_elements": {
                    "title_text": "Test Title",
                    "color_scheme": ["#FF0000"],
                    "font_choices": ["Arial"]
                },
                "quality_metrics": {
                    "visual_appeal": 0.88,
                    "text_readability": 0.92,
                    "brand_consistency": 0.85,
                    "engagement_potential": 0.90
                },
                "variations": []
            }
        }
        
        assert not agent.validate_output(invalid_output)
    
    @pytest.mark.unit
    def test_validate_output_invalid_quality_metrics(self, agent, mock_thumbnail_file):
        """Test output validation with invalid quality metrics."""
        invalid_output = {
            "status": "success",
            "thumbnail": {
                "file_path": mock_thumbnail_file,
                "resolution": "1280x720",
                "format": "jpg",
                "file_size": 102400,
                "design_elements": {
                    "title_text": "Test Title",
                    "color_scheme": ["#FF0000"],
                    "font_choices": ["Arial"]
                },
                "quality_metrics": {
                    "visual_appeal": 1.5,  # Invalid score > 1.0
                    "text_readability": 0.92,
                    "brand_consistency": 0.85,
                    "engagement_potential": 0.90
                },
                "variations": []
            }
        }
        
        assert not agent.validate_output(invalid_output)
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_design_generation(self, agent, context, mock_thumbnail_file, mock_design_response):
        """Test design generation functionality."""
        with patch('agents.agent_11_thumbnail_optimizer.generate_thumbnail_design') as mock_design:
            with patch('agents.agent_11_thumbnail_optimizer.create_thumbnail_image') as mock_create:
                with patch('agents.agent_11_thumbnail_optimizer.optimize_thumbnail') as mock_optimize:
                    with patch('agents.agent_11_thumbnail_optimizer.analyze_thumbnail_quality') as mock_analyze:
                        mock_design.return_value = mock_design_response
                        mock_create.return_value = mock_thumbnail_file
                        mock_optimize.return_value = mock_thumbnail_file
                        mock_analyze.return_value = {
                            "visual_appeal": 0.88,
                            "text_readability": 0.92,
                            "brand_consistency": 0.85,
                            "engagement_potential": 0.90
                        }
                        
                        result = await agent.execute(context)
                        
                        # Check design generation
                        assert mock_design.called
                        design_elements = result["thumbnail"]["design_elements"]
                        assert "title_text" in design_elements
                        assert "color_scheme" in design_elements
                        assert "font_choices" in design_elements
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_thumbnail_creation(self, agent, context, mock_thumbnail_file, mock_design_response):
        """Test thumbnail creation functionality."""
        with patch('agents.agent_11_thumbnail_optimizer.generate_thumbnail_design') as mock_design:
            with patch('agents.agent_11_thumbnail_optimizer.create_thumbnail_image') as mock_create:
                with patch('agents.agent_11_thumbnail_optimizer.optimize_thumbnail') as mock_optimize:
                    with patch('agents.agent_11_thumbnail_optimizer.analyze_thumbnail_quality') as mock_analyze:
                        mock_design.return_value = mock_design_response
                        mock_create.return_value = mock_thumbnail_file
                        mock_optimize.return_value = mock_thumbnail_file
                        mock_analyze.return_value = {
                            "visual_appeal": 0.88,
                            "text_readability": 0.92,
                            "brand_consistency": 0.85,
                            "engagement_potential": 0.90
                        }
                        
                        result = await agent.execute(context)
                        
                        # Check thumbnail creation
                        assert mock_create.called
                        assert result["thumbnail"]["file_path"] == mock_thumbnail_file
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_thumbnail_optimization(self, agent, context, mock_thumbnail_file, mock_design_response):
        """Test thumbnail optimization functionality."""
        with patch('agents.agent_11_thumbnail_optimizer.generate_thumbnail_design') as mock_design:
            with patch('agents.agent_11_thumbnail_optimizer.create_thumbnail_image') as mock_create:
                with patch('agents.agent_11_thumbnail_optimizer.optimize_thumbnail') as mock_optimize:
                    with patch('agents.agent_11_thumbnail_optimizer.analyze_thumbnail_quality') as mock_analyze:
                        mock_design.return_value = mock_design_response
                        mock_create.return_value = mock_thumbnail_file
                        mock_optimize.return_value = mock_thumbnail_file
                        mock_analyze.return_value = {
                            "visual_appeal": 0.88,
                            "text_readability": 0.92,
                            "brand_consistency": 0.85,
                            "engagement_potential": 0.90
                        }
                        
                        result = await agent.execute(context)
                        
                        # Check optimization
                        assert mock_optimize.called
                        assert result["status"] == "success"
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_quality_analysis(self, agent, context, mock_thumbnail_file, mock_design_response):
        """Test quality analysis functionality."""
        with patch('agents.agent_11_thumbnail_optimizer.generate_thumbnail_design') as mock_design:
            with patch('agents.agent_11_thumbnail_optimizer.create_thumbnail_image') as mock_create:
                with patch('agents.agent_11_thumbnail_optimizer.optimize_thumbnail') as mock_optimize:
                    with patch('agents.agent_11_thumbnail_optimizer.analyze_thumbnail_quality') as mock_analyze:
                        mock_design.return_value = mock_design_response
                        mock_create.return_value = mock_thumbnail_file
                        mock_optimize.return_value = mock_thumbnail_file
                        mock_analyze.return_value = {
                            "visual_appeal": 0.88,
                            "text_readability": 0.92,
                            "brand_consistency": 0.85,
                            "engagement_potential": 0.90
                        }
                        
                        result = await agent.execute(context)
                        
                        # Check quality analysis
                        assert mock_analyze.called
                        quality_metrics = result["thumbnail"]["quality_metrics"]
                        
                        for metric, score in quality_metrics.items():
                            assert isinstance(score, float)
                            assert 0.0 <= score <= 1.0
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_variation_generation(self, agent, context, mock_thumbnail_file, mock_design_response):
        """Test variation generation functionality."""
        with patch('agents.agent_11_thumbnail_optimizer.generate_thumbnail_design') as mock_design:
            with patch('agents.agent_11_thumbnail_optimizer.create_thumbnail_image') as mock_create:
                with patch('agents.agent_11_thumbnail_optimizer.optimize_thumbnail') as mock_optimize:
                    with patch('agents.agent_11_thumbnail_optimizer.analyze_thumbnail_quality') as mock_analyze:
                        mock_design.return_value = mock_design_response
                        mock_create.return_value = mock_thumbnail_file
                        mock_optimize.return_value = mock_thumbnail_file
                        mock_analyze.return_value = {
                            "visual_appeal": 0.88,
                            "text_readability": 0.92,
                            "brand_consistency": 0.85,
                            "engagement_potential": 0.90
                        }
                        
                        result = await agent.execute(context)
                        
                        # Check variations
                        variations = result["thumbnail"]["variations"]
                        assert len(variations) > 0
                        
                        for variation in variations:
                            assert "style" in variation
                            assert "file_path" in variation
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_different_styles(self, agent, context, mock_thumbnail_file, mock_design_response):
        """Test different thumbnail styles."""
        styles = ["modern", "nature", "minimalist", "bold", "calming"]
        
        for style in styles:
            context.data["thumbnail_settings"]["style"] = style
            
            with patch('agents.agent_11_thumbnail_optimizer.generate_thumbnail_design') as mock_design:
                with patch('agents.agent_11_thumbnail_optimizer.create_thumbnail_image') as mock_create:
                    with patch('agents.agent_11_thumbnail_optimizer.optimize_thumbnail') as mock_optimize:
                        with patch('agents.agent_11_thumbnail_optimizer.analyze_thumbnail_quality') as mock_analyze:
                            mock_design.return_value = mock_design_response
                            mock_create.return_value = mock_thumbnail_file
                            mock_optimize.return_value = mock_thumbnail_file
                            mock_analyze.return_value = {
                                "visual_appeal": 0.88,
                                "text_readability": 0.92,
                                "brand_consistency": 0.85,
                                "engagement_potential": 0.90
                            }
                            
                            result = await agent.execute(context)
                            
                            assert result["status"] == "success"
                            # Style should influence design
                            assert mock_design.called
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_title_integration(self, agent, context, mock_thumbnail_file, mock_design_response):
        """Test title integration in thumbnails."""
        with patch('agents.agent_11_thumbnail_optimizer.generate_thumbnail_design') as mock_design:
            with patch('agents.agent_11_thumbnail_optimizer.create_thumbnail_image') as mock_create:
                with patch('agents.agent_11_thumbnail_optimizer.optimize_thumbnail') as mock_optimize:
                    with patch('agents.agent_11_thumbnail_optimizer.analyze_thumbnail_quality') as mock_analyze:
                        mock_design.return_value = mock_design_response
                        mock_create.return_value = mock_thumbnail_file
                        mock_optimize.return_value = mock_thumbnail_file
                        mock_analyze.return_value = {
                            "visual_appeal": 0.88,
                            "text_readability": 0.92,
                            "brand_consistency": 0.85,
                            "engagement_potential": 0.90
                        }
                        
                        result = await agent.execute(context)
                        
                        # Check title integration
                        design_elements = result["thumbnail"]["design_elements"]
                        title = context.data["meditation_script"]["title"]
                        
                        assert title in design_elements["title_text"] or \
                               any(word in design_elements["title_text"].lower() 
                                   for word in title.lower().split())
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_keyword_optimization(self, agent, context, mock_thumbnail_file, mock_design_response):
        """Test keyword optimization in thumbnails."""
        with patch('agents.agent_11_thumbnail_optimizer.generate_thumbnail_design') as mock_design:
            with patch('agents.agent_11_thumbnail_optimizer.create_thumbnail_image') as mock_create:
                with patch('agents.agent_11_thumbnail_optimizer.optimize_thumbnail') as mock_optimize:
                    with patch('agents.agent_11_thumbnail_optimizer.analyze_thumbnail_quality') as mock_analyze:
                        mock_design.return_value = mock_design_response
                        mock_create.return_value = mock_thumbnail_file
                        mock_optimize.return_value = mock_thumbnail_file
                        mock_analyze.return_value = {
                            "visual_appeal": 0.88,
                            "text_readability": 0.92,
                            "brand_consistency": 0.85,
                            "engagement_potential": 0.90
                        }
                        
                        result = await agent.execute(context)
                        
                        # Check keyword optimization
                        design_elements = result["thumbnail"]["design_elements"]
                        keywords = context.data["market_data"]["target_keywords"]
                        
                        # At least some keywords should be reflected in design
                        title_text = design_elements["title_text"].lower()
                        keyword_found = any(keyword.lower() in title_text for keyword in keywords)
                        assert keyword_found or len(keywords) == 0
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_file_format_options(self, agent, context, mock_thumbnail_file, mock_design_response):
        """Test different file format options."""
        formats = ["jpg", "png", "webp"]
        
        for format_type in formats:
            context.data["thumbnail_settings"]["format"] = format_type
            
            with patch('agents.agent_11_thumbnail_optimizer.generate_thumbnail_design') as mock_design:
                with patch('agents.agent_11_thumbnail_optimizer.create_thumbnail_image') as mock_create:
                    with patch('agents.agent_11_thumbnail_optimizer.optimize_thumbnail') as mock_optimize:
                        with patch('agents.agent_11_thumbnail_optimizer.analyze_thumbnail_quality') as mock_analyze:
                            mock_design.return_value = mock_design_response
                            mock_create.return_value = mock_thumbnail_file
                            mock_optimize.return_value = mock_thumbnail_file
                            mock_analyze.return_value = {
                                "visual_appeal": 0.88,
                                "text_readability": 0.92,
                                "brand_consistency": 0.85,
                                "engagement_potential": 0.90
                            }
                            
                            result = await agent.execute(context)
                            
                            assert result["status"] == "success"
                            assert result["thumbnail"]["format"] == format_type
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_resolution_options(self, agent, context, mock_thumbnail_file, mock_design_response):
        """Test different resolution options."""
        resolutions = ["1280x720", "1920x1080", "640x360"]
        
        for resolution in resolutions:
            context.data["thumbnail_settings"]["resolution"] = resolution
            
            with patch('agents.agent_11_thumbnail_optimizer.generate_thumbnail_design') as mock_design:
                with patch('agents.agent_11_thumbnail_optimizer.create_thumbnail_image') as mock_create:
                    with patch('agents.agent_11_thumbnail_optimizer.optimize_thumbnail') as mock_optimize:
                        with patch('agents.agent_11_thumbnail_optimizer.analyze_thumbnail_quality') as mock_analyze:
                            mock_design.return_value = mock_design_response
                            mock_create.return_value = mock_thumbnail_file
                            mock_optimize.return_value = mock_thumbnail_file
                            mock_analyze.return_value = {
                                "visual_appeal": 0.88,
                                "text_readability": 0.92,
                                "brand_consistency": 0.85,
                                "engagement_potential": 0.90
                            }
                            
                            result = await agent.execute(context)
                            
                            assert result["status"] == "success"
                            assert result["thumbnail"]["resolution"] == resolution
    
    @pytest.mark.integration
    @pytest.mark.async_test
    async def test_full_workflow(self, agent, context, mock_thumbnail_file, mock_design_response):
        """Test complete thumbnail optimization workflow."""
        with patch('agents.agent_11_thumbnail_optimizer.generate_thumbnail_design') as mock_design:
            with patch('agents.agent_11_thumbnail_optimizer.create_thumbnail_image') as mock_create:
                with patch('agents.agent_11_thumbnail_optimizer.optimize_thumbnail') as mock_optimize:
                    with patch('agents.agent_11_thumbnail_optimizer.analyze_thumbnail_quality') as mock_analyze:
                        with patch('os.path.getsize') as mock_size:
                            mock_design.return_value = mock_design_response
                            mock_create.return_value = mock_thumbnail_file
                            mock_optimize.return_value = mock_thumbnail_file
                            mock_analyze.return_value = {
                                "visual_appeal": 0.88,
                                "text_readability": 0.92,
                                "brand_consistency": 0.85,
                                "engagement_potential": 0.90
                            }
                            mock_size.return_value = 102400
                            
                            # Execute the agent
                            result = await agent.run_with_retry(context)
                            
                            # Verify complete workflow
                            assert result["status"] == "success"
                            assert agent.validate_output(result)
                            assert context.cost_tracking["total_cost"] > 0
                            
                            # Check all required data is present
                            thumbnail = result["thumbnail"]
                            required_fields = [
                                "file_path", "resolution", "format", "file_size",
                                "design_elements", "quality_metrics", "variations"
                            ]
                            
                            for field in required_fields:
                                assert field in thumbnail
                                assert thumbnail[field] is not None
    
    @pytest.mark.performance
    @pytest.mark.async_test
    async def test_performance_benchmark(self, agent, context, mock_thumbnail_file, mock_design_response, performance_benchmark):
        """Test agent performance benchmarking."""
        with patch('agents.agent_11_thumbnail_optimizer.generate_thumbnail_design') as mock_design:
            with patch('agents.agent_11_thumbnail_optimizer.create_thumbnail_image') as mock_create:
                with patch('agents.agent_11_thumbnail_optimizer.optimize_thumbnail') as mock_optimize:
                    with patch('agents.agent_11_thumbnail_optimizer.analyze_thumbnail_quality') as mock_analyze:
                        mock_design.return_value = mock_design_response
                        mock_create.return_value = mock_thumbnail_file
                        mock_optimize.return_value = mock_thumbnail_file
                        mock_analyze.return_value = {
                            "visual_appeal": 0.88,
                            "text_readability": 0.92,
                            "brand_consistency": 0.85,
                            "engagement_potential": 0.90
                        }
                        
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
    async def test_color_scheme_generation(self, agent, context, mock_thumbnail_file, mock_design_response):
        """Test color scheme generation."""
        with patch('agents.agent_11_thumbnail_optimizer.generate_thumbnail_design') as mock_design:
            with patch('agents.agent_11_thumbnail_optimizer.create_thumbnail_image') as mock_create:
                with patch('agents.agent_11_thumbnail_optimizer.optimize_thumbnail') as mock_optimize:
                    with patch('agents.agent_11_thumbnail_optimizer.analyze_thumbnail_quality') as mock_analyze:
                        mock_design.return_value = mock_design_response
                        mock_create.return_value = mock_thumbnail_file
                        mock_optimize.return_value = mock_thumbnail_file
                        mock_analyze.return_value = {
                            "visual_appeal": 0.88,
                            "text_readability": 0.92,
                            "brand_consistency": 0.85,
                            "engagement_potential": 0.90
                        }
                        
                        result = await agent.execute(context)
                        
                        # Check color scheme
                        color_scheme = result["thumbnail"]["design_elements"]["color_scheme"]
                        assert len(color_scheme) > 0
                        
                        # Should be valid hex colors
                        for color in color_scheme:
                            assert color.startswith("#")
                            assert len(color) == 7 or len(color) == 4
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_text_readability_optimization(self, agent, context, mock_thumbnail_file, mock_design_response):
        """Test text readability optimization."""
        with patch('agents.agent_11_thumbnail_optimizer.generate_thumbnail_design') as mock_design:
            with patch('agents.agent_11_thumbnail_optimizer.create_thumbnail_image') as mock_create:
                with patch('agents.agent_11_thumbnail_optimizer.optimize_thumbnail') as mock_optimize:
                    with patch('agents.agent_11_thumbnail_optimizer.analyze_thumbnail_quality') as mock_analyze:
                        mock_design.return_value = mock_design_response
                        mock_create.return_value = mock_thumbnail_file
                        mock_optimize.return_value = mock_thumbnail_file
                        mock_analyze.return_value = {
                            "visual_appeal": 0.88,
                            "text_readability": 0.92,
                            "brand_consistency": 0.85,
                            "engagement_potential": 0.90
                        }
                        
                        result = await agent.execute(context)
                        
                        # Check text readability score
                        readability_score = result["thumbnail"]["quality_metrics"]["text_readability"]
                        assert readability_score >= 0.8  # Should be highly readable