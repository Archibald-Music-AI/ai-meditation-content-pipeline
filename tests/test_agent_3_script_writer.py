"""
Unit tests for ScriptWriterAgent.

Tests the meditation script writing functionality
of the third agent in the content creation phase.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any

from agents.agent_3_script_writer import ScriptWriterAgent
from orchestrator.pipeline_context import PipelineContext


class TestScriptWriterAgent:
    """Test cases for ScriptWriterAgent."""
    
    @pytest.fixture
    def agent(self):
        """Create ScriptWriterAgent instance."""
        return ScriptWriterAgent()
    
    @pytest.fixture
    def context(self, mock_context):
        """Provide test context with meditation concept data."""
        mock_context.data.update({
            "market_data": {
                "recommended_concept": "5-Minute Office Stress Reset",
                "target_keywords": ["stress relief", "office", "5 minute"]
            },
            "meditation_concept": {
                "teaching_approach": ["breath awareness", "progressive relaxation"],
                "key_elements": ["gentle guidance", "clear instructions"],
                "learning_objectives": ["reduce stress", "improve focus"],
                "meditation_structure": {
                    "intro": "Welcome and grounding",
                    "main": "Breath awareness and relaxation",
                    "outro": "Integration and closing"
                },
                "quality_score": 0.88
            },
            "target_audience": "professionals",
            "duration_minutes": 5
        })
        return mock_context
    
    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI API response."""
        return {
            "choices": [{
                "message": {
                    "content": """
                    Welcome to this 5-minute stress relief meditation, designed specifically for busy professionals.

                    Find a comfortable position in your chair, and let's begin by taking a moment to pause from your busy day.

                    Take a deep breath in through your nose... and slowly exhale through your mouth. 

                    As you breathe, notice the sensation of your feet on the floor, grounding you in this moment.

                    Continue breathing naturally, allowing each exhale to release any tension you've been carrying.

                    Now, let's do a quick body scan. Starting from your shoulders, notice if you're holding any stress there. 

                    Gently roll your shoulders back and let them drop. Feel the release.

                    Move your attention to your jaw. Often we hold tension here without realizing it. 

                    Let your jaw soften, creating space between your upper and lower teeth.

                    Return your focus to your breath. This is your anchor in this moment of calm.

                    With each breath, you're creating space between yourself and the demands of your day.

                    As we prepare to close, take one final deep breath. 

                    When you're ready, slowly open your eyes and carry this sense of calm with you.

                    Remember, you can return to this peaceful state whenever you need it.
                    """
                }
            }],
            "usage": {
                "prompt_tokens": 200,
                "completion_tokens": 250,
                "total_tokens": 450
            }
        }
    
    @pytest.mark.unit
    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.agent_id == "script_writer"
        assert agent.cost_per_call == 0.008
        assert agent.timeout == 60.0
        assert agent.max_retries == 3
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_execute_success(self, agent, context, mock_openai_response):
        """Test successful execution of script writing."""
        with patch('agents.agent_3_script_writer.openai_client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response
            )
            
            result = await agent.execute(context)
            
            assert result["status"] == "success"
            assert "meditation_script" in result
            assert "title" in result["meditation_script"]
            assert "duration" in result["meditation_script"]
            assert "script_content" in result["meditation_script"]
            assert "word_count" in result["meditation_script"]
            assert "estimated_duration" in result["meditation_script"]
            assert "quality_metrics" in result["meditation_script"]
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_execute_with_api_error(self, agent, context):
        """Test execution with API error."""
        with patch('agents.agent_3_script_writer.openai_client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("API Error")
            )
            
            with pytest.raises(Exception, match="API Error"):
                await agent.execute(context)
    
    @pytest.mark.unit
    def test_validate_output_success(self, agent):
        """Test output validation with valid data."""
        valid_output = {
            "status": "success",
            "meditation_script": {
                "title": "5-Minute Office Stress Reset",
                "duration": "5 minutes",
                "script_content": "Welcome to this meditation...",
                "word_count": 150,
                "estimated_duration": 4.8,
                "quality_metrics": {
                    "readability_score": 0.85,
                    "pacing_score": 0.90,
                    "engagement_score": 0.88
                }
            }
        }
        
        assert agent.validate_output(valid_output)
    
    @pytest.mark.unit
    def test_validate_output_missing_fields(self, agent):
        """Test output validation with missing fields."""
        invalid_output = {
            "status": "success",
            "meditation_script": {
                "title": "Test Title",
                "script_content": "Test content",
                # Missing required fields
            }
        }
        
        assert not agent.validate_output(invalid_output)
    
    @pytest.mark.unit
    def test_validate_output_invalid_duration(self, agent):
        """Test output validation with invalid duration."""
        invalid_output = {
            "status": "success",
            "meditation_script": {
                "title": "Test Title",
                "duration": "5 minutes",
                "script_content": "Test content",
                "word_count": 50,
                "estimated_duration": -1.0,  # Invalid duration
                "quality_metrics": {
                    "readability_score": 0.85,
                    "pacing_score": 0.90,
                    "engagement_score": 0.88
                }
            }
        }
        
        assert not agent.validate_output(invalid_output)
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_script_content_generation(self, agent, context, mock_openai_response):
        """Test script content generation."""
        with patch('agents.agent_3_script_writer.openai_client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response
            )
            
            result = await agent.execute(context)
            
            # Check script content quality
            script = result["meditation_script"]["script_content"]
            assert len(script) > 100  # Should be substantial
            assert "welcome" in script.lower() or "begin" in script.lower()
            assert "breath" in script.lower() or "breathing" in script.lower()
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_word_count_calculation(self, agent, context, mock_openai_response):
        """Test word count calculation."""
        with patch('agents.agent_3_script_writer.openai_client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response
            )
            
            result = await agent.execute(context)
            
            # Check word count
            word_count = result["meditation_script"]["word_count"]
            script_content = result["meditation_script"]["script_content"]
            
            # Verify word count accuracy
            actual_words = len(script_content.split())
            assert abs(word_count - actual_words) <= 5  # Allow small variance
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_duration_estimation(self, agent, context, mock_openai_response):
        """Test duration estimation."""
        with patch('agents.agent_3_script_writer.openai_client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response
            )
            
            result = await agent.execute(context)
            
            # Check duration estimation
            estimated_duration = result["meditation_script"]["estimated_duration"]
            target_duration = context.data["duration_minutes"]
            
            # Should be close to target duration
            assert abs(estimated_duration - target_duration) <= 1.0
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_quality_metrics_calculation(self, agent, context, mock_openai_response):
        """Test quality metrics calculation."""
        with patch('agents.agent_3_script_writer.openai_client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response
            )
            
            result = await agent.execute(context)
            
            # Check quality metrics
            quality_metrics = result["meditation_script"]["quality_metrics"]
            
            for metric_name, score in quality_metrics.items():
                assert isinstance(score, float)
                assert 0.0 <= score <= 1.0
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_different_durations(self, agent, context, mock_openai_response):
        """Test script adaptation to different durations."""
        durations = [3, 5, 10, 15, 20]
        
        for duration in durations:
            context.data["duration_minutes"] = duration
            
            with patch('agents.agent_3_script_writer.openai_client') as mock_client:
                mock_client.chat.completions.create = AsyncMock(
                    return_value=mock_openai_response
                )
                
                result = await agent.execute(context)
                
                assert result["status"] == "success"
                
                # Check duration adaptation
                estimated_duration = result["meditation_script"]["estimated_duration"]
                assert abs(estimated_duration - duration) <= 1.5
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_audience_adaptation(self, agent, context, mock_openai_response):
        """Test script adaptation to different audiences."""
        audiences = [
            "professionals",
            "students",
            "seniors",
            "beginners",
            "children"
        ]
        
        for audience in audiences:
            context.data["target_audience"] = audience
            
            with patch('agents.agent_3_script_writer.openai_client') as mock_client:
                mock_client.chat.completions.create = AsyncMock(
                    return_value=mock_openai_response
                )
                
                result = await agent.execute(context)
                
                assert result["status"] == "success"
                
                # Check adaptation to audience
                script = result["meditation_script"]["script_content"]
                assert len(script) > 0
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_prompt_generation(self, agent, context):
        """Test prompt generation for script writing."""
        prompt = agent._generate_script_prompt(context)
        
        # Check prompt contains key elements
        assert "meditation script" in prompt.lower()
        assert "5-minute" in prompt.lower() or "5 minute" in prompt.lower()
        assert "professionals" in prompt.lower()
        assert "stress relief" in prompt.lower()
        assert len(prompt) > 200  # Should be detailed
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_script_structure_validation(self, agent, context, mock_openai_response):
        """Test script structure validation."""
        with patch('agents.agent_3_script_writer.openai_client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response
            )
            
            result = await agent.execute(context)
            
            # Check script structure
            script = result["meditation_script"]["script_content"]
            
            # Should have introduction
            assert any(word in script.lower() for word in ["welcome", "begin", "start"])
            
            # Should have breathing guidance
            assert any(word in script.lower() for word in ["breath", "breathing", "inhale", "exhale"])
            
            # Should have closing
            assert any(word in script.lower() for word in ["close", "end", "finish", "ready"])
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_pacing_analysis(self, agent, context, mock_openai_response):
        """Test pacing analysis."""
        with patch('agents.agent_3_script_writer.openai_client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response
            )
            
            result = await agent.execute(context)
            
            # Check pacing score
            pacing_score = result["meditation_script"]["quality_metrics"]["pacing_score"]
            assert isinstance(pacing_score, float)
            assert 0.0 <= pacing_score <= 1.0
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_readability_analysis(self, agent, context, mock_openai_response):
        """Test readability analysis."""
        with patch('agents.agent_3_script_writer.openai_client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response
            )
            
            result = await agent.execute(context)
            
            # Check readability score
            readability_score = result["meditation_script"]["quality_metrics"]["readability_score"]
            assert isinstance(readability_score, float)
            assert 0.0 <= readability_score <= 1.0
    
    @pytest.mark.integration
    @pytest.mark.async_test
    async def test_full_workflow(self, agent, context, mock_openai_response):
        """Test complete script writing workflow."""
        with patch('agents.agent_3_script_writer.openai_client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response
            )
            
            # Execute the agent
            result = await agent.run_with_retry(context)
            
            # Verify complete workflow
            assert result["status"] == "success"
            assert agent.validate_output(result)
            assert context.cost_tracking["total_cost"] > 0
            
            # Check all required data is present
            meditation_script = result["meditation_script"]
            required_fields = [
                "title", "duration", "script_content", 
                "word_count", "estimated_duration", "quality_metrics"
            ]
            
            for field in required_fields:
                assert field in meditation_script
                assert meditation_script[field] is not None
    
    @pytest.mark.performance
    @pytest.mark.async_test
    async def test_performance_benchmark(self, agent, context, mock_openai_response, performance_benchmark):
        """Test agent performance benchmarking."""
        with patch('agents.agent_3_script_writer.openai_client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response
            )
            
            benchmark = performance_benchmark
            benchmark.start()
            
            await agent.run_with_retry(context)
            
            benchmark.end()
            metrics = benchmark.get_metrics()
            
            # Performance assertions
            assert metrics["duration"] < 60.0  # Should complete within timeout
            assert metrics["duration"] > 0
            
            # Check agent-specific metrics
            agent_metrics = agent.get_metrics()
            assert agent_metrics["execution_count"] == 1
            assert agent_metrics["error_count"] == 0
            assert agent_metrics["total_cost"] > 0
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_keyword_integration(self, agent, context, mock_openai_response):
        """Test keyword integration in script."""
        with patch('agents.agent_3_script_writer.openai_client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response
            )
            
            result = await agent.execute(context)
            
            # Check keyword integration
            script = result["meditation_script"]["script_content"]
            keywords = context.data["market_data"]["target_keywords"]
            
            # At least some keywords should be present
            keyword_found = any(keyword.lower() in script.lower() for keyword in keywords)
            assert keyword_found
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_title_generation(self, agent, context, mock_openai_response):
        """Test title generation."""
        with patch('agents.agent_3_script_writer.openai_client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response
            )
            
            result = await agent.execute(context)
            
            # Check title quality
            title = result["meditation_script"]["title"]
            assert len(title) > 0
            assert len(title) < 100  # Should be concise
            
            # Should relate to concept
            concept = context.data["market_data"]["recommended_concept"]
            assert any(word in title.lower() for word in concept.lower().split())
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_error_handling_malformed_response(self, agent, context):
        """Test error handling with malformed API response."""
        malformed_response = {
            "choices": [{
                "message": {
                    "content": "Invalid response format"
                }
            }],
            "usage": {"total_tokens": 50}
        }
        
        with patch('agents.agent_3_script_writer.openai_client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=malformed_response
            )
            
            result = await agent.execute(context)
            
            # Should still produce valid output structure
            assert result["status"] == "success"
            assert "meditation_script" in result