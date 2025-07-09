"""
Unit tests for MeditationTeacherAgent.

Tests the meditation concept development and teaching methodology
of the second agent in the content creation phase.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any

from agents.agent_2_meditation_teacher import MeditationTeacherAgent
from orchestrator.pipeline_context import PipelineContext


class TestMeditationTeacherAgent:
    """Test cases for MeditationTeacherAgent."""
    
    @pytest.fixture
    def agent(self):
        """Create MeditationTeacherAgent instance."""
        return MeditationTeacherAgent()
    
    @pytest.fixture
    def context(self, mock_context):
        """Provide test context with market intelligence data."""
        mock_context.data.update({
            "market_data": {
                "trending_topics": ["stress relief", "office meditation", "mindfulness"],
                "target_keywords": ["5 minute meditation", "stress relief", "office"],
                "recommended_concept": "5-Minute Office Stress Reset"
            },
            "target_audience": "professionals",
            "duration_minutes": 5
        })
        return mock_context
    
    @pytest.fixture
    def mock_claude_response(self):
        """Mock Claude API response."""
        return {
            "content": [{
                "text": """
                Meditation Concept: "5-Minute Office Stress Reset"
                
                Teaching Approach:
                - Progressive muscle relaxation
                - Breath awareness
                - Mindful body scan
                - Stress release visualization
                
                Key Elements:
                - Gentle voice guidance
                - Clear, simple instructions
                - Professional context awareness
                - Immediate stress relief focus
                
                Learning Objectives:
                - Reduce immediate stress
                - Improve focus and clarity
                - Provide tools for workplace wellness
                - Build sustainable practice
                
                Meditation Structure:
                1. Grounding (1 minute)
                2. Breath awareness (2 minutes)
                3. Body scan (1.5 minutes)
                4. Integration (30 seconds)
                
                Quality Score: 0.92
                """
            }],
            "usage": {
                "input_tokens": 150,
                "output_tokens": 200
            }
        }
    
    @pytest.mark.unit
    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.agent_id == "meditation_teacher"
        assert agent.cost_per_call == 0.01
        assert agent.timeout == 45.0
        assert agent.max_retries == 3
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_execute_success(self, agent, context, mock_claude_response):
        """Test successful execution of meditation teaching."""
        with patch('agents.agent_2_meditation_teacher.claude_client') as mock_client:
            mock_client.messages.create = AsyncMock(
                return_value=mock_claude_response
            )
            
            result = await agent.execute(context)
            
            assert result["status"] == "success"
            assert "meditation_concept" in result
            assert "teaching_approach" in result["meditation_concept"]
            assert "key_elements" in result["meditation_concept"]
            assert "learning_objectives" in result["meditation_concept"]
            assert "meditation_structure" in result["meditation_concept"]
            assert "quality_score" in result["meditation_concept"]
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_execute_with_api_error(self, agent, context):
        """Test execution with API error."""
        with patch('agents.agent_2_meditation_teacher.claude_client') as mock_client:
            mock_client.messages.create = AsyncMock(
                side_effect=Exception("API Error")
            )
            
            with pytest.raises(Exception, match="API Error"):
                await agent.execute(context)
    
    @pytest.mark.unit
    def test_validate_output_success(self, agent):
        """Test output validation with valid data."""
        valid_output = {
            "status": "success",
            "meditation_concept": {
                "teaching_approach": ["technique1", "technique2"],
                "key_elements": ["element1", "element2"],
                "learning_objectives": ["objective1", "objective2"],
                "meditation_structure": {
                    "phase1": "description1",
                    "phase2": "description2"
                },
                "quality_score": 0.85
            }
        }
        
        assert agent.validate_output(valid_output)
    
    @pytest.mark.unit
    def test_validate_output_missing_fields(self, agent):
        """Test output validation with missing fields."""
        invalid_output = {
            "status": "success",
            "meditation_concept": {
                "teaching_approach": ["technique1"],
                # Missing required fields
            }
        }
        
        assert not agent.validate_output(invalid_output)
    
    @pytest.mark.unit
    def test_validate_output_invalid_quality_score(self, agent):
        """Test output validation with invalid quality score."""
        invalid_output = {
            "status": "success",
            "meditation_concept": {
                "teaching_approach": ["technique1"],
                "key_elements": ["element1"],
                "learning_objectives": ["objective1"],
                "meditation_structure": {"phase1": "desc"},
                "quality_score": 1.5  # Invalid score > 1.0
            }
        }
        
        assert not agent.validate_output(invalid_output)
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_teaching_approach_development(self, agent, context, mock_claude_response):
        """Test teaching approach development."""
        with patch('agents.agent_2_meditation_teacher.claude_client') as mock_client:
            mock_client.messages.create = AsyncMock(
                return_value=mock_claude_response
            )
            
            result = await agent.execute(context)
            
            # Check teaching approach quality
            teaching_approach = result["meditation_concept"]["teaching_approach"]
            assert len(teaching_approach) > 0
            assert all(isinstance(approach, str) for approach in teaching_approach)
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_learning_objectives_creation(self, agent, context, mock_claude_response):
        """Test learning objectives creation."""
        with patch('agents.agent_2_meditation_teacher.claude_client') as mock_client:
            mock_client.messages.create = AsyncMock(
                return_value=mock_claude_response
            )
            
            result = await agent.execute(context)
            
            # Check learning objectives
            objectives = result["meditation_concept"]["learning_objectives"]
            assert len(objectives) > 0
            assert all(isinstance(obj, str) for obj in objectives)
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_meditation_structure_design(self, agent, context, mock_claude_response):
        """Test meditation structure design."""
        with patch('agents.agent_2_meditation_teacher.claude_client') as mock_client:
            mock_client.messages.create = AsyncMock(
                return_value=mock_claude_response
            )
            
            result = await agent.execute(context)
            
            # Check meditation structure
            structure = result["meditation_concept"]["meditation_structure"]
            assert isinstance(structure, dict)
            assert len(structure) > 0
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_quality_score_calculation(self, agent, context, mock_claude_response):
        """Test quality score calculation."""
        with patch('agents.agent_2_meditation_teacher.claude_client') as mock_client:
            mock_client.messages.create = AsyncMock(
                return_value=mock_claude_response
            )
            
            result = await agent.execute(context)
            
            # Check quality score
            quality_score = result["meditation_concept"]["quality_score"]
            assert isinstance(quality_score, float)
            assert 0.0 <= quality_score <= 1.0
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_context_adaptation(self, agent, context, mock_claude_response):
        """Test adaptation to different contexts."""
        contexts = [
            {"target_audience": "professionals", "duration_minutes": 5},
            {"target_audience": "students", "duration_minutes": 10},
            {"target_audience": "seniors", "duration_minutes": 15}
        ]
        
        for ctx_data in contexts:
            context.data.update(ctx_data)
            
            with patch('agents.agent_2_meditation_teacher.claude_client') as mock_client:
                mock_client.messages.create = AsyncMock(
                    return_value=mock_claude_response
                )
                
                result = await agent.execute(context)
                
                assert result["status"] == "success"
                assert "meditation_concept" in result
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_prompt_generation(self, agent, context):
        """Test prompt generation for different scenarios."""
        # Test with different market data
        market_concepts = [
            "5-Minute Office Stress Reset",
            "Student Focus Meditation",
            "Evening Relaxation Practice"
        ]
        
        for concept in market_concepts:
            context.data["market_data"]["recommended_concept"] = concept
            prompt = agent._generate_teaching_prompt(context)
            
            assert concept in prompt
            assert "meditation teacher" in prompt.lower()
            assert len(prompt) > 200  # Should be detailed
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_response_parsing(self, agent, context):
        """Test response parsing with different formats."""
        test_responses = [
            {
                "content": [{
                    "text": """
                    Teaching Approach:
                    - Technique 1
                    - Technique 2
                    
                    Key Elements:
                    - Element 1
                    - Element 2
                    
                    Learning Objectives:
                    - Objective 1
                    - Objective 2
                    
                    Structure:
                    Phase 1: Description
                    Phase 2: Description
                    
                    Quality Score: 0.88
                    """
                }],
                "usage": {"input_tokens": 100, "output_tokens": 150}
            }
        ]
        
        for response in test_responses:
            with patch('agents.agent_2_meditation_teacher.claude_client') as mock_client:
                mock_client.messages.create = AsyncMock(return_value=response)
                
                result = await agent.execute(context)
                
                assert result["status"] == "success"
                assert "meditation_concept" in result
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_duration_adaptation(self, agent, context, mock_claude_response):
        """Test adaptation to different durations."""
        durations = [3, 5, 10, 15, 20]
        
        for duration in durations:
            context.data["duration_minutes"] = duration
            
            with patch('agents.agent_2_meditation_teacher.claude_client') as mock_client:
                mock_client.messages.create = AsyncMock(
                    return_value=mock_claude_response
                )
                
                result = await agent.execute(context)
                
                assert result["status"] == "success"
                # Structure should adapt to duration
                structure = result["meditation_concept"]["meditation_structure"]
                assert len(structure) > 0
    
    @pytest.mark.integration
    @pytest.mark.async_test
    async def test_full_workflow(self, agent, context, mock_claude_response):
        """Test complete meditation teaching workflow."""
        with patch('agents.agent_2_meditation_teacher.claude_client') as mock_client:
            mock_client.messages.create = AsyncMock(
                return_value=mock_claude_response
            )
            
            # Execute the agent
            result = await agent.run_with_retry(context)
            
            # Verify complete workflow
            assert result["status"] == "success"
            assert agent.validate_output(result)
            assert context.cost_tracking["total_cost"] > 0
            
            # Check all required data is present
            meditation_concept = result["meditation_concept"]
            required_fields = [
                "teaching_approach", "key_elements", 
                "learning_objectives", "meditation_structure", "quality_score"
            ]
            
            for field in required_fields:
                assert field in meditation_concept
                assert meditation_concept[field] is not None
    
    @pytest.mark.performance
    @pytest.mark.async_test
    async def test_performance_benchmark(self, agent, context, mock_claude_response, performance_benchmark):
        """Test agent performance benchmarking."""
        with patch('agents.agent_2_meditation_teacher.claude_client') as mock_client:
            mock_client.messages.create = AsyncMock(
                return_value=mock_claude_response
            )
            
            benchmark = performance_benchmark
            benchmark.start()
            
            await agent.run_with_retry(context)
            
            benchmark.end()
            metrics = benchmark.get_metrics()
            
            # Performance assertions
            assert metrics["duration"] < 45.0  # Should complete within timeout
            assert metrics["duration"] > 0
            
            # Check agent-specific metrics
            agent_metrics = agent.get_metrics()
            assert agent_metrics["execution_count"] == 1
            assert agent_metrics["error_count"] == 0
            assert agent_metrics["total_cost"] > 0
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_expertise_validation(self, agent, context, mock_claude_response):
        """Test meditation expertise validation."""
        with patch('agents.agent_2_meditation_teacher.claude_client') as mock_client:
            mock_client.messages.create = AsyncMock(
                return_value=mock_claude_response
            )
            
            result = await agent.execute(context)
            
            # Validate meditation expertise
            concept = result["meditation_concept"]
            
            # Check for meditation-specific terminology
            teaching_approach = " ".join(concept["teaching_approach"])
            assert any(term in teaching_approach.lower() for term in [
                "breath", "mindful", "awareness", "relax", "meditation"
            ])
            
            # Check quality score reflects expertise
            assert concept["quality_score"] >= 0.7  # Should be reasonably high
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_audience_specific_adaptations(self, agent, context, mock_claude_response):
        """Test audience-specific adaptations."""
        audiences = [
            "professionals",
            "students", 
            "seniors",
            "beginners",
            "parents"
        ]
        
        for audience in audiences:
            context.data["target_audience"] = audience
            
            with patch('agents.agent_2_meditation_teacher.claude_client') as mock_client:
                mock_client.messages.create = AsyncMock(
                    return_value=mock_claude_response
                )
                
                result = await agent.execute(context)
                
                assert result["status"] == "success"
                
                # Check adaptation to audience
                concept = result["meditation_concept"]
                assert len(concept["teaching_approach"]) > 0
                assert len(concept["learning_objectives"]) > 0