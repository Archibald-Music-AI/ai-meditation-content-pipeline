"""
Unit tests for MarketIntelligenceAgent.

Tests the market research and trend analysis functionality
of the first agent in the content creation phase.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any

from agents.agent_1_market_intelligence import MarketIntelligenceAgent
from orchestrator.pipeline_context import PipelineContext


class TestMarketIntelligenceAgent:
    """Test cases for MarketIntelligenceAgent."""
    
    @pytest.fixture
    def agent(self):
        """Create MarketIntelligenceAgent instance."""
        return MarketIntelligenceAgent()
    
    @pytest.fixture
    def context(self, mock_context):
        """Provide test context with market research data."""
        mock_context.data.update({
            "target_audience": "professionals",
            "content_category": "stress_relief",
            "duration_preference": "5-10 minutes"
        })
        return mock_context
    
    @pytest.fixture
    def mock_perplexity_response(self):
        """Mock Perplexity API response."""
        return {
            "choices": [{
                "message": {
                    "content": """
                    Based on current market trends:
                    
                    Trending Topics:
                    - Quick office meditation (high demand)
                    - Stress relief for professionals
                    - Mindfulness breaks
                    
                    Target Keywords:
                    - "5 minute meditation"
                    - "office stress relief"
                    - "quick mindfulness"
                    
                    Competitor Analysis:
                    - Top performers: Headspace, Calm
                    - Content gaps: ultra-short professional meditations
                    
                    Recommended concept: "5-Minute Office Stress Reset"
                    """
                }
            }],
            "usage": {
                "total_tokens": 250
            }
        }
    
    @pytest.mark.unit
    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.agent_id == "market_intelligence"
        assert agent.cost_per_call == 0.002
        assert agent.timeout == 30.0
        assert agent.max_retries == 3
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_execute_success(self, agent, context, mock_perplexity_response):
        """Test successful execution of market intelligence."""
        with patch('agents.agent_1_market_intelligence.perplexity_client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_perplexity_response
            )
            
            result = await agent.execute(context)
            
            assert result["status"] == "success"
            assert "market_data" in result
            assert "trending_topics" in result["market_data"]
            assert "target_keywords" in result["market_data"]
            assert "competitor_analysis" in result["market_data"]
            assert "recommended_concept" in result["market_data"]
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_execute_with_api_error(self, agent, context):
        """Test execution with API error."""
        with patch('agents.agent_1_market_intelligence.perplexity_client') as mock_client:
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
            "market_data": {
                "trending_topics": ["topic1", "topic2"],
                "target_keywords": ["keyword1", "keyword2"],
                "competitor_analysis": {
                    "top_performers": ["competitor1"],
                    "content_gaps": ["gap1"]
                },
                "recommended_concept": "Test Concept"
            }
        }
        
        assert agent.validate_output(valid_output)
    
    @pytest.mark.unit
    def test_validate_output_missing_fields(self, agent):
        """Test output validation with missing fields."""
        invalid_output = {
            "status": "success",
            "market_data": {
                "trending_topics": ["topic1"],
                # Missing required fields
            }
        }
        
        assert not agent.validate_output(invalid_output)
    
    @pytest.mark.unit
    def test_validate_output_invalid_structure(self, agent):
        """Test output validation with invalid structure."""
        invalid_output = {
            "status": "success",
            "market_data": "invalid_structure"  # Should be dict
        }
        
        assert not agent.validate_output(invalid_output)
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_trend_analysis(self, agent, context, mock_perplexity_response):
        """Test trend analysis functionality."""
        with patch('agents.agent_1_market_intelligence.perplexity_client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_perplexity_response
            )
            
            result = await agent.execute(context)
            
            # Check trend analysis quality
            market_data = result["market_data"]
            assert len(market_data["trending_topics"]) > 0
            assert all(isinstance(topic, str) for topic in market_data["trending_topics"])
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_keyword_extraction(self, agent, context, mock_perplexity_response):
        """Test keyword extraction functionality."""
        with patch('agents.agent_1_market_intelligence.perplexity_client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_perplexity_response
            )
            
            result = await agent.execute(context)
            
            # Check keyword extraction
            keywords = result["market_data"]["target_keywords"]
            assert len(keywords) > 0
            assert all(isinstance(keyword, str) for keyword in keywords)
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_competitor_analysis(self, agent, context, mock_perplexity_response):
        """Test competitor analysis functionality."""
        with patch('agents.agent_1_market_intelligence.perplexity_client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_perplexity_response
            )
            
            result = await agent.execute(context)
            
            # Check competitor analysis
            competitor_data = result["market_data"]["competitor_analysis"]
            assert "top_performers" in competitor_data
            assert "content_gaps" in competitor_data
            assert len(competitor_data["top_performers"]) > 0
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_concept_recommendation(self, agent, context, mock_perplexity_response):
        """Test concept recommendation functionality."""
        with patch('agents.agent_1_market_intelligence.perplexity_client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_perplexity_response
            )
            
            result = await agent.execute(context)
            
            # Check concept recommendation
            concept = result["market_data"]["recommended_concept"]
            assert isinstance(concept, str)
            assert len(concept) > 0
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_context_updates(self, agent, context, mock_perplexity_response):
        """Test that context is properly updated."""
        with patch('agents.agent_1_market_intelligence.perplexity_client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_perplexity_response
            )
            
            await agent.run_with_retry(context)
            
            # Check context updates
            assert context.cost_tracking["total_cost"] > 0
            assert context.cost_tracking["api_calls"] > 0
            assert context.cost_tracking["tokens_used"] > 0
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_prompt_generation(self, agent, context):
        """Test prompt generation for different contexts."""
        # Test with different target audiences
        contexts = [
            {"target_audience": "professionals", "content_category": "stress_relief"},
            {"target_audience": "students", "content_category": "focus"},
            {"target_audience": "seniors", "content_category": "relaxation"}
        ]
        
        for ctx_data in contexts:
            context.data.update(ctx_data)
            prompt = agent._generate_market_research_prompt(context)
            
            assert ctx_data["target_audience"] in prompt
            assert ctx_data["content_category"] in prompt
            assert len(prompt) > 100  # Should be substantial
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_response_parsing(self, agent, context):
        """Test response parsing with different formats."""
        test_responses = [
            {
                "choices": [{
                    "message": {
                        "content": """
                        Trending Topics:
                        - Topic 1
                        - Topic 2
                        
                        Keywords: keyword1, keyword2
                        
                        Competitors: Comp1, Comp2
                        
                        Recommendation: Test concept
                        """
                    }
                }],
                "usage": {"total_tokens": 100}
            }
        ]
        
        for response in test_responses:
            with patch('agents.agent_1_market_intelligence.perplexity_client') as mock_client:
                mock_client.chat.completions.create = AsyncMock(return_value=response)
                
                result = await agent.execute(context)
                
                assert result["status"] == "success"
                assert "market_data" in result
    
    @pytest.mark.integration
    @pytest.mark.async_test
    async def test_full_workflow(self, agent, context, mock_perplexity_response):
        """Test complete market intelligence workflow."""
        with patch('agents.agent_1_market_intelligence.perplexity_client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_perplexity_response
            )
            
            # Execute the agent
            result = await agent.run_with_retry(context)
            
            # Verify complete workflow
            assert result["status"] == "success"
            assert agent.validate_output(result)
            assert context.cost_tracking["total_cost"] > 0
            
            # Check all required data is present
            market_data = result["market_data"]
            required_fields = [
                "trending_topics", "target_keywords", 
                "competitor_analysis", "recommended_concept"
            ]
            
            for field in required_fields:
                assert field in market_data
                assert market_data[field] is not None
    
    @pytest.mark.performance
    @pytest.mark.async_test
    async def test_performance_benchmark(self, agent, context, mock_perplexity_response, performance_benchmark):
        """Test agent performance benchmarking."""
        with patch('agents.agent_1_market_intelligence.perplexity_client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_perplexity_response
            )
            
            benchmark = performance_benchmark
            benchmark.start()
            
            await agent.run_with_retry(context)
            
            benchmark.end()
            metrics = benchmark.get_metrics()
            
            # Performance assertions
            assert metrics["duration"] < 30.0  # Should complete within timeout
            assert metrics["duration"] > 0
            
            # Check agent-specific metrics
            agent_metrics = agent.get_metrics()
            assert agent_metrics["execution_count"] == 1
            assert agent_metrics["error_count"] == 0
            assert agent_metrics["total_cost"] > 0
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_rate_limiting(self, agent, context, mock_perplexity_response):
        """Test rate limiting behavior."""
        with patch('agents.agent_1_market_intelligence.perplexity_client') as mock_client:
            # Simulate rate limiting
            mock_client.chat.completions.create = AsyncMock(
                side_effect=[
                    Exception("Rate limit exceeded"),
                    mock_perplexity_response
                ]
            )
            
            # Should succeed after retry
            result = await agent.run_with_retry(context)
            assert result["status"] == "success"
            assert agent.retry_count == 1
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_data_quality_validation(self, agent, context, mock_perplexity_response):
        """Test data quality validation."""
        with patch('agents.agent_1_market_intelligence.perplexity_client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_perplexity_response
            )
            
            result = await agent.execute(context)
            
            # Validate data quality
            market_data = result["market_data"]
            
            # Check trending topics quality
            assert len(market_data["trending_topics"]) >= 2
            assert all(len(topic.strip()) > 0 for topic in market_data["trending_topics"])
            
            # Check keywords quality
            assert len(market_data["target_keywords"]) >= 2
            assert all(len(keyword.strip()) > 0 for keyword in market_data["target_keywords"])
            
            # Check recommendation quality
            assert len(market_data["recommended_concept"]) > 10  # Should be descriptive