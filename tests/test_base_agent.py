"""
Unit tests for BaseAgent class.

Tests the core functionality of the base agent class that all
specialized agents inherit from.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any

from agents.base_agent import BaseAgent
from orchestrator.pipeline_context import PipelineContext


class TestableAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""
    
    def __init__(self, agent_id: str = "test_agent"):
        super().__init__(agent_id)
        self.execute_called = False
        self.validate_called = False
        self.test_output = {"status": "success", "result": "test_result"}
    
    async def execute(self, context: PipelineContext) -> Dict[str, Any]:
        """Test implementation of execute method."""
        self.execute_called = True
        await asyncio.sleep(0.01)  # Simulate async work
        return self.test_output
    
    def validate_output(self, output: Dict[str, Any]) -> bool:
        """Test implementation of validate_output method."""
        self.validate_called = True
        return "status" in output and output["status"] == "success"


class TestBaseAgent:
    """Test cases for BaseAgent class."""
    
    @pytest.fixture
    def agent(self):
        """Create a testable agent instance."""
        return TestableAgent("test_agent_001")
    
    @pytest.fixture
    def context(self, mock_context):
        """Provide test context."""
        return mock_context
    
    @pytest.mark.unit
    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.agent_id == "test_agent_001"
        assert agent.retry_count == 0
        assert agent.max_retries == 3
        assert agent.timeout == 30.0
        assert agent.cost_per_call == 0.0
        assert agent.last_execution_time is None
        assert agent.execution_count == 0
        assert agent.error_count == 0
        assert agent.total_cost == 0.0
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_execute_method(self, agent, context):
        """Test execute method is called correctly."""
        result = await agent.execute(context)
        
        assert agent.execute_called
        assert result == agent.test_output
        assert result["status"] == "success"
        assert result["result"] == "test_result"
    
    @pytest.mark.unit
    def test_validate_output_method(self, agent):
        """Test validate_output method."""
        valid_output = {"status": "success", "data": "test"}
        invalid_output = {"status": "error", "data": "test"}
        
        assert agent.validate_output(valid_output)
        assert not agent.validate_output(invalid_output)
        assert agent.validate_called
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_run_with_retry_success(self, agent, context):
        """Test run_with_retry with successful execution."""
        result = await agent.run_with_retry(context)
        
        assert result == agent.test_output
        assert agent.execution_count == 1
        assert agent.error_count == 0
        assert agent.retry_count == 0
        assert agent.last_execution_time is not None
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_run_with_retry_failure_then_success(self, agent, context):
        """Test run_with_retry with initial failure then success."""
        call_count = 0
        
        async def failing_execute(ctx):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary failure")
            return agent.test_output
        
        agent.execute = failing_execute
        
        result = await agent.run_with_retry(context)
        
        assert result == agent.test_output
        assert agent.execution_count == 2
        assert agent.error_count == 1
        assert agent.retry_count == 1
        assert call_count == 2
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_run_with_retry_max_retries_exceeded(self, agent, context):
        """Test run_with_retry when max retries exceeded."""
        agent.max_retries = 2
        
        async def always_failing_execute(ctx):
            raise Exception("Persistent failure")
        
        agent.execute = always_failing_execute
        
        with pytest.raises(Exception, match="Persistent failure"):
            await agent.run_with_retry(context)
        
        assert agent.execution_count == 0  # No successful executions
        assert agent.error_count == 3  # Initial attempt + 2 retries
        assert agent.retry_count == 2
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_run_with_timeout(self, agent, context):
        """Test execution with timeout."""
        agent.timeout = 0.1
        
        async def slow_execute(ctx):
            await asyncio.sleep(0.2)  # Longer than timeout
            return agent.test_output
        
        agent.execute = slow_execute
        
        with pytest.raises(asyncio.TimeoutError):
            await agent.run_with_retry(context)
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_cost_tracking(self, agent, context):
        """Test cost tracking functionality."""
        agent.cost_per_call = 0.05
        
        await agent.run_with_retry(context)
        
        assert agent.total_cost == 0.05
        assert context.cost_tracking["total_cost"] == 0.05
        assert context.cost_tracking["api_calls"] == 1
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_performance_metrics(self, agent, context):
        """Test performance metrics tracking."""
        start_time = datetime.now()
        
        await agent.run_with_retry(context)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        assert agent.last_execution_time is not None
        assert agent.last_execution_time <= execution_time
        assert agent.execution_count == 1
        
        # Check context metrics
        assert "agent_times" in context.metrics
        assert agent.agent_id in context.metrics["agent_times"]
    
    @pytest.mark.unit
    def test_get_metrics(self, agent):
        """Test get_metrics method."""
        agent.execution_count = 5
        agent.error_count = 2
        agent.total_cost = 0.25
        agent.last_execution_time = 1.5
        
        metrics = agent.get_metrics()
        
        assert metrics["agent_id"] == agent.agent_id
        assert metrics["execution_count"] == 5
        assert metrics["error_count"] == 2
        assert metrics["total_cost"] == 0.25
        assert metrics["last_execution_time"] == 1.5
        assert metrics["success_rate"] == 0.6  # 3/5 successful
    
    @pytest.mark.unit
    def test_reset_metrics(self, agent):
        """Test reset_metrics method."""
        agent.execution_count = 5
        agent.error_count = 2
        agent.total_cost = 0.25
        agent.retry_count = 1
        
        agent.reset_metrics()
        
        assert agent.execution_count == 0
        assert agent.error_count == 0
        assert agent.total_cost == 0.0
        assert agent.retry_count == 0
        assert agent.last_execution_time is None
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_context_updates(self, agent, context):
        """Test that context is properly updated during execution."""
        await agent.run_with_retry(context)
        
        # Check that context was updated
        assert context.cost_tracking["total_cost"] > 0
        assert context.cost_tracking["api_calls"] > 0
        assert "agent_times" in context.metrics
        assert agent.agent_id in context.metrics["agent_times"]
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_validation_failure(self, agent, context):
        """Test behavior when output validation fails."""
        agent.test_output = {"status": "error", "result": "invalid"}
        
        with pytest.raises(ValueError, match="Agent output validation failed"):
            await agent.run_with_retry(context)
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_exponential_backoff(self, agent, context):
        """Test exponential backoff between retries."""
        agent.max_retries = 3
        retry_times = []
        
        async def track_retry_execute(ctx):
            retry_times.append(datetime.now())
            if len(retry_times) < 3:
                raise Exception("Retry needed")
            return agent.test_output
        
        agent.execute = track_retry_execute
        
        await agent.run_with_retry(context)
        
        # Check that delays increase (exponential backoff)
        assert len(retry_times) == 3
        if len(retry_times) >= 3:
            delay1 = (retry_times[1] - retry_times[0]).total_seconds()
            delay2 = (retry_times[2] - retry_times[1]).total_seconds()
            assert delay2 > delay1  # Second delay should be longer
    
    @pytest.mark.unit
    def test_agent_configuration(self):
        """Test agent configuration parameters."""
        agent = TestableAgent("custom_agent")
        agent.max_retries = 5
        agent.timeout = 60.0
        agent.cost_per_call = 0.10
        
        assert agent.max_retries == 5
        assert agent.timeout == 60.0
        assert agent.cost_per_call == 0.10
    
    @pytest.mark.unit
    @pytest.mark.async_test
    async def test_concurrent_execution(self, context):
        """Test concurrent execution of multiple agents."""
        agents = [TestableAgent(f"agent_{i}") for i in range(3)]
        
        tasks = [agent.run_with_retry(context) for agent in agents]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result == agents[i].test_output
            assert agents[i].execution_count == 1
    
    @pytest.mark.performance
    @pytest.mark.async_test
    async def test_performance_benchmark(self, agent, context, performance_benchmark):
        """Test agent performance benchmarking."""
        benchmark = performance_benchmark
        
        benchmark.start()
        await agent.run_with_retry(context)
        benchmark.end()
        
        metrics = benchmark.get_metrics()
        
        assert metrics["duration"] > 0
        assert metrics["start_time"] is not None
        assert metrics["end_time"] is not None
        
        # Performance threshold test
        assert metrics["duration"] < 5.0  # Should complete within 5 seconds
    
    @pytest.mark.integration
    @pytest.mark.async_test
    async def test_context_persistence(self, agent, context):
        """Test that context persists across agent executions."""
        context.data["test_key"] = "test_value"
        
        await agent.run_with_retry(context)
        
        # Context should maintain data
        assert context.data["test_key"] == "test_value"
        assert context.session_id == "test_session_123"
        assert context.cost_tracking["total_cost"] > 0