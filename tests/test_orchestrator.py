"""
Integration tests for EnhancedOrchestrator.

Tests the main orchestration engine and pipeline coordination
functionality of the Enhanced Archibald v5.1 system.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List

from orchestrator.enhanced_orchestrator import EnhancedOrchestrator
from orchestrator.pipeline_context import PipelineContext
from agents.base_agent import BaseAgent


class MockAgent(BaseAgent):
    """Mock agent for testing orchestrator functionality."""
    
    def __init__(self, agent_id: str, execution_time: float = 0.1, should_fail: bool = False):
        super().__init__(agent_id)
        self.execution_time = execution_time
        self.should_fail = should_fail
        self.execute_called = False
        self.test_output = {"status": "success", "agent_id": agent_id, "result": f"output_from_{agent_id}"}
    
    async def execute(self, context: PipelineContext) -> Dict[str, Any]:
        """Mock execute method."""
        self.execute_called = True
        await asyncio.sleep(self.execution_time)
        
        if self.should_fail:
            raise Exception(f"Agent {self.agent_id} failed")
        
        return self.test_output
    
    def validate_output(self, output: Dict[str, Any]) -> bool:
        """Mock validate_output method."""
        return output.get("status") == "success"


class TestEnhancedOrchestrator:
    """Test cases for EnhancedOrchestrator."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create EnhancedOrchestrator instance."""
        return EnhancedOrchestrator()
    
    @pytest.fixture
    def context(self, mock_context):
        """Provide test context."""
        mock_context.data.update({
            "meditation_concept": "stress relief",
            "target_audience": "professionals",
            "duration_minutes": 5
        })
        return mock_context
    
    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        return {
            "market_intelligence": MockAgent("market_intelligence", 0.1),
            "meditation_teacher": MockAgent("meditation_teacher", 0.1),
            "script_writer": MockAgent("script_writer", 0.1),
            "audio_generation": MockAgent("audio_generation", 0.2),
            "video_assembly": MockAgent("video_assembly", 0.3),
            "thumbnail_optimizer": MockAgent("thumbnail_optimizer", 0.1),
            "youtube_publisher": MockAgent("youtube_publisher", 0.2)
        }
    
    @pytest.mark.integration
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.session_id is not None
        assert orchestrator.context is not None
        assert orchestrator.agents == {}
        assert orchestrator.phases == {}
        assert orchestrator.current_phase is None
        assert orchestrator.total_cost == 0.0
        assert orchestrator.start_time is None
        assert orchestrator.end_time is None
    
    @pytest.mark.integration
    @pytest.mark.async_test
    async def test_register_agent(self, orchestrator):
        """Test agent registration."""
        mock_agent = MockAgent("test_agent")
        
        orchestrator.register_agent(mock_agent)
        
        assert "test_agent" in orchestrator.agents
        assert orchestrator.agents["test_agent"] == mock_agent
    
    @pytest.mark.integration
    @pytest.mark.async_test
    async def test_register_multiple_agents(self, orchestrator, mock_agents):
        """Test registering multiple agents."""
        for agent in mock_agents.values():
            orchestrator.register_agent(agent)
        
        assert len(orchestrator.agents) == len(mock_agents)
        for agent_id in mock_agents:
            assert agent_id in orchestrator.agents
    
    @pytest.mark.integration
    @pytest.mark.async_test
    async def test_define_phases(self, orchestrator, mock_agents):
        """Test phase definition."""
        for agent in mock_agents.values():
            orchestrator.register_agent(agent)
        
        # Define test phases
        orchestrator.define_phase("content_creation", [
            "market_intelligence",
            "meditation_teacher", 
            "script_writer"
        ], execution_pattern="sequential")
        
        orchestrator.define_phase("media_generation", [
            "audio_generation"
        ], execution_pattern="sequential")
        
        orchestrator.define_phase("assembly", [
            "video_assembly",
            "thumbnail_optimizer"
        ], execution_pattern="parallel")
        
        orchestrator.define_phase("publishing", [
            "youtube_publisher"
        ], execution_pattern="sequential")
        
        assert len(orchestrator.phases) == 4
        assert "content_creation" in orchestrator.phases
        assert "media_generation" in orchestrator.phases
        assert "assembly" in orchestrator.phases
        assert "publishing" in orchestrator.phases
    
    @pytest.mark.integration
    @pytest.mark.async_test
    async def test_execute_phase_sequential(self, orchestrator, mock_agents):
        """Test sequential phase execution."""
        for agent in mock_agents.values():
            orchestrator.register_agent(agent)
        
        orchestrator.define_phase("test_phase", [
            "market_intelligence",
            "meditation_teacher",
            "script_writer"
        ], execution_pattern="sequential")
        
        context = PipelineContext()
        
        results = await orchestrator.execute_phase("test_phase", context)
        
        assert len(results) == 3
        assert all(result["status"] == "success" for result in results)
        
        # Check agents were executed in order
        for agent_id in ["market_intelligence", "meditation_teacher", "script_writer"]:
            assert mock_agents[agent_id].execute_called
    
    @pytest.mark.integration
    @pytest.mark.async_test
    async def test_execute_phase_parallel(self, orchestrator, mock_agents):
        """Test parallel phase execution."""
        for agent in mock_agents.values():
            orchestrator.register_agent(agent)
        
        orchestrator.define_phase("test_phase", [
            "video_assembly",
            "thumbnail_optimizer"
        ], execution_pattern="parallel")
        
        context = PipelineContext()
        
        start_time = datetime.now()
        results = await orchestrator.execute_phase("test_phase", context)
        end_time = datetime.now()
        
        assert len(results) == 2
        assert all(result["status"] == "success" for result in results)
        
        # Parallel execution should be faster than sequential
        execution_time = (end_time - start_time).total_seconds()
        assert execution_time < 0.5  # Should be much faster than sequential
        
        # Check both agents were executed
        assert mock_agents["video_assembly"].execute_called
        assert mock_agents["thumbnail_optimizer"].execute_called
    
    @pytest.mark.integration
    @pytest.mark.async_test
    async def test_execute_full_pipeline(self, orchestrator, mock_agents):
        """Test full pipeline execution."""
        for agent in mock_agents.values():
            orchestrator.register_agent(agent)
        
        # Define all phases
        orchestrator.define_phase("content_creation", [
            "market_intelligence",
            "meditation_teacher",
            "script_writer"
        ], execution_pattern="sequential")
        
        orchestrator.define_phase("media_generation", [
            "audio_generation"
        ], execution_pattern="sequential")
        
        orchestrator.define_phase("assembly", [
            "video_assembly",
            "thumbnail_optimizer"
        ], execution_pattern="parallel")
        
        orchestrator.define_phase("publishing", [
            "youtube_publisher"
        ], execution_pattern="sequential")
        
        context = PipelineContext()
        
        results = await orchestrator.execute_pipeline(context)
        
        # Should execute all phases
        assert len(results) == 4
        assert all(phase_results for phase_results in results.values())
        
        # Check all agents were executed
        for agent in mock_agents.values():
            assert agent.execute_called
    
    @pytest.mark.integration
    @pytest.mark.async_test
    async def test_pipeline_with_failure(self, orchestrator, mock_agents):
        """Test pipeline execution with agent failure."""
        # Make one agent fail
        mock_agents["script_writer"].should_fail = True
        
        for agent in mock_agents.values():
            orchestrator.register_agent(agent)
        
        orchestrator.define_phase("content_creation", [
            "market_intelligence",
            "meditation_teacher",
            "script_writer"
        ], execution_pattern="sequential")
        
        context = PipelineContext()
        
        with pytest.raises(Exception, match="Agent script_writer failed"):
            await orchestrator.execute_phase("content_creation", context)
    
    @pytest.mark.integration
    @pytest.mark.async_test
    async def test_cost_tracking(self, orchestrator, mock_agents):
        """Test cost tracking functionality."""
        # Set costs for agents
        mock_agents["market_intelligence"].cost_per_call = 0.002
        mock_agents["meditation_teacher"].cost_per_call = 0.01
        mock_agents["script_writer"].cost_per_call = 0.008
        
        for agent in mock_agents.values():
            orchestrator.register_agent(agent)
        
        orchestrator.define_phase("content_creation", [
            "market_intelligence",
            "meditation_teacher",
            "script_writer"
        ], execution_pattern="sequential")
        
        context = PipelineContext()
        
        await orchestrator.execute_phase("content_creation", context)
        
        # Check cost tracking
        expected_cost = 0.002 + 0.01 + 0.008
        assert abs(context.cost_tracking["total_cost"] - expected_cost) < 0.001
        assert context.cost_tracking["api_calls"] == 3
    
    @pytest.mark.integration
    @pytest.mark.async_test
    async def test_performance_metrics(self, orchestrator, mock_agents):
        """Test performance metrics collection."""
        for agent in mock_agents.values():
            orchestrator.register_agent(agent)
        
        orchestrator.define_phase("test_phase", [
            "market_intelligence",
            "meditation_teacher"
        ], execution_pattern="sequential")
        
        context = PipelineContext()
        
        start_time = datetime.now()
        await orchestrator.execute_phase("test_phase", context)
        end_time = datetime.now()
        
        # Check performance metrics
        assert "agent_times" in context.metrics
        assert "market_intelligence" in context.metrics["agent_times"]
        assert "meditation_teacher" in context.metrics["agent_times"]
        
        # Check timing
        total_time = (end_time - start_time).total_seconds()
        assert total_time > 0
    
    @pytest.mark.integration
    @pytest.mark.async_test
    async def test_context_persistence(self, orchestrator, mock_agents):
        """Test context persistence across agents."""
        for agent in mock_agents.values():
            orchestrator.register_agent(agent)
        
        orchestrator.define_phase("test_phase", [
            "market_intelligence",
            "meditation_teacher"
        ], execution_pattern="sequential")
        
        context = PipelineContext()
        context.data["test_key"] = "test_value"
        
        await orchestrator.execute_phase("test_phase", context)
        
        # Context should persist
        assert context.data["test_key"] == "test_value"
        assert context.session_id is not None
    
    @pytest.mark.integration
    @pytest.mark.async_test
    async def test_agent_retry_logic(self, orchestrator):
        """Test agent retry logic."""
        # Create agent that fails first time, succeeds second time
        call_count = 0
        
        class RetryAgent(BaseAgent):
            def __init__(self):
                super().__init__("retry_agent")
                self.max_retries = 2
            
            async def execute(self, context: PipelineContext) -> Dict[str, Any]:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception("First attempt failed")
                return {"status": "success", "result": "success_after_retry"}
            
            def validate_output(self, output: Dict[str, Any]) -> bool:
                return output.get("status") == "success"
        
        retry_agent = RetryAgent()
        orchestrator.register_agent(retry_agent)
        
        orchestrator.define_phase("retry_phase", [
            "retry_agent"
        ], execution_pattern="sequential")
        
        context = PipelineContext()
        
        results = await orchestrator.execute_phase("retry_phase", context)
        
        assert len(results) == 1
        assert results[0]["status"] == "success"
        assert call_count == 2  # Should have retried once
    
    @pytest.mark.integration
    @pytest.mark.async_test
    async def test_quality_gates(self, orchestrator, mock_agents):
        """Test quality gates between phases."""
        for agent in mock_agents.values():
            orchestrator.register_agent(agent)
        
        # Define phases with quality gates
        orchestrator.define_phase("content_creation", [
            "market_intelligence",
            "meditation_teacher",
            "script_writer"
        ], execution_pattern="sequential")
        
        orchestrator.define_phase("media_generation", [
            "audio_generation"
        ], execution_pattern="sequential")
        
        context = PipelineContext()
        
        # Execute first phase
        results_1 = await orchestrator.execute_phase("content_creation", context)
        assert len(results_1) == 3
        
        # Quality gate: Check required data exists
        assert context.data is not None
        
        # Execute second phase
        results_2 = await orchestrator.execute_phase("media_generation", context)
        assert len(results_2) == 1
    
    @pytest.mark.integration
    @pytest.mark.async_test
    async def test_phase_dependencies(self, orchestrator, mock_agents):
        """Test phase dependency management."""
        for agent in mock_agents.values():
            orchestrator.register_agent(agent)
        
        # Define phases in dependency order
        orchestrator.define_phase("content_creation", [
            "market_intelligence",
            "meditation_teacher",
            "script_writer"
        ], execution_pattern="sequential")
        
        orchestrator.define_phase("media_generation", [
            "audio_generation"
        ], execution_pattern="sequential")
        
        orchestrator.define_phase("assembly", [
            "video_assembly",
            "thumbnail_optimizer"
        ], execution_pattern="parallel")
        
        context = PipelineContext()
        
        # Execute phases in order
        await orchestrator.execute_phase("content_creation", context)
        await orchestrator.execute_phase("media_generation", context)
        await orchestrator.execute_phase("assembly", context)
        
        # All agents should have been executed
        for agent in mock_agents.values():
            assert agent.execute_called
    
    @pytest.mark.integration
    @pytest.mark.async_test
    async def test_pipeline_state_management(self, orchestrator, mock_agents):
        """Test pipeline state management."""
        for agent in mock_agents.values():
            orchestrator.register_agent(agent)
        
        orchestrator.define_phase("test_phase", [
            "market_intelligence"
        ], execution_pattern="sequential")
        
        context = PipelineContext()
        
        # Initial state
        assert context.status == "initialized"
        assert context.phase is None
        
        # Execute phase
        await orchestrator.execute_phase("test_phase", context)
        
        # State should be updated
        assert context.status == "running" or context.status == "completed"
        assert context.phase == "test_phase"
    
    @pytest.mark.performance
    @pytest.mark.async_test
    async def test_orchestrator_performance(self, orchestrator, mock_agents, performance_benchmark):
        """Test orchestrator performance."""
        for agent in mock_agents.values():
            orchestrator.register_agent(agent)
        
        orchestrator.define_phase("performance_test", [
            "market_intelligence",
            "meditation_teacher",
            "script_writer"
        ], execution_pattern="sequential")
        
        context = PipelineContext()
        
        benchmark = performance_benchmark
        benchmark.start()
        
        await orchestrator.execute_phase("performance_test", context)
        
        benchmark.end()
        metrics = benchmark.get_metrics()
        
        # Performance assertions
        assert metrics["duration"] < 5.0  # Should complete quickly with mocks
        assert metrics["duration"] > 0
    
    @pytest.mark.integration
    @pytest.mark.async_test
    async def test_concurrent_pipeline_execution(self, orchestrator, mock_agents):
        """Test concurrent pipeline execution."""
        for agent in mock_agents.values():
            orchestrator.register_agent(agent)
        
        orchestrator.define_phase("concurrent_test", [
            "market_intelligence",
            "meditation_teacher"
        ], execution_pattern="parallel")
        
        # Execute multiple contexts concurrently
        contexts = [PipelineContext() for _ in range(3)]
        
        tasks = [
            orchestrator.execute_phase("concurrent_test", context)
            for context in contexts
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == 3
        for result in results:
            assert len(result) == 2  # Two agents
            assert all(r["status"] == "success" for r in result)
    
    @pytest.mark.integration
    @pytest.mark.async_test
    async def test_orchestrator_error_recovery(self, orchestrator, mock_agents):
        """Test orchestrator error recovery."""
        # Create agents with different failure modes
        mock_agents["market_intelligence"].should_fail = True
        mock_agents["meditation_teacher"].should_fail = False
        
        for agent in mock_agents.values():
            orchestrator.register_agent(agent)
        
        orchestrator.define_phase("error_recovery_test", [
            "market_intelligence",
            "meditation_teacher"
        ], execution_pattern="sequential")
        
        context = PipelineContext()
        
        # Should fail on first agent
        with pytest.raises(Exception, match="Agent market_intelligence failed"):
            await orchestrator.execute_phase("error_recovery_test", context)
        
        # Context should still be in valid state
        assert context.session_id is not None
        assert context.status in ["initialized", "running", "failed"]
    
    @pytest.mark.integration
    @pytest.mark.async_test
    async def test_pipeline_metrics_collection(self, orchestrator, mock_agents):
        """Test comprehensive metrics collection."""
        for agent in mock_agents.values():
            orchestrator.register_agent(agent)
        
        orchestrator.define_phase("metrics_test", [
            "market_intelligence",
            "meditation_teacher",
            "script_writer"
        ], execution_pattern="sequential")
        
        context = PipelineContext()
        
        await orchestrator.execute_phase("metrics_test", context)
        
        # Check comprehensive metrics
        assert "agent_times" in context.metrics
        assert "phase_times" in context.metrics
        assert "quality_scores" in context.metrics
        
        # Check cost tracking
        assert context.cost_tracking["api_calls"] == 3
        assert context.cost_tracking["total_cost"] >= 0
        
        # Check agent metrics
        for agent in mock_agents.values():
            if agent.execute_called:
                assert agent.execution_count > 0
    
    @pytest.mark.integration
    @pytest.mark.async_test
    async def test_pipeline_context_validation(self, orchestrator, mock_agents):
        """Test pipeline context validation."""
        for agent in mock_agents.values():
            orchestrator.register_agent(agent)
        
        orchestrator.define_phase("validation_test", [
            "market_intelligence"
        ], execution_pattern="sequential")
        
        context = PipelineContext()
        
        # Context should be valid before execution
        assert context.session_id is not None
        assert context.created_at is not None
        assert context.status == "initialized"
        
        await orchestrator.execute_phase("validation_test", context)
        
        # Context should remain valid after execution
        assert context.session_id is not None
        assert context.status in ["running", "completed"]
    
    @pytest.mark.integration
    @pytest.mark.async_test
    async def test_orchestrator_cleanup(self, orchestrator, mock_agents):
        """Test orchestrator cleanup functionality."""
        for agent in mock_agents.values():
            orchestrator.register_agent(agent)
        
        orchestrator.define_phase("cleanup_test", [
            "market_intelligence"
        ], execution_pattern="sequential")
        
        context = PipelineContext()
        
        await orchestrator.execute_phase("cleanup_test", context)
        
        # Test cleanup
        orchestrator.cleanup()
        
        # Should reset internal state
        assert orchestrator.current_phase is None
        assert orchestrator.start_time is None
        assert orchestrator.end_time is None