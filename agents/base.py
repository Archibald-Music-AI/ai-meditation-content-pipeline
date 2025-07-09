"""
Base Agent Class for Enhanced Archibald v5.1 Pipeline

This module provides the foundational agent class that all specialized agents inherit from.
It implements common functionality including error handling, logging, cost tracking,
and standard interfaces for the meta-orchestration pattern.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import logging
from enum import Enum
import time


class AgentStatus(Enum):
    """Enumeration of possible agent states during execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class AgentMetrics:
    """Tracks performance metrics for individual agent executions."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time_seconds: float = 0.0
    api_calls_made: int = 0
    tokens_used: int = 0
    estimated_cost: float = 0.0
    retry_count: int = 0
    error_count: int = 0


@dataclass
class AgentConfig:
    """Configuration parameters for agent initialization."""
    name: str
    phase: int
    timeout_seconds: int = 300
    max_retries: int = 3
    retry_delay_seconds: int = 5
    enable_fallback: bool = True
    cost_limit: float = 1.0
    required_capabilities: List[str] = field(default_factory=list)


class BaseAgent(ABC):
    """
    Abstract base class for all pipeline agents.
    
    This class provides common functionality including:
    - Standardized execution flow with error handling
    - Automatic metric collection and cost tracking
    - Retry logic with exponential backoff
    - Validation interfaces for quality gates
    - Async support for parallel execution
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize the base agent with configuration.
        
        Args:
            config: AgentConfig object containing agent parameters
        """
        self.config = config
        self.metrics = AgentMetrics()
        self.status = AgentStatus.PENDING
        self.logger = self._setup_logger()
        self._error_history: List[Tuple[datetime, str]] = []
        
    def _setup_logger(self) -> logging.Logger:
        """Configure agent-specific logger with proper formatting."""
        logger = logging.getLogger(f"agent.{self.config.name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'[%(asctime)s] [{self.config.name}] %(levelname)s: %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method with comprehensive error handling and metrics.
        
        Args:
            context: Pipeline context containing data from previous agents
            
        Returns:
            Updated context with agent's output
            
        Raises:
            Exception: If agent fails after all retries
        """
        self.logger.info(f"Starting execution for phase {self.config.phase}")
        self.status = AgentStatus.RUNNING
        self.metrics.start_time = datetime.now()
        
        attempt = 0
        last_error = None
        
        while attempt <= self.config.max_retries:
            try:
                # Pre-execution validation
                self._validate_prerequisites(context)
                
                # Execute main agent logic
                result = await self._execute_with_timeout(context)
                
                # Validate output quality
                if not await self.validate_output(result):
                    raise ValueError("Output validation failed quality checks")
                
                # Update context with results
                context = self._update_context(context, result)
                
                # Mark successful completion
                self._finalize_success()
                return context
                
            except asyncio.TimeoutError:
                last_error = f"Execution timeout after {self.config.timeout_seconds}s"
                self.logger.warning(f"Attempt {attempt + 1} timed out")
                
            except Exception as e:
                last_error = str(e)
                self.logger.error(f"Attempt {attempt + 1} failed: {e}")
                self._error_history.append((datetime.now(), str(e)))
                
            # Retry logic with exponential backoff
            if attempt < self.config.max_retries:
                self.status = AgentStatus.RETRYING
                delay = self.config.retry_delay_seconds * (2 ** attempt)
                self.logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                self.metrics.retry_count += 1
                
            attempt += 1
        
        # All retries exhausted
        self._finalize_failure(last_error)
        raise Exception(f"Agent {self.config.name} failed after {attempt} attempts: {last_error}")
    
    async def _execute_with_timeout(self, context: Dict[str, Any]) -> Any:
        """Execute agent logic with timeout protection."""
        return await asyncio.wait_for(
            self.execute(context),
            timeout=self.config.timeout_seconds
        )
    
    def _validate_prerequisites(self, context: Dict[str, Any]) -> None:
        """
        Validate that required context data exists before execution.
        
        Args:
            context: Pipeline context to validate
            
        Raises:
            ValueError: If required data is missing
        """
        required_keys = self.get_required_context_keys()
        missing_keys = [key for key in required_keys if key not in context]
        
        if missing_keys:
            raise ValueError(f"Missing required context keys: {missing_keys}")
    
    def _update_context(self, context: Dict[str, Any], result: Any) -> Dict[str, Any]:
        """
        Update pipeline context with agent results.
        
        Args:
            context: Current pipeline context
            result: Agent execution result
            
        Returns:
            Updated context dictionary
        """
        output_key = self.get_output_key()
        context[output_key] = result
        
        # Add agent metrics to context for monitoring
        context[f"{self.config.name}_metrics"] = {
            "execution_time": self.metrics.execution_time_seconds,
            "cost": self.metrics.estimated_cost,
            "retries": self.metrics.retry_count
        }
        
        return context
    
    def _finalize_success(self) -> None:
        """Update metrics and status for successful completion."""
        self.status = AgentStatus.COMPLETED
        self.metrics.end_time = datetime.now()
        self.metrics.execution_time_seconds = (
            self.metrics.end_time - self.metrics.start_time
        ).total_seconds()
        
        self.logger.info(
            f"Completed successfully in {self.metrics.execution_time_seconds:.2f}s "
            f"(Cost: ${self.metrics.estimated_cost:.4f})"
        )
    
    def _finalize_failure(self, error: str) -> None:
        """Update metrics and status for failed execution."""
        self.status = AgentStatus.FAILED
        self.metrics.end_time = datetime.now()
        self.metrics.execution_time_seconds = (
            self.metrics.end_time - self.metrics.start_time
        ).total_seconds()
        self.metrics.error_count = len(self._error_history)
        
        self.logger.error(
            f"Failed after {self.metrics.retry_count} retries "
            f"in {self.metrics.execution_time_seconds:.2f}s: {error}"
        )
    
    def estimate_cost(self, tokens: int, model: str = "gpt-4") -> float:
        """
        Estimate API cost based on token usage.
        
        Args:
            tokens: Number of tokens used
            model: Model identifier for pricing
            
        Returns:
            Estimated cost in USD
        """
        # Mock pricing structure for demonstration
        pricing = {
            "gpt-4": 0.03 / 1000,  # per token
            "gpt-4-turbo": 0.01 / 1000,
            "claude-3-opus": 0.015 / 1000,
            "deepseek": 0.001 / 1000,
        }
        
        rate = pricing.get(model, 0.01 / 1000)
        return tokens * rate
    
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Any:
        """
        Main execution logic for the agent.
        
        This method must be implemented by all subclasses to define
        the specific behavior of each specialized agent.
        
        Args:
            context: Pipeline context with data from previous agents
            
        Returns:
            Agent-specific output to be added to context
        """
        pass
    
    @abstractmethod
    async def validate_output(self, output: Any) -> bool:
        """
        Validate the quality and completeness of agent output.
        
        This method implements agent-specific validation logic to ensure
        outputs meet quality standards before proceeding.
        
        Args:
            output: The output produced by execute()
            
        Returns:
            True if output passes validation, False otherwise
        """
        pass
    
    @abstractmethod
    def get_required_context_keys(self) -> List[str]:
        """
        Define required context keys this agent needs to execute.
        
        Returns:
            List of required context key names
        """
        pass
    
    @abstractmethod
    def get_output_key(self) -> str:
        """
        Define the context key where this agent stores its output.
        
        Returns:
            Context key name for agent output
        """
        pass
    
    def get_capabilities(self) -> List[str]:
        """
        Return list of capabilities this agent provides.
        
        Used by meta-orchestrator for intelligent task delegation.
        
        Returns:
            List of capability identifiers
        """
        return self.config.required_capabilities
    
    def can_handle_task(self, task_type: str) -> bool:
        """
        Check if this agent can handle a specific task type.
        
        Args:
            task_type: Type of task to check
            
        Returns:
            True if agent has the required capability
        """
        return task_type in self.get_capabilities()