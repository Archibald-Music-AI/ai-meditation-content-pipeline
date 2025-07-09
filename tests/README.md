# Enhanced Archibald v5.1 Test Suite

A comprehensive test suite demonstrating professional testing practices for the AI meditation pipeline system.

## Overview

This test suite provides complete coverage for the Enhanced Archibald v5.1 system, including:

- **Unit Tests**: Individual agent and component testing
- **Integration Tests**: Orchestrator and pipeline testing
- **Performance Tests**: Benchmarking and optimization validation
- **Mock Implementations**: Professional API mocking patterns
- **Async Testing**: Comprehensive async/await testing patterns

## Test Structure

```
tests/
├── conftest.py                          # Pytest fixtures and configuration
├── test_base_agent.py                   # Base agent functionality tests
├── test_agent_1_market_intelligence.py  # Market intelligence agent tests
├── test_agent_2_meditation_teacher.py   # Meditation teacher agent tests
├── test_agent_3_script_writer.py        # Script writer agent tests
├── test_agent_7_audio_generation.py     # Audio generation agent tests
├── test_agent_10_video_assembly.py      # Video assembly agent tests
├── test_agent_11_thumbnail_optimizer.py # Thumbnail optimizer agent tests
├── test_agent_12_youtube_publisher.py   # YouTube publisher agent tests
├── test_orchestrator.py                 # Orchestrator integration tests
└── README.md                           # This file
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_base_agent.py

# Run specific test class
pytest tests/test_base_agent.py::TestBaseAgent

# Run specific test method
pytest tests/test_base_agent.py::TestBaseAgent::test_agent_initialization
```

### Test Categories

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only performance tests
pytest -m performance

# Run only async tests
pytest -m async_test

# Skip slow tests
pytest -m "not slow"
```

### Advanced Test Options

```bash
# Run with coverage
pytest --cov=agents --cov=orchestrator --cov-report=html

# Run with parallel execution
pytest -n auto

# Run with verbose output
pytest -v

# Run with specific timeout
pytest --timeout=120

# Run with benchmark reporting
pytest --benchmark-only
```

## Test Features

### Professional Mocking

The test suite demonstrates advanced mocking patterns:

```python
# API mocking with realistic responses
@pytest.fixture
def mock_openai_response():
    return {
        "choices": [{"message": {"content": "Test response"}}],
        "usage": {"total_tokens": 100}
    }

# Async mocking
with patch('agents.agent_1.api_client') as mock_client:
    mock_client.generate = AsyncMock(return_value=expected_response)
    result = await agent.execute(context)
```

### Comprehensive Fixtures

Reusable test fixtures for common scenarios:

```python
@pytest.fixture
def mock_context():
    """Create a mock PipelineContext for testing."""
    context = PipelineContext()
    context.session_id = "test_session_123"
    context.data = {"test_key": "test_value"}
    return context

@pytest.fixture
def mock_external_apis():
    """Create mock external API clients."""
    return {
        "openai": MockOpenAI(),
        "elevenlabs": MockElevenLabs(),
        "youtube": MockYouTube()
    }
```

### Performance Benchmarking

Performance tests with realistic thresholds:

```python
@pytest.mark.performance
async def test_agent_performance(agent, context, performance_benchmark):
    benchmark = performance_benchmark
    benchmark.start()
    
    await agent.execute(context)
    
    benchmark.end()
    metrics = benchmark.get_metrics()
    
    # Performance assertions
    assert metrics["duration"] < 30.0  # Should complete within 30 seconds
    assert metrics["duration"] > 0
```

### Async Testing Patterns

Comprehensive async testing support:

```python
@pytest.mark.async_test
async def test_concurrent_execution(agent, context):
    """Test concurrent execution of multiple agents."""
    tasks = [agent.execute(context) for _ in range(3)]
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 3
    for result in results:
        assert result["status"] == "success"
```

## Test Categories

### Unit Tests (`-m unit`)

Test individual components in isolation:

- Agent initialization and configuration
- Input validation and error handling
- Output validation and quality metrics
- Cost tracking and performance metrics
- Retry logic and error recovery

### Integration Tests (`-m integration`)

Test component interactions:

- Agent communication through PipelineContext
- Phase execution and dependencies
- End-to-end pipeline workflows
- Quality gates and validation
- Context persistence across agents

### Performance Tests (`-m performance`)

Validate system performance:

- Execution time benchmarks
- Memory usage validation
- Throughput measurements
- Cost optimization validation
- Scalability testing

### Async Tests (`-m async_test`)

Test asynchronous functionality:

- Concurrent agent execution
- Parallel phase processing
- Async API interactions
- Timeout handling
- Error propagation in async contexts

## Quality Assurance

### Test Coverage

The test suite achieves comprehensive coverage:

```bash
# Generate coverage report
pytest --cov=agents --cov=orchestrator --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Code Quality

Tests demonstrate best practices:

- Clear test names and documentation
- Proper setup and teardown
- Realistic test data and scenarios
- Comprehensive error testing
- Performance validation

### Continuous Integration

The test suite is designed for CI/CD:

```bash
# CI-friendly test execution
pytest --timeout=300 --maxfail=5 --tb=short

# Generate reports for CI
pytest --junit-xml=test-results.xml --cov-report=xml
```

## Mock Implementations

### External API Mocking

Professional mocking of external services:

```python
class MockOpenAI:
    def __init__(self):
        self.chat = Mock()
        self.chat.completions = Mock()
        self.chat.completions.create = AsyncMock()

class MockElevenLabs:
    def __init__(self):
        self.generate = AsyncMock(return_value=b"mock audio data")

class MockYouTube:
    def __init__(self):
        self.upload = AsyncMock(return_value={"video_id": "test_id"})
```

### File System Mocking

Proper file system mocking:

```python
@pytest.fixture
def mock_file_system(tmp_path):
    """Create mock file system for testing."""
    audio_file = tmp_path / "test.mp3"
    video_file = tmp_path / "test.mp4"
    
    audio_file.write_text("mock audio content")
    video_file.write_text("mock video content")
    
    return {
        "audio_file": str(audio_file),
        "video_file": str(video_file)
    }
```

## Error Testing

Comprehensive error handling tests:

```python
@pytest.mark.unit
async def test_api_error_handling(agent, context):
    """Test API error handling."""
    with patch('agent.api_client') as mock_client:
        mock_client.generate.side_effect = Exception("API Error")
        
        with pytest.raises(Exception, match="API Error"):
            await agent.execute(context)
```

## Best Practices Demonstrated

1. **Comprehensive Coverage**: Tests cover happy paths, error cases, and edge conditions
2. **Professional Mocking**: Realistic mock implementations that mirror actual API behavior
3. **Async Testing**: Proper async/await testing patterns with timeout handling
4. **Performance Validation**: Benchmark tests that validate system performance characteristics
5. **Quality Metrics**: Tests that validate output quality and system reliability
6. **CI/CD Ready**: Test configuration suitable for continuous integration
7. **Clear Documentation**: Well-documented test cases with clear purpose and expectations

## Contributing

When adding new tests:

1. Follow the existing naming conventions
2. Use appropriate test markers (`@pytest.mark.unit`, etc.)
3. Include both success and failure scenarios
4. Add performance benchmarks for critical paths
5. Mock external dependencies appropriately
6. Update this documentation as needed

## Configuration

Test configuration is managed through:

- `pytest.ini`: Main pytest configuration
- `conftest.py`: Shared fixtures and utilities
- `requirements-test.txt`: Testing dependencies

The test suite is designed to be a showcase of professional testing practices suitable for a portfolio repository while providing comprehensive validation of the Enhanced Archibald v5.1 system.