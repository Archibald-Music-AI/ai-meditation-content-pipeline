# 📋 Agent Specifications

## Overview

This document provides detailed specifications for each of the 12 agents in the meditation content generation pipeline.

## Agent Implementation Pattern

Each agent follows a consistent pattern:

```python
class BaseAgent:
    def __init__(self, api_client):
        self.api_client = api_client
        self.retry_policy = RetryPolicy()
        
    async def execute(self, context):
        """Execute agent's primary function"""
        try:
            result = await self._process(context)
            return self._validate_output(result)
        except Exception as e:
            return await self._handle_error(e, context)
```

## Phase 1: Content Creation Agents

### Agent 1: Market Intelligence Specialist

**Purpose**: Analyze current meditation and wellness trends to inform content creation.

**Implementation**:
```python
class MarketIntelligenceAgent:
    def __init__(self):
        self.client = PerplexityClient()
        
    async def analyze_trends(self, theme):
        prompt = f"""
        Research current trends in {theme} meditation:
        1. What are people struggling with?
        2. What solutions are trending?
        3. What language resonates?
        """
        
        response = await self.client.search(prompt)
        return self._extract_insights(response)
```

**Key Features**:
- Real-time trend analysis
- Competitor content awareness
- Audience pain point identification

### Agent 2: Meditation Teacher

**Purpose**: Create authentic meditation concepts based on market intelligence.

**Implementation**:
```python
class MeditationTeacherAgent:
    def __init__(self):
        self.client = OpenAIClient(model="gpt-4")
        
    async def create_concept(self, market_data):
        prompt = f"""
        Based on these trends: {market_data}
        Create a meditation concept that:
        1. Addresses current needs
        2. Uses proven techniques
        3. Feels authentic and accessible
        """
        
        return await self.client.generate(prompt)
```

**Key Features**:
- Ancient wisdom integration
- Modern problem solving
- Beginner-friendly approach

### Agent 3: Script Writer

**Purpose**: Transform concepts into complete meditation scripts.

**Implementation**:
```python
class ScriptWriterAgent:
    def __init__(self):
        self.client = AnthropicClient(model="claude-3-opus")
        
    async def write_script(self, concept, duration):
        prompt = f"""
        Write a {duration}-minute meditation script based on:
        {concept}
        
        Include:
        - Opening to settle listeners
        - Main practice guidance
        - Integration and closing
        """
        
        script = await self.client.generate(prompt)
        return self._format_for_narration(script)
```

**Key Features**:
- Natural pacing
- Clear instructions
- Emotional resonance

### Agent 4: Script Reviewer

**Purpose**: Analyze script quality and market potential.

**Implementation**:
```python
class ScriptReviewerAgent:
    def __init__(self):
        self.client = DeepSeekClient()
        
    async def review_script(self, script):
        analysis = await self.client.analyze({
            'script': script,
            'criteria': [
                'market_appeal',
                'clarity',
                'effectiveness',
                'duration_appropriateness'
            ]
        })
        
        return {
            'score': analysis.overall_score,
            'suggestions': analysis.improvements
        }
```

### Agent 5: Quality Polisher

**Purpose**: Enhance script while maintaining authentic voice.

**Features**:
- Gentle refinement
- Voice preservation
- Flow optimization

### Agent 6: TTS QA Specialist

**Purpose**: Prepare script for optimal voice generation.

**Key Optimizations**:
- SSML markup insertion
- Pause placement
- Pronunciation guides
- Segment boundaries

## Phase 2: Audio Generation

### Agent 7: Voice Generation Specialist

**Purpose**: Generate high-quality meditation narration.

**Implementation Details**:
```python
class VoiceGenerationAgent:
    def __init__(self):
        self.client = ElevenLabsClient()
        self.voice_id = "meditation_voice_id"
        
    async def generate_audio(self, script):
        # Split into segments for long content
        segments = self._split_script(script)
        
        audio_parts = []
        for segment in segments:
            audio = await self.client.text_to_speech(
                text=segment,
                voice_id=self.voice_id,
                settings={
                    'stability': 0.85,
                    'similarity_boost': 0.75,
                    'style': 0.15
                }
            )
            audio_parts.append(audio)
            
        return self._concatenate_audio(audio_parts)
```

**Optimization Strategies**:
- Segment-based processing for long content
- Voice consistency preservation
- Natural pacing control

## Phase 3: Background Assets

### Agent 8: Background Music Specialist

**Purpose**: Generate meditation-appropriate background music.

**Enhanced Prompting Strategy**:
```python
def create_music_prompt(theme):
    positive = f"peaceful ambient music for {theme} meditation"
    negative = "no percussion, no sudden changes, no harsh sounds"
    technical = "soft textures, gentle progression, 60-70 BPM"
    
    return f"{positive}. {technical}. Avoid: {negative}"
```

### Agent 9: Background Image Specialist

**Purpose**: Create visually calming meditation backgrounds.

**Image Generation Approach**:
- Nature-inspired themes
- Soft color palettes
- Minimal distractions
- HD resolution (1920x1080)

## Phase 4: Assembly & Optimization

### Agent 10: Video Assembly Specialist

**Purpose**: Combine all assets into professional meditation video.

**FFmpeg Pipeline**:
```bash
ffmpeg -i background.jpg -i narration.mp3 -i music.mp3 \
  -filter_complex "[1:a][2:a]amix=inputs=2:duration=first" \
  -c:v libx264 -c:a aac -shortest output.mp4
```

### Agent 11: Thumbnail Optimizer

**Purpose**: Create engaging, click-worthy thumbnails.

**Optimization Criteria**:
- Clear, readable text
- Calming visual elements
- Brand consistency
- Mobile optimization

## Phase 5: Publishing

### Agent 12: YouTube Publisher & Analytics

**Purpose**: Upload content and optimize for discovery.

**Key Features**:
- SEO-optimized titles and descriptions
- Relevant tag generation
- Optimal scheduling
- Analytics tracking setup

## Inter-Agent Communication

Agents communicate through a shared context object:

```python
class PipelineContext:
    def __init__(self):
        self.market_data = None
        self.concept = None
        self.script = None
        self.audio = None
        self.music = None
        self.image = None
        self.video = None
        self.metadata = {}
        
    def update(self, agent_name, result):
        # Store agent output for next phase
        setattr(self, agent_name, result)
```

## Performance Optimization

### Caching Strategy
- Cache expensive API calls
- Reuse successful outputs
- Implement smart invalidation

### Resource Management
- Connection pooling for APIs
- Memory-efficient file handling
- Cleanup after processing

---

[← Architecture](ARCHITECTURE.md) | [Setup Guide →](SETUP.md)