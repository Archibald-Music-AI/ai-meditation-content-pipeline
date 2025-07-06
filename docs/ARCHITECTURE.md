# 🏗️ System Architecture

## Overview

The AI Meditation Pipeline is a multi-agent system that coordinates 12 specialized AI agents across 5 execution phases to generate meditation content from concept to published video.

## Core Design Principles

### 1. **Agent Specialization**
Each agent has a single, well-defined responsibility:
- Improves reliability through focused functionality
- Enables independent testing and optimization
- Allows for easy agent replacement or upgrades

### 2. **Phase Isolation**
5 distinct phases with clear boundaries:
- **Phase 1**: Content Creation (sequential execution)
- **Phase 2**: Audio Generation (dedicated processing)
- **Phase 3**: Background Assets (parallel execution)
- **Phase 4**: Assembly & Optimization
- **Phase 5**: Publishing & Analytics

### 3. **Quality Checkpoints**
Manual review points between phases ensure:
- Content quality before expensive operations
- Early detection of issues
- Flexibility to adjust before proceeding

## Detailed Agent Specifications

### Phase 1: Content Creation Agents

#### Agent 1: Market Intelligence Specialist
- **API**: Perplexity
- **Purpose**: Research current meditation trends
- **Output**: Market insights and trending topics
- **Execution Time**: ~11 seconds

#### Agent 2: Meditation Teacher
- **API**: OpenAI GPT-4
- **Purpose**: Create meditation concept from market data
- **Output**: Meditation theme and approach
- **Execution Time**: ~4 seconds

#### Agent 3: Script Writer
- **API**: Anthropic Claude-3
- **Purpose**: Generate full meditation script
- **Output**: Complete narration text
- **Execution Time**: ~27 seconds

#### Agent 4: Script Reviewer
- **API**: DeepSeek
- **Purpose**: Analyze script quality and market fit
- **Output**: Quality assessment and suggestions
- **Execution Time**: ~10 seconds

#### Agent 5: Quality Polisher
- **API**: Anthropic Claude-3
- **Purpose**: Enhance script while preserving voice
- **Output**: Polished final script
- **Execution Time**: ~22 seconds

#### Agent 6: TTS QA Specialist
- **API**: OpenAI GPT-4 Turbo
- **Purpose**: Optimize script for voice generation
- **Output**: TTS-ready script with SSML markup
- **Execution Time**: ~5 seconds

### Phase 2: Audio Generation

#### Agent 7: Voice Generation Specialist
- **API**: ElevenLabs
- **Purpose**: Generate meditation narration
- **Features**: 
  - Custom voice models
  - Meditation-optimized pacing
  - Segment-based generation for long content
- **Output**: MP3 audio file
- **Execution Time**: 8-15 seconds

### Phase 3: Background Assets (Parallel)

#### Agent 8: Background Music Specialist
- **API**: FAL AI
- **Purpose**: Generate meditation-appropriate music
- **Features**:
  - Enhanced prompting for quality
  - Multiple model fallbacks
  - Theme-responsive generation
- **Output**: Background music file

#### Agent 9: Background Image Specialist
- **API**: FAL AI (Flux)
- **Purpose**: Create meditation visuals
- **Features**:
  - HD resolution (1920x1080)
  - Theme-aligned imagery
  - Meditation-optimized aesthetics
- **Output**: Background image

### Phase 4: Assembly & Optimization

#### Agent 10: Video Assembly Specialist
- **Technology**: FFmpeg
- **Purpose**: Combine all assets into final video
- **Features**:
  - Professional audio mixing
  - HD video encoding
  - Intro/outro integration
- **Output**: MP4 video file

#### Agent 11: Thumbnail Optimizer
- **API**: OpenAI GPT-4 Vision
- **Purpose**: Create clickable thumbnail
- **Features**:
  - CTR optimization
  - Brand consistency
  - A/B test variations
- **Output**: Thumbnail image

### Phase 5: Publishing

#### Agent 12: YouTube Publisher & Analytics
- **API**: YouTube Data API v3
- **Purpose**: Upload and optimize for YouTube
- **Features**:
  - Metadata optimization
  - SEO enhancement
  - Analytics integration
  - Scheduling capabilities
- **Output**: Published video URL

## Data Flow

```
User Request → Phase 1 → [QA Gate] → Phase 2 → [QA Gate] → 
Phase 3 → [QA Gate] → Phase 4 → [QA Gate] → Phase 5 → Published Content
```

Each phase passes a context object containing:
- Generated content from previous phases
- Metadata and configuration
- Quality scores and validation results
- Error logs and retry information

## Error Handling Strategy

### Agent-Level
- Exponential backoff for API failures
- Fallback options for critical agents
- Detailed error logging for debugging

### Phase-Level
- Phase rollback capabilities
- Partial retry mechanisms
- Manual intervention points

### System-Level
- Comprehensive monitoring
- Cost tracking and alerts
- Performance metrics collection

## Scalability Considerations

### Current Limitations
- Sequential execution in some phases
- API rate limits
- Manual quality checkpoints

### Future Enhancements
- Batch processing capabilities
- Agent parallelization
- Automated quality validation
- Multi-pipeline concurrency

---

[← Back to README](../README.md) | [Agent Specifications →](AGENTS.md)