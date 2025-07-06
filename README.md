# 🧘‍♂️ AI-Orchestrated Meditation Content Generation Pipeline

> A 12-agent AI system that automates meditation video production from concept to YouTube upload

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Architecture](https://img.shields.io/badge/agents-12-green.svg)](docs/ARCHITECTURE.md)

## 🎯 Project Overview

This project demonstrates a sophisticated multi-agent AI system that generates professional meditation content through coordinated orchestration of 12 specialized agents. Built as a real-world exploration of AI agent coordination, API integration, and content generation pipelines.

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 ENHANCED ARCHIBALD ORCHESTRATOR              │
├─────────────────────────────────────────────────────────────┤
│  Phase 1: Content Creation (15-20s)                         │
│  ├─ Agent 1: Market Intelligence (Perplexity API)           │
│  ├─ Agent 2: Meditation Teacher (GPT-4)                     │
│  ├─ Agent 3: Script Writer (Claude-3)                       │
│  ├─ Agent 4: Script Reviewer (DeepSeek)                     │
│  ├─ Agent 5: Quality Polisher (Claude-3)                    │
│  └─ Agent 6: TTS QA Specialist (GPT-4 Turbo)               │
├─────────────────────────────────────────────────────────────┤
│  Phase 2: Audio Generation (8-15s)                          │
│  └─ Agent 7: Voice Generation (ElevenLabs)                  │
├─────────────────────────────────────────────────────────────┤
│  Phase 3: Background Assets (5-8s)                          │
│  ├─ Agent 8: Music Generation (FAL AI)                      │
│  └─ Agent 9: Image Generation (FAL AI)                      │
├─────────────────────────────────────────────────────────────┤
│  Phase 4: Assembly & Optimization (5-8s)                    │
│  ├─ Agent 10: Video Assembly (FFmpeg)                       │
│  └─ Agent 11: Thumbnail Optimization (GPT-4 Vision)         │
├─────────────────────────────────────────────────────────────┤
│  Phase 5: Publishing & Analytics (3-5s)                     │
│  └─ Agent 12: YouTube Publisher (YouTube API + Analytics)   │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Technical Specifications

### System Capabilities
- **Pipeline Execution**: 8-12 minutes for complete meditation video
- **Agent Coordination**: 12 specialized AI agents across 5 phases
- **API Integrations**: OpenAI, Anthropic, ElevenLabs, FAL AI, Perplexity, YouTube
- **Production Output**: HD video with voice narration, background music, and visuals
- **Quality Control**: Manual checkpoints between phases for reliability

### Technology Stack
- **Languages**: Python 3.9+, Shell scripting
- **AI Models**: GPT-4, Claude-3, ElevenLabs voices, FAL AI media generation
- **Video/Audio**: FFmpeg for professional video assembly
- **Infrastructure**: Async/await architecture, API orchestration
- **Version Control**: Git with comprehensive documentation

## 🚀 Key Engineering Achievements

### 1. **Multi-Agent Orchestration System**
Designed and implemented a 12-agent system where each agent specializes in one aspect of content creation:
- Autonomous phase execution with quality checkpoints
- Intelligent error handling and retry mechanisms
- State management across agent boundaries
- API rate limit management across services

### 2. **Complex API Integration**
Successfully integrated 6+ different AI services into a cohesive pipeline:
- Standardized input/output formats across diverse APIs
- Error handling for various failure modes
- Cost optimization through intelligent API usage
- Consistent quality despite API variability

### 3. **Production Pipeline Architecture**
Built a 5-phase execution model that transforms ideas into published content:
- Phase isolation for reliability and debugging
- Parallel execution where possible (Phase 3)
- Quality gates between phases
- Comprehensive logging and monitoring

### 4. **Content Generation Automation**
Automated the traditionally manual process of meditation content creation:
- Market research → Script writing → Voice generation
- Background music and image creation
- Video assembly with professional standards
- Direct YouTube publishing with metadata

## 💼 Real-World Implementation

### Production Statistics
- **Successful Runs**: 6 complete production cycles
- **Content Generated**: Multiple meditation videos
- **Pipeline Reliability**: Consistent output quality
- **Cost Efficiency**: ~$0.40 per complete meditation video

### Technical Challenges Solved
- **Voice Consistency**: Maintaining coherent narration across segments
- **Audio Quality**: Enhanced prompting for meditation-appropriate music
- **Video Assembly**: Seamless integration of audio, video, and imagery
- **API Coordination**: Managing multiple external services reliably

## 🛠️ Technical Deep Dive

### Agent Communication Pattern
```python
class AgentOrchestrator:
    def __init__(self):
        self.agents = self._initialize_agents()
        self.phases = self._define_phases()
        
    async def execute_pipeline(self, request):
        """Execute 5-phase pipeline with checkpoints"""
        context = PipelineContext()
        
        for phase in self.phases:
            # Execute phase autonomously
            result = await self._execute_phase(phase, context)
            
            # Quality checkpoint (manual review point)
            if not self._quality_check(result):
                result = await self._handle_quality_issue(result)
            
            context.update(result)
            
        return context.final_output()
```

### API Integration Example
```python
async def generate_meditation_audio(script_segments):
    """Generate audio using ElevenLabs API"""
    audio_segments = []
    
    for segment in script_segments:
        try:
            audio = await elevenlabs_client.generate(
                text=segment,
                voice_id=MEDITATION_VOICE_ID,
                model="eleven_multilingual_v2"
            )
            audio_segments.append(audio)
        except APIError as e:
            # Retry logic with exponential backoff
            audio = await retry_with_backoff(segment)
            audio_segments.append(audio)
    
    return concatenate_audio(audio_segments)
```

## 🎓 Skills Demonstrated

- **System Design**: Multi-agent architecture with clear separation of concerns
- **API Integration**: Complex orchestration of external services
- **Async Programming**: Efficient handling of I/O-bound operations
- **Error Handling**: Robust recovery mechanisms for production reliability
- **Documentation**: Comprehensive technical documentation
- **Problem Solving**: Creative solutions for content generation challenges

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.9+
python --version

# FFmpeg for video processing
ffmpeg -version

# Required API Keys (add to .env)
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
ELEVENLABS_API_KEY=your_key
FAL_API_KEY=your_key
# ... see docs/SETUP.md for complete list
```

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/ai-meditation-pipeline.git
cd ai-meditation-pipeline

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

## 📖 Documentation

- [Architecture Overview](docs/ARCHITECTURE.md) - Detailed system design
- [Agent Specifications](docs/AGENTS.md) - Individual agent documentation
- [Setup Guide](docs/SETUP.md) - Complete installation instructions
- [API Integration](docs/API_INTEGRATION.md) - External service details

## 🤝 Project Context

This project was built to explore the possibilities of AI agent orchestration in content generation. It serves as a practical implementation of:
- Multi-agent system design patterns
- Complex API orchestration strategies
- Production pipeline architecture
- Content generation automation

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

**Note**: This is a portfolio demonstration of AI orchestration capabilities. API keys and proprietary content have been removed for public sharing.

[⭐ Star this repo](https://github.com/yourusername/ai-meditation-pipeline) | [💼 LinkedIn](https://linkedin.com/in/yourusername) | [📧 Contact](mailto:your.email@example.com)