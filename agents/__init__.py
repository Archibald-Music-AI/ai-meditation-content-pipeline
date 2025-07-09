"""
Enhanced Archibald v5.1 - Agent Package

This package contains all specialized agents for the meditation content pipeline.
Each agent is designed with single responsibility and well-defined interfaces
for seamless integration into the meta-orchestration system.

Available Agents:
- Agent 1: Market Intelligence (market research and trend analysis)
- Agent 2: Meditation Teacher (authentic meditation content creation)
- Agent 3: Script Writer (voice-optimized script generation)
- Agent 7: Audio Generation (text-to-speech synthesis)
- Agent 10: Video Assembly (video composition and encoding)
- Agent 11: Thumbnail Optimizer (A/B testing and optimization)
- Agent 12: YouTube Publisher (publication and SEO optimization)

Usage Example:
    from agents import MarketIntelligenceAgent, MeditationTeacherAgent
    
    # Initialize agents
    market_agent = MarketIntelligenceAgent()
    teacher_agent = MeditationTeacherAgent()
    
    # Execute pipeline
    context = {}
    context = await market_agent.run(context)
    context = await teacher_agent.run(context)
"""

from .base import BaseAgent, AgentConfig, AgentMetrics, AgentStatus
from .agent_1_market_intelligence import MarketIntelligenceAgent, ContentRecommendation
from .agent_2_meditation_teacher import MeditationTeacherAgent, MeditationSession
from .agent_3_script_writer import ScriptWriterAgent, VoiceScript
from .agent_7_audio_generation import AudioGenerationAgent, GeneratedAudio
from .agent_10_video_assembly import VideoAssemblyAgent, AssembledVideo
from .agent_11_thumbnail_optimizer import ThumbnailOptimizerAgent, ThumbnailSuite
from .agent_12_youtube_publisher import YouTubePublisherAgent, YouTubePublication

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentConfig", 
    "AgentMetrics",
    "AgentStatus",
    
    # Agent classes
    "MarketIntelligenceAgent",
    "MeditationTeacherAgent", 
    "ScriptWriterAgent",
    "AudioGenerationAgent",
    "VideoAssemblyAgent",
    "ThumbnailOptimizerAgent",
    "YouTubePublisherAgent",
    
    # Data classes
    "ContentRecommendation",
    "MeditationSession",
    "VoiceScript", 
    "GeneratedAudio",
    "AssembledVideo",
    "ThumbnailSuite",
    "YouTubePublication"
]

# Version info
__version__ = "5.1.0"
__author__ = "Enhanced Archibald Pipeline"
__description__ = "Specialized agents for AI-powered meditation content creation"