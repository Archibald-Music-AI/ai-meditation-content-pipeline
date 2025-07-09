"""
Agent 2: Meditation Teacher Agent

This agent creates authentic meditation concepts and experiences based on traditional
wisdom and modern neuroscience. It transforms market intelligence into pedagogically
sound meditation structures that are both effective and engaging.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import asyncio

from .base import BaseAgent, AgentConfig
from .agent_1_market_intelligence import ContentRecommendation


class MeditationStyle(Enum):
    """Enumeration of meditation styles supported by the system."""
    MINDFULNESS = "mindfulness"
    BODY_SCAN = "body_scan"
    LOVING_KINDNESS = "loving_kindness"
    BREATHWORK = "breathwork"
    VISUALIZATION = "visualization"
    MANTRA = "mantra"
    MOVEMENT = "movement"
    SOUND_HEALING = "sound_healing"


class DifficultyLevel(Enum):
    """Meditation difficulty levels for appropriate guidance."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


@dataclass
class MeditationTechnique:
    """Structured representation of a meditation technique."""
    name: str
    description: str
    duration_minutes: int
    instructions: List[str]
    breathing_pattern: Optional[str] = None
    posture_guidance: Optional[str] = None
    focus_points: List[str] = None
    common_challenges: List[str] = None
    benefits: List[str] = None


@dataclass
class MeditationSession:
    """Complete meditation session structure."""
    title: str
    style: MeditationStyle
    difficulty: DifficultyLevel
    duration_minutes: int
    introduction: str
    main_technique: MeditationTechnique
    transitions: List[str]
    closing: str
    key_phrases: List[str]
    pacing_notes: List[str]
    voice_guidance: Dict[str, str]
    therapeutic_benefits: List[str]
    scientific_backing: Optional[str] = None


class MeditationTeacherAgent(BaseAgent):
    """
    Agent 2: Meditation Teacher and Content Architect
    
    This agent serves as the spiritual and pedagogical backbone of the pipeline,
    creating authentic meditation experiences based on traditional wisdom and
    modern neuroscience research. It transforms market intelligence into
    structured, effective meditation sessions.
    
    Capabilities:
    - Traditional meditation knowledge synthesis
    - Modern neuroscience integration
    - Pedagogical structure creation
    - Therapeutic benefit optimization
    - Cultural sensitivity and authenticity
    - Progressive difficulty scaling
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Meditation Teacher Agent."""
        if config is None:
            config = AgentConfig(
                name="meditation_teacher",
                phase=1,
                timeout_seconds=180,
                max_retries=3,
                required_capabilities=[
                    "meditation_knowledge",
                    "pedagogical_design",
                    "therapeutic_guidance",
                    "cultural_authenticity",
                    "neuroscience_integration"
                ]
            )
        
        super().__init__(config)
        self.meditation_traditions = [
            "vipassana",
            "zen",
            "tibetan_buddhism",
            "hindu_meditation",
            "secular_mindfulness",
            "somatic_practices"
        ]
        
        self.therapeutic_frameworks = [
            "mindfulness_based_stress_reduction",
            "acceptance_commitment_therapy",
            "dialectical_behavior_therapy",
            "cognitive_behavioral_therapy"
        ]
        
    async def execute(self, context: Dict[str, Any]) -> MeditationSession:
        """
        Create authentic meditation session based on market intelligence.
        
        Args:
            context: Pipeline context containing market research data
            
        Returns:
            MeditationSession with complete session structure
        """
        self.logger.info("Designing meditation session architecture")
        
        # Extract market intelligence
        market_data = context.get("market_intelligence")
        if not market_data:
            raise ValueError("Market intelligence data required for meditation design")
        
        # Phase 1: Analyze therapeutic needs
        self.logger.info("Analyzing therapeutic requirements")
        therapeutic_needs = await self._analyze_therapeutic_needs(market_data)
        
        # Phase 2: Select appropriate meditation style
        self.logger.info("Selecting optimal meditation style")
        style = await self._select_meditation_style(market_data, therapeutic_needs)
        
        # Phase 3: Design core technique
        self.logger.info("Designing core meditation technique")
        technique = await self._design_meditation_technique(style, market_data)
        
        # Phase 4: Create session structure
        self.logger.info("Structuring complete meditation session")
        session = await self._create_session_structure(technique, market_data)
        
        # Phase 5: Add therapeutic elements
        self.logger.info("Integrating therapeutic benefits")
        session = await self._integrate_therapeutic_elements(session, therapeutic_needs)
        
        # Phase 6: Optimize for voice delivery
        self.logger.info("Optimizing for audio delivery")
        session = await self._optimize_for_voice_delivery(session)
        
        # Update metrics
        self.metrics.tokens_used = 1800  # Estimated for meditation design
        self.metrics.api_calls_made = 4
        self.metrics.estimated_cost = self.estimate_cost(self.metrics.tokens_used, "claude-3-opus")
        
        self.logger.info(f"Meditation session created: {session.title}")
        return session
    
    async def _analyze_therapeutic_needs(self, market_data: ContentRecommendation) -> Dict[str, Any]:
        """Analyze therapeutic requirements based on market intelligence."""
        await asyncio.sleep(0.3)
        
        # Map themes to therapeutic needs
        theme_mapping = {
            "sleep_meditation": {
                "primary_need": "sleep_quality",
                "secondary_needs": ["anxiety_reduction", "nervous_system_regulation"],
                "neuroscience_focus": "parasympathetic_activation",
                "therapeutic_approach": "somatic_relaxation"
            },
            "anxiety_relief": {
                "primary_need": "anxiety_management",
                "secondary_needs": ["emotional_regulation", "stress_reduction"],
                "neuroscience_focus": "amygdala_regulation",
                "therapeutic_approach": "mindfulness_based_intervention"
            },
            "stress_reduction": {
                "primary_need": "stress_management",
                "secondary_needs": ["resilience_building", "emotional_balance"],
                "neuroscience_focus": "cortisol_regulation",
                "therapeutic_approach": "breath_regulation"
            },
            "focus_enhancement": {
                "primary_need": "attention_training",
                "secondary_needs": ["cognitive_clarity", "mental_discipline"],
                "neuroscience_focus": "prefrontal_cortex_strengthening",
                "therapeutic_approach": "concentration_meditation"
            }
        }
        
        return theme_mapping.get(market_data.primary_theme, {
            "primary_need": "general_wellbeing",
            "secondary_needs": ["awareness", "presence"],
            "neuroscience_focus": "default_mode_network",
            "therapeutic_approach": "mindfulness_meditation"
        })
    
    async def _select_meditation_style(
        self, 
        market_data: ContentRecommendation, 
        therapeutic_needs: Dict[str, Any]
    ) -> MeditationStyle:
        """Select optimal meditation style based on needs and market data."""
        await asyncio.sleep(0.2)
        
        # Map therapeutic needs to meditation styles
        style_mapping = {
            "sleep_quality": MeditationStyle.BODY_SCAN,
            "anxiety_management": MeditationStyle.BREATHWORK,
            "stress_management": MeditationStyle.MINDFULNESS,
            "attention_training": MeditationStyle.MINDFULNESS,
            "emotional_regulation": MeditationStyle.LOVING_KINDNESS,
            "general_wellbeing": MeditationStyle.MINDFULNESS
        }
        
        primary_need = therapeutic_needs.get("primary_need", "general_wellbeing")
        return style_mapping.get(primary_need, MeditationStyle.MINDFULNESS)
    
    async def _design_meditation_technique(
        self, 
        style: MeditationStyle, 
        market_data: ContentRecommendation
    ) -> MeditationTechnique:
        """Design specific meditation technique based on style and requirements."""
        await asyncio.sleep(0.5)
        
        technique_templates = {
            MeditationStyle.MINDFULNESS: {
                "name": "Mindful Awareness Practice",
                "description": "A foundational mindfulness technique focusing on present-moment awareness",
                "instructions": [
                    "Begin by finding a comfortable seated position",
                    "Close your eyes gently or soften your gaze",
                    "Notice your breath without trying to change it",
                    "When thoughts arise, acknowledge them with kindness",
                    "Gently return attention to the breath",
                    "Expand awareness to include body sensations",
                    "Rest in open, spacious awareness"
                ],
                "breathing_pattern": "natural_breathing",
                "posture_guidance": "comfortable_seated_position",
                "focus_points": ["breath", "body_sensations", "thoughts", "emotions"],
                "common_challenges": ["restlessness", "sleepiness", "overthinking"],
                "benefits": ["stress_reduction", "emotional_regulation", "clarity"]
            },
            MeditationStyle.BODY_SCAN: {
                "name": "Progressive Body Awareness",
                "description": "A systematic exploration of physical sensations throughout the body",
                "instructions": [
                    "Lie down comfortably with arms at your sides",
                    "Begin with three deep, releasing breaths",
                    "Start awareness at the top of your head",
                    "Slowly move attention through each body part",
                    "Notice sensations without judgment",
                    "Allow each area to soften and release",
                    "Complete the scan from head to toes"
                ],
                "breathing_pattern": "slow_deep_breathing",
                "posture_guidance": "lying_down_position",
                "focus_points": ["physical_sensations", "tension_release", "body_awareness"],
                "common_challenges": ["falling_asleep", "physical_discomfort", "impatience"],
                "benefits": ["physical_relaxation", "sleep_improvement", "body_awareness"]
            },
            MeditationStyle.BREATHWORK: {
                "name": "Rhythmic Breath Regulation",
                "description": "A breathing technique designed to calm the nervous system",
                "instructions": [
                    "Sit comfortably with spine naturally straight",
                    "Place one hand on chest, one on belly",
                    "Inhale slowly through nose for 4 counts",
                    "Hold the breath gently for 4 counts",
                    "Exhale through mouth for 6 counts",
                    "Repeat this pattern with full attention",
                    "Allow the rhythm to become natural"
                ],
                "breathing_pattern": "4-4-6_breathing",
                "posture_guidance": "upright_seated_position",
                "focus_points": ["breath_rhythm", "nervous_system", "inner_calm"],
                "common_challenges": ["breath_holding", "dizziness", "forcing_rhythm"],
                "benefits": ["anxiety_reduction", "nervous_system_regulation", "mental_clarity"]
            }
        }
        
        template = technique_templates.get(style, technique_templates[MeditationStyle.MINDFULNESS])
        
        return MeditationTechnique(
            name=template["name"],
            description=template["description"],
            duration_minutes=market_data.target_duration // 60,
            instructions=template["instructions"],
            breathing_pattern=template["breathing_pattern"],
            posture_guidance=template["posture_guidance"],
            focus_points=template["focus_points"],
            common_challenges=template["common_challenges"],
            benefits=template["benefits"]
        )
    
    async def _create_session_structure(
        self, 
        technique: MeditationTechnique, 
        market_data: ContentRecommendation
    ) -> MeditationSession:
        """Create complete meditation session structure."""
        await asyncio.sleep(0.4)
        
        # Determine difficulty level
        difficulty = DifficultyLevel.BEGINNER  # Default for market accessibility
        
        # Create session structure
        session = MeditationSession(
            title=f"{technique.name} - {market_data.primary_theme.replace('_', ' ').title()}",
            style=MeditationStyle.MINDFULNESS,  # Example default
            difficulty=difficulty,
            duration_minutes=technique.duration_minutes,
            introduction=await self._create_introduction(technique, market_data),
            main_technique=technique,
            transitions=await self._create_transitions(technique),
            closing=await self._create_closing(technique),
            key_phrases=await self._generate_key_phrases(technique),
            pacing_notes=await self._create_pacing_notes(technique),
            voice_guidance=await self._create_voice_guidance(technique),
            therapeutic_benefits=technique.benefits
        )
        
        return session
    
    async def _create_introduction(self, technique: MeditationTechnique, market_data: ContentRecommendation) -> str:
        """Create welcoming introduction for the meditation session."""
        await asyncio.sleep(0.1)
        
        return f"""
        Welcome to this {technique.name.lower()} session. 
        
        In the next {technique.duration_minutes} minutes, we'll explore {technique.description.lower()}.
        
        This practice is designed to help you find peace and clarity in your daily life.
        
        Let's begin by settling into a comfortable position and preparing our mind and body for this journey inward.
        """
    
    async def _create_transitions(self, technique: MeditationTechnique) -> List[str]:
        """Create smooth transitions between meditation phases."""
        await asyncio.sleep(0.1)
        
        return [
            "Now, let's deepen our practice...",
            "As we continue, notice how your awareness naturally expands...",
            "In this moment, allow yourself to settle even more deeply...",
            "We're moving into the heart of our practice now..."
        ]
    
    async def _create_closing(self, technique: MeditationTechnique) -> str:
        """Create gentle closing for the meditation session."""
        await asyncio.sleep(0.1)
        
        return f"""
        As we come to the end of our {technique.name.lower()}, take a moment to appreciate this time you've given yourself.
        
        Notice any sense of calm or clarity that may have emerged.
        
        When you're ready, gently wiggle your fingers and toes, take a deep breath, and slowly open your eyes.
        
        Carry this sense of peace with you into your day.
        """
    
    async def _generate_key_phrases(self, technique: MeditationTechnique) -> List[str]:
        """Generate key phrases for emphasis during the session."""
        await asyncio.sleep(0.1)
        
        return [
            "breathe naturally",
            "notice without judgment",
            "return to the present moment",
            "allow whatever arises",
            "rest in awareness",
            "be gentle with yourself"
        ]
    
    async def _create_pacing_notes(self, technique: MeditationTechnique) -> List[str]:
        """Create pacing guidelines for optimal delivery."""
        await asyncio.sleep(0.1)
        
        return [
            "Speak 20% slower than normal conversation",
            "Leave 3-5 second pauses between instructions",
            "Use gentle, soothing tone throughout",
            "Emphasize key phrases with slight intonation",
            "Allow for natural breath rhythm between words"
        ]
    
    async def _create_voice_guidance(self, technique: MeditationTechnique) -> Dict[str, str]:
        """Create voice delivery guidance for optimal therapeutic effect."""
        await asyncio.sleep(0.1)
        
        return {
            "tone": "warm, calm, and nurturing",
            "pace": "slow and deliberate",
            "volume": "gentle and consistent",
            "inflection": "minimal, avoiding dramatic changes",
            "pauses": "strategic, allowing for integration",
            "breathing": "audible but not distracting"
        }
    
    async def _integrate_therapeutic_elements(
        self, 
        session: MeditationSession, 
        therapeutic_needs: Dict[str, Any]
    ) -> MeditationSession:
        """Integrate specific therapeutic elements based on needs."""
        await asyncio.sleep(0.2)
        
        # Add scientific backing
        session.scientific_backing = await self._add_scientific_backing(therapeutic_needs)
        
        # Enhance therapeutic benefits
        session.therapeutic_benefits.extend([
            "nervous_system_regulation",
            "emotional_balance",
            "cognitive_flexibility"
        ])
        
        return session
    
    async def _add_scientific_backing(self, therapeutic_needs: Dict[str, Any]) -> str:
        """Add scientific backing for the meditation approach."""
        await asyncio.sleep(0.1)
        
        return f"""
        This meditation is grounded in neuroscience research showing that regular practice
        can enhance {therapeutic_needs.get('neuroscience_focus', 'brain function')} and
        support {therapeutic_needs.get('primary_need', 'overall wellbeing')}.
        """
    
    async def _optimize_for_voice_delivery(self, session: MeditationSession) -> MeditationSession:
        """Optimize session structure for voice synthesis and delivery."""
        await asyncio.sleep(0.2)
        
        # Add pronunciation guides for complex terms
        session.voice_guidance["pronunciation_notes"] = {
            "mindfulness": "MIND-ful-ness",
            "awareness": "a-WARE-ness",
            "meditation": "med-i-TAY-shun"
        }
        
        # Add SSML-friendly pacing
        session.voice_guidance["ssml_tags"] = {
            "pause_short": "<break time='1s'/>",
            "pause_medium": "<break time='2s'/>",
            "pause_long": "<break time='3s'/>",
            "emphasis": "<emphasis level='moderate'>",
            "slow_speech": "<prosody rate='slow'>"
        }
        
        return session
    
    async def validate_output(self, output: MeditationSession) -> bool:
        """
        Validate meditation session quality and completeness.
        
        Args:
            output: MeditationSession to validate
            
        Returns:
            True if session meets quality standards
        """
        validation_checks = [
            output.title is not None and len(output.title) > 0,
            output.introduction is not None and len(output.introduction) > 50,
            output.main_technique is not None,
            len(output.main_technique.instructions) >= 5,
            output.closing is not None and len(output.closing) > 50,
            len(output.key_phrases) >= 3,
            len(output.therapeutic_benefits) >= 2,
            output.duration_minutes > 0,
            output.voice_guidance is not None
        ]
        
        validation_passed = all(validation_checks)
        
        if not validation_passed:
            self.logger.warning("Meditation session validation failed")
            
        return validation_passed
    
    def get_required_context_keys(self) -> List[str]:
        """Define required context keys for meditation session creation."""
        return ["market_intelligence"]
    
    def get_output_key(self) -> str:
        """Define output key for meditation session."""
        return "meditation_session"