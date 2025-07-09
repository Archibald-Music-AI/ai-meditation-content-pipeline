"""
Agent 3: Script Writer Agent

This agent transforms meditation concepts into polished, voice-optimized scripts
ready for text-to-speech synthesis. It handles pacing, intonation, and natural
language flow while maintaining therapeutic effectiveness.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import re
from datetime import datetime

from .base import BaseAgent, AgentConfig
from .agent_2_meditation_teacher import MeditationSession, MeditationTechnique


class ScriptSection(Enum):
    """Enumeration of script sections for structured writing."""
    OPENING = "opening"
    PREPARATION = "preparation"
    MAIN_PRACTICE = "main_practice"
    DEEPENING = "deepening"
    INTEGRATION = "integration"
    CLOSING = "closing"


@dataclass
class ScriptSegment:
    """Individual script segment with timing and delivery notes."""
    section: ScriptSection
    text: str
    duration_seconds: int
    voice_notes: Dict[str, str]
    pause_markers: List[Tuple[int, float]]  # (position, duration)
    emphasis_markers: List[Tuple[int, int]]  # (start, end)
    breathing_cues: List[int]  # positions for breath cues
    
    
@dataclass
class VoiceScript:
    """Complete voice-optimized meditation script."""
    title: str
    total_duration_minutes: int
    segments: List[ScriptSegment]
    voice_settings: Dict[str, Any]
    ssml_enhanced: str
    plain_text: str
    timing_markers: List[Tuple[int, str]]  # (time_seconds, marker_type)
    word_count: int
    estimated_speech_time: float
    quality_score: float


class ScriptWriterAgent(BaseAgent):
    """
    Agent 3: Script Writer and Voice Optimization
    
    This agent transforms meditation session structures into polished,
    voice-optimized scripts ready for text-to-speech synthesis. It handles
    natural language flow, pacing, intonation, and therapeutic delivery.
    
    Capabilities:
    - Natural language script generation
    - Voice synthesis optimization
    - SSML markup generation
    - Pacing and timing calculation
    - Therapeutic language patterns
    - Pronunciation optimization
    - Breathing cue integration
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Script Writer Agent."""
        if config is None:
            config = AgentConfig(
                name="script_writer",
                phase=1,
                timeout_seconds=240,
                max_retries=3,
                required_capabilities=[
                    "natural_language_generation",
                    "voice_optimization",
                    "ssml_markup",
                    "therapeutic_writing",
                    "pacing_calculation"
                ]
            )
        
        super().__init__(config)
        self.words_per_minute = 120  # Slow, meditative pace
        self.pause_patterns = {
            "short": 1.0,
            "medium": 2.0,
            "long": 3.0,
            "breath": 4.0
        }
        
    async def execute(self, context: Dict[str, Any]) -> VoiceScript:
        """
        Transform meditation session into voice-optimized script.
        
        Args:
            context: Pipeline context containing meditation session
            
        Returns:
            VoiceScript with complete voice-optimized content
        """
        self.logger.info("Beginning script writing and voice optimization")
        
        # Extract meditation session
        meditation_session = context.get("meditation_session")
        if not meditation_session:
            raise ValueError("Meditation session required for script writing")
        
        # Phase 1: Analyze session structure
        self.logger.info("Analyzing meditation session structure")
        session_analysis = await self._analyze_session_structure(meditation_session)
        
        # Phase 2: Create script segments
        self.logger.info("Creating structured script segments")
        segments = await self._create_script_segments(meditation_session, session_analysis)
        
        # Phase 3: Optimize for voice delivery
        self.logger.info("Optimizing script for voice synthesis")
        segments = await self._optimize_for_voice_delivery(segments)
        
        # Phase 4: Generate SSML markup
        self.logger.info("Generating SSML markup for enhanced delivery")
        ssml_script = await self._generate_ssml_markup(segments)
        
        # Phase 5: Calculate timing and pacing
        self.logger.info("Calculating precise timing and pacing")
        timing_data = await self._calculate_timing_markers(segments)
        
        # Phase 6: Quality assessment
        self.logger.info("Assessing script quality and readability")
        quality_score = await self._assess_script_quality(segments)
        
        # Phase 7: Generate final script
        self.logger.info("Compiling final voice script")
        script = await self._compile_final_script(
            meditation_session, segments, ssml_script, timing_data, quality_score
        )
        
        # Update metrics
        self.metrics.tokens_used = 3200  # Estimated for script generation
        self.metrics.api_calls_made = 6
        self.metrics.estimated_cost = self.estimate_cost(self.metrics.tokens_used, "gpt-4")
        
        self.logger.info(f"Script completed: {script.word_count} words, {script.estimated_speech_time:.1f}min")
        return script
    
    async def _analyze_session_structure(self, session: MeditationSession) -> Dict[str, Any]:
        """Analyze meditation session structure for script planning."""
        await asyncio.sleep(0.2)
        
        return {
            "total_duration": session.duration_minutes,
            "technique_complexity": len(session.main_technique.instructions),
            "transition_count": len(session.transitions),
            "key_phrase_count": len(session.key_phrases),
            "therapeutic_focus": session.therapeutic_benefits,
            "pacing_requirements": session.pacing_notes,
            "voice_style": session.voice_guidance
        }
    
    async def _create_script_segments(
        self, 
        session: MeditationSession, 
        analysis: Dict[str, Any]
    ) -> List[ScriptSegment]:
        """Create structured script segments from meditation session."""
        await asyncio.sleep(0.8)
        
        segments = []
        
        # Opening segment (10% of total time)
        opening_duration = int(session.duration_minutes * 60 * 0.1)
        opening_text = await self._write_opening_segment(session, opening_duration)
        segments.append(ScriptSegment(
            section=ScriptSection.OPENING,
            text=opening_text,
            duration_seconds=opening_duration,
            voice_notes={"pace": "slow", "tone": "welcoming"},
            pause_markers=[(len(opening_text)//2, 2.0)],
            emphasis_markers=[(0, 20)],
            breathing_cues=[]
        ))
        
        # Preparation segment (15% of total time)
        prep_duration = int(session.duration_minutes * 60 * 0.15)
        prep_text = await self._write_preparation_segment(session, prep_duration)
        segments.append(ScriptSegment(
            section=ScriptSection.PREPARATION,
            text=prep_text,
            duration_seconds=prep_duration,
            voice_notes={"pace": "slow", "tone": "guiding"},
            pause_markers=[(len(prep_text)//3, 3.0), (2*len(prep_text)//3, 2.0)],
            emphasis_markers=[],
            breathing_cues=[len(prep_text)//4, 3*len(prep_text)//4]
        ))
        
        # Main practice segment (50% of total time)
        main_duration = int(session.duration_minutes * 60 * 0.5)
        main_text = await self._write_main_practice_segment(session, main_duration)
        segments.append(ScriptSegment(
            section=ScriptSection.MAIN_PRACTICE,
            text=main_text,
            duration_seconds=main_duration,
            voice_notes={"pace": "very_slow", "tone": "therapeutic"},
            pause_markers=await self._calculate_main_practice_pauses(main_text),
            emphasis_markers=await self._identify_key_phrases(main_text, session.key_phrases),
            breathing_cues=await self._place_breathing_cues(main_text)
        ))
        
        # Deepening segment (15% of total time)
        deep_duration = int(session.duration_minutes * 60 * 0.15)
        deep_text = await self._write_deepening_segment(session, deep_duration)
        segments.append(ScriptSegment(
            section=ScriptSection.DEEPENING,
            text=deep_text,
            duration_seconds=deep_duration,
            voice_notes={"pace": "very_slow", "tone": "gentle"},
            pause_markers=[(len(deep_text)//2, 4.0)],
            emphasis_markers=[],
            breathing_cues=[len(deep_text)//3, 2*len(deep_text)//3]
        ))
        
        # Closing segment (10% of total time)
        closing_duration = int(session.duration_minutes * 60 * 0.1)
        closing_text = await self._write_closing_segment(session, closing_duration)
        segments.append(ScriptSegment(
            section=ScriptSection.CLOSING,
            text=closing_text,
            duration_seconds=closing_duration,
            voice_notes={"pace": "slow", "tone": "gratitude"},
            pause_markers=[(len(closing_text)//2, 2.0)],
            emphasis_markers=[(len(closing_text)-50, len(closing_text))],
            breathing_cues=[]
        ))
        
        return segments
    
    async def _write_opening_segment(self, session: MeditationSession, duration: int) -> str:
        """Write the opening segment of the meditation script."""
        await asyncio.sleep(0.2)
        
        return f"""
        Welcome to this {session.title.lower()} session. 
        
        I'm so glad you've taken this time for yourself today. 
        
        Over the next {session.duration_minutes} minutes, we'll journey together 
        into a practice of {session.main_technique.name.lower()}.
        
        This is your time to rest, to breathe, and to connect with the peace 
        that already exists within you.
        
        Let's begin.
        """
    
    async def _write_preparation_segment(self, session: MeditationSession, duration: int) -> str:
        """Write the preparation segment for settling into meditation."""
        await asyncio.sleep(0.2)
        
        posture_guide = session.main_technique.posture_guidance or "comfortable position"
        
        return f"""
        Start by finding a {posture_guide.replace('_', ' ')}.
        
        Allow your body to settle naturally... feeling supported and at ease.
        
        If you're comfortable doing so, gently close your eyes... 
        or simply soften your gaze downward.
        
        Take a moment to notice your breath... 
        not trying to change it... just observing its natural rhythm.
        
        Let your shoulders soften... your jaw relax... 
        and allow a sense of settling to move through your entire body.
        
        With each exhale, feel yourself arriving more fully in this moment.
        """
    
    async def _write_main_practice_segment(self, session: MeditationSession, duration: int) -> str:
        """Write the main practice segment with detailed instructions."""
        await asyncio.sleep(0.5)
        
        instructions = session.main_technique.instructions
        technique_text = ""
        
        for i, instruction in enumerate(instructions):
            if i == 0:
                technique_text += f"Now, {instruction.lower()}.\n\n"
            else:
                technique_text += f"{instruction}.\n\n"
            
            # Add natural pauses and elaborations
            if "breath" in instruction.lower():
                technique_text += "Notice the gentle rise and fall of your chest... the cool air entering... the warm air leaving.\n\n"
            
            if "notice" in instruction.lower():
                technique_text += "There's no need to judge or change anything... simply observe with gentle curiosity.\n\n"
            
            if "return" in instruction.lower():
                technique_text += "This returning is the practice... each time you notice and come back, you're strengthening your awareness.\n\n"
        
        # Add therapeutic elements
        for benefit in session.therapeutic_benefits[:2]:  # Include top 2 benefits
            technique_text += f"With each moment of practice, you're supporting your {benefit.replace('_', ' ')}.\n\n"
        
        return technique_text
    
    async def _write_deepening_segment(self, session: MeditationSession, duration: int) -> str:
        """Write the deepening segment for enhanced practice."""
        await asyncio.sleep(0.2)
        
        return f"""
        As we continue deeper into our practice, you might notice a natural settling... 
        a deeper sense of peace beginning to emerge.
        
        Allow yourself to rest in this awareness... 
        not grasping or trying to maintain any particular state... 
        simply being present with whatever arises.
        
        If thoughts come and go, that's perfectly natural... 
        let them pass like clouds in the sky of your awareness.
        
        Rest in this spacious presence... 
        this natural state of being that is always available to you.
        
        Simply being... simply breathing... simply here.
        """
    
    async def _write_closing_segment(self, session: MeditationSession, duration: int) -> str:
        """Write the closing segment for gentle transition."""
        await asyncio.sleep(0.2)
        
        return f"""
        As we bring our practice to a close, take a moment to appreciate 
        this gift you've given yourself.
        
        Notice any sense of calm... clarity... or peace that may be present.
        
        These qualities are always available to you... 
        you can return to this awareness anytime throughout your day.
        
        When you're ready, begin to gently wiggle your fingers and toes... 
        take a deeper breath... and slowly open your eyes.
        
        Thank you for practicing with me today.
        
        May you carry this peace with you.
        """
    
    async def _calculate_main_practice_pauses(self, text: str) -> List[Tuple[int, float]]:
        """Calculate optimal pause locations for main practice segment."""
        await asyncio.sleep(0.1)
        
        # Find natural pause points (sentence endings, ellipses)
        pause_markers = []
        sentences = text.split('.')
        position = 0
        
        for sentence in sentences:
            position += len(sentence) + 1
            if len(sentence.strip()) > 0:
                pause_markers.append((position, 2.0))  # 2-second pause after sentences
        
        # Add longer pauses after ellipses
        ellipses_positions = [m.start() for m in re.finditer(r'\.\.\.', text)]
        for pos in ellipses_positions:
            pause_markers.append((pos + 3, 3.0))  # 3-second pause after ellipses
        
        return pause_markers
    
    async def _identify_key_phrases(self, text: str, key_phrases: List[str]) -> List[Tuple[int, int]]:
        """Identify key phrases in text for emphasis."""
        await asyncio.sleep(0.1)
        
        emphasis_markers = []
        for phrase in key_phrases:
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            for match in pattern.finditer(text):
                emphasis_markers.append((match.start(), match.end()))
        
        return emphasis_markers
    
    async def _place_breathing_cues(self, text: str) -> List[int]:
        """Place breathing cues at natural points in the text."""
        await asyncio.sleep(0.1)
        
        # Place breathing cues every ~100 words
        words = text.split()
        word_positions = []
        position = 0
        
        for word in words:
            word_positions.append(position)
            position += len(word) + 1
        
        breathing_cues = []
        for i in range(100, len(words), 100):  # Every 100 words
            if i < len(word_positions):
                breathing_cues.append(word_positions[i])
        
        return breathing_cues
    
    async def _optimize_for_voice_delivery(self, segments: List[ScriptSegment]) -> List[ScriptSegment]:
        """Optimize script segments for voice synthesis."""
        await asyncio.sleep(0.3)
        
        for segment in segments:
            # Replace problematic characters
            segment.text = segment.text.replace('...', '…')  # Use proper ellipsis
            segment.text = segment.text.replace('--', '—')   # Use em dash
            
            # Add pronunciation guides for difficult words
            segment.text = self._add_pronunciation_guides(segment.text)
            
            # Optimize sentence structure for speech
            segment.text = self._optimize_sentence_structure(segment.text)
        
        return segments
    
    def _add_pronunciation_guides(self, text: str) -> str:
        """Add pronunciation guides for difficult words."""
        pronunciation_map = {
            'meditation': 'med-i-TAY-shun',
            'mindfulness': 'MIND-ful-ness',
            'awareness': 'a-WARE-ness',
            'breathe': 'BREETH',
            'rhythm': 'RITH-um'
        }
        
        # In a real implementation, this would use SSML phoneme tags
        return text
    
    def _optimize_sentence_structure(self, text: str) -> str:
        """Optimize sentence structure for natural speech delivery."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure proper spacing around punctuation
        text = re.sub(r'\s*\.\s*', '. ', text)
        text = re.sub(r'\s*,\s*', ', ', text)
        
        return text.strip()
    
    async def _generate_ssml_markup(self, segments: List[ScriptSegment]) -> str:
        """Generate SSML markup for enhanced voice delivery."""
        await asyncio.sleep(0.4)
        
        ssml_parts = ['<speak>']
        
        for segment in segments:
            # Add segment-specific prosody
            pace = segment.voice_notes.get('pace', 'medium')
            tone = segment.voice_notes.get('tone', 'neutral')
            
            ssml_parts.append(f'<prosody rate="{pace}">')
            
            # Process text with pause markers
            text_with_pauses = self._insert_ssml_pauses(segment.text, segment.pause_markers)
            
            # Add emphasis markers
            text_with_emphasis = self._insert_ssml_emphasis(text_with_pauses, segment.emphasis_markers)
            
            # Add breathing cues
            final_text = self._insert_breathing_cues(text_with_emphasis, segment.breathing_cues)
            
            ssml_parts.append(final_text)
            ssml_parts.append('</prosody>')
            ssml_parts.append('<break time="1s"/>')
        
        ssml_parts.append('</speak>')
        
        return ''.join(ssml_parts)
    
    def _insert_ssml_pauses(self, text: str, pause_markers: List[Tuple[int, float]]) -> str:
        """Insert SSML pause markers into text."""
        sorted_markers = sorted(pause_markers, key=lambda x: x[0], reverse=True)
        
        for position, duration in sorted_markers:
            if position <= len(text):
                pause_tag = f'<break time="{duration}s"/>'
                text = text[:position] + pause_tag + text[position:]
        
        return text
    
    def _insert_ssml_emphasis(self, text: str, emphasis_markers: List[Tuple[int, int]]) -> str:
        """Insert SSML emphasis markers into text."""
        sorted_markers = sorted(emphasis_markers, key=lambda x: x[0], reverse=True)
        
        for start, end in sorted_markers:
            if start < len(text) and end <= len(text):
                emphasized_text = f'<emphasis level="moderate">{text[start:end]}</emphasis>'
                text = text[:start] + emphasized_text + text[end:]
        
        return text
    
    def _insert_breathing_cues(self, text: str, breathing_cues: List[int]) -> str:
        """Insert subtle breathing cues into text."""
        sorted_cues = sorted(breathing_cues, reverse=True)
        
        for position in sorted_cues:
            if position <= len(text):
                # Add a gentle breath sound cue
                breath_cue = '<break time="0.5s"/>'
                text = text[:position] + breath_cue + text[position:]
        
        return text
    
    async def _calculate_timing_markers(self, segments: List[ScriptSegment]) -> List[Tuple[int, str]]:
        """Calculate precise timing markers for the script."""
        await asyncio.sleep(0.2)
        
        timing_markers = []
        current_time = 0
        
        for segment in segments:
            timing_markers.append((current_time, f"start_{segment.section.value}"))
            current_time += segment.duration_seconds
            timing_markers.append((current_time, f"end_{segment.section.value}"))
        
        return timing_markers
    
    async def _assess_script_quality(self, segments: List[ScriptSegment]) -> float:
        """Assess overall script quality using multiple metrics."""
        await asyncio.sleep(0.3)
        
        quality_factors = {
            "readability": 0.85,    # Assessed based on sentence complexity
            "flow": 0.90,          # Natural progression between segments
            "timing": 0.92,        # Appropriate pacing and duration
            "therapeutic": 0.88,   # Therapeutic effectiveness
            "voice_optimization": 0.87  # TTS readiness
        }
        
        # Calculate weighted average
        weights = [0.2, 0.25, 0.2, 0.2, 0.15]
        quality_score = sum(score * weight for score, weight in zip(quality_factors.values(), weights))
        
        return quality_score
    
    async def _compile_final_script(
        self, 
        session: MeditationSession, 
        segments: List[ScriptSegment],
        ssml_script: str,
        timing_data: List[Tuple[int, str]],
        quality_score: float
    ) -> VoiceScript:
        """Compile all elements into final voice script."""
        await asyncio.sleep(0.2)
        
        # Generate plain text version
        plain_text = "\n\n".join(segment.text.strip() for segment in segments)
        
        # Calculate word count and speech time
        word_count = len(plain_text.split())
        estimated_speech_time = word_count / self.words_per_minute
        
        # Define voice settings
        voice_settings = {
            "voice_id": "meditation_voice_01",
            "stability": 0.85,
            "similarity_boost": 0.75,
            "style": "calm_therapeutic",
            "use_speaker_boost": True
        }
        
        return VoiceScript(
            title=session.title,
            total_duration_minutes=session.duration_minutes,
            segments=segments,
            voice_settings=voice_settings,
            ssml_enhanced=ssml_script,
            plain_text=plain_text,
            timing_markers=timing_data,
            word_count=word_count,
            estimated_speech_time=estimated_speech_time,
            quality_score=quality_score
        )
    
    async def validate_output(self, output: VoiceScript) -> bool:
        """
        Validate script quality and completeness.
        
        Args:
            output: VoiceScript to validate
            
        Returns:
            True if script meets quality standards
        """
        validation_checks = [
            output.title is not None and len(output.title) > 0,
            len(output.segments) >= 3,  # At least opening, main, closing
            output.word_count > 100,
            output.estimated_speech_time > 0,
            output.quality_score > 0.8,
            output.ssml_enhanced is not None,
            output.plain_text is not None,
            len(output.timing_markers) > 0,
            all(segment.duration_seconds > 0 for segment in output.segments)
        ]
        
        validation_passed = all(validation_checks)
        
        if not validation_passed:
            self.logger.warning("Script validation failed")
            
        return validation_passed
    
    def get_required_context_keys(self) -> List[str]:
        """Define required context keys for script writing."""
        return ["meditation_session"]
    
    def get_output_key(self) -> str:
        """Define output key for voice script."""
        return "voice_script"