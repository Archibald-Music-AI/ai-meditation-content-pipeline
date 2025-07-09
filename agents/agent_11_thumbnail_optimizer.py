"""
Agent 11: Thumbnail Optimizer Agent

This agent creates and optimizes thumbnails for maximum click-through rates on
YouTube and other platforms. It uses A/B testing data and visual psychology
principles to generate compelling thumbnails.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
from pathlib import Path
import tempfile
from datetime import datetime
import hashlib

from .base import BaseAgent, AgentConfig
from .agent_10_video_assembly import AssembledVideo


class ThumbnailStyle(Enum):
    """Thumbnail style options based on A/B testing data."""
    MINIMALIST = "minimalist"
    NATURE_FOCUSED = "nature_focused"
    PERSON_CENTERED = "person_centered"
    TEXT_HEAVY = "text_heavy"
    ABSTRACT_ARTISTIC = "abstract_artistic"
    MEDITATION_SYMBOLS = "meditation_symbols"


class ColorScheme(Enum):
    """Color schemes optimized for meditation content."""
    CALMING_BLUES = "calming_blues"
    WARM_SUNSET = "warm_sunset"
    NATURE_GREENS = "nature_greens"
    PURPLE_MYSTICAL = "purple_mystical"
    GOLDEN_HOUR = "golden_hour"
    MONOCHROME_PEACE = "monochrome_peace"


@dataclass
class ThumbnailElement:
    """Individual element within a thumbnail."""
    element_id: str
    element_type: str  # "text", "image", "shape", "gradient", "icon"
    content: str
    position: Tuple[int, int]
    size: Tuple[int, int]
    color: str
    font_family: Optional[str] = None
    font_size: Optional[int] = None
    opacity: float = 1.0
    rotation: float = 0.0
    effects: List[str] = field(default_factory=list)


@dataclass
class ThumbnailVariant:
    """Individual thumbnail variant for A/B testing."""
    variant_id: str
    title: str
    style: ThumbnailStyle
    color_scheme: ColorScheme
    elements: List[ThumbnailElement]
    file_path: str
    predicted_ctr: float
    engagement_score: float
    psychological_principles: List[str]
    target_audience: str
    test_group: str = "A"


@dataclass
class ThumbnailSuite:
    """Complete thumbnail suite with multiple variants."""
    video_title: str
    primary_thumbnail: ThumbnailVariant
    variants: List[ThumbnailVariant]
    recommendations: Dict[str, str]
    performance_predictions: Dict[str, float]
    optimization_notes: List[str]
    creation_timestamp: datetime
    total_variants: int


class ThumbnailOptimizerAgent(BaseAgent):
    """
    Agent 11: Thumbnail Optimizer and A/B Testing
    
    This agent creates optimized thumbnails for meditation videos using
    data-driven design principles and psychological insights to maximize
    click-through rates and engagement.
    
    Capabilities:
    - A/B testing variant generation
    - Visual psychology optimization
    - Color scheme analysis
    - Typography optimization
    - Engagement prediction
    - Platform-specific formatting
    - Performance tracking integration
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Thumbnail Optimizer Agent."""
        if config is None:
            config = AgentConfig(
                name="thumbnail_optimizer",
                phase=4,
                timeout_seconds=180,
                max_retries=3,
                required_capabilities=[
                    "image_generation",
                    "visual_design",
                    "a_b_testing",
                    "engagement_prediction",
                    "color_psychology"
                ]
            )
        
        super().__init__(config)
        self.temp_dir = Path(tempfile.mkdtemp())
        self.thumbnail_dimensions = (1280, 720)  # YouTube optimal
        self.a_b_testing_data = self._load_historical_performance_data()
        
    def _load_historical_performance_data(self) -> Dict[str, Any]:
        """Load historical A/B testing data for optimization."""
        return {
            "best_performing_styles": {
                ThumbnailStyle.MINIMALIST: 0.078,
                ThumbnailStyle.NATURE_FOCUSED: 0.089,
                ThumbnailStyle.PERSON_CENTERED: 0.072,
                ThumbnailStyle.TEXT_HEAVY: 0.065,
                ThumbnailStyle.ABSTRACT_ARTISTIC: 0.081,
                ThumbnailStyle.MEDITATION_SYMBOLS: 0.076
            },
            "color_performance": {
                ColorScheme.CALMING_BLUES: 0.082,
                ColorScheme.WARM_SUNSET: 0.094,
                ColorScheme.NATURE_GREENS: 0.087,
                ColorScheme.PURPLE_MYSTICAL: 0.079,
                ColorScheme.GOLDEN_HOUR: 0.091,
                ColorScheme.MONOCHROME_PEACE: 0.074
            },
            "text_principles": {
                "large_readable_font": 0.15,
                "contrasting_colors": 0.12,
                "emotional_keywords": 0.18,
                "number_inclusion": 0.08,
                "benefit_focused": 0.22
            }
        }
    
    async def execute(self, context: Dict[str, Any]) -> ThumbnailSuite:
        """
        Create optimized thumbnail suite for meditation video.
        
        Args:
            context: Pipeline context containing assembled video
            
        Returns:
            ThumbnailSuite with multiple optimized variants
        """
        self.logger.info("Starting thumbnail optimization process")
        
        # Extract video information
        assembled_video = context.get("assembled_video")
        market_intelligence = context.get("market_intelligence")
        
        if not assembled_video:
            raise ValueError("Assembled video required for thumbnail optimization")
        
        # Phase 1: Analyze video content
        self.logger.info("Analyzing video content for thumbnail optimization")
        content_analysis = await self._analyze_video_content(assembled_video, market_intelligence)
        
        # Phase 2: Generate thumbnail variants
        self.logger.info("Generating thumbnail variants for A/B testing")
        thumbnail_variants = await self._generate_thumbnail_variants(content_analysis)
        
        # Phase 3: Optimize each variant
        self.logger.info("Optimizing individual thumbnail variants")
        optimized_variants = await self._optimize_thumbnail_variants(thumbnail_variants)
        
        # Phase 4: Predict performance
        self.logger.info("Predicting thumbnail performance")
        performance_predictions = await self._predict_thumbnail_performance(optimized_variants)
        
        # Phase 5: Select primary thumbnail
        self.logger.info("Selecting primary thumbnail")
        primary_thumbnail = await self._select_primary_thumbnail(optimized_variants, performance_predictions)
        
        # Phase 6: Generate recommendations
        self.logger.info("Generating optimization recommendations")
        recommendations = await self._generate_recommendations(optimized_variants, performance_predictions)
        
        # Phase 7: Create final thumbnail suite
        self.logger.info("Compiling final thumbnail suite")
        thumbnail_suite = await self._create_thumbnail_suite(
            assembled_video, primary_thumbnail, optimized_variants, 
            recommendations, performance_predictions
        )
        
        # Update metrics
        self.metrics.tokens_used = 800  # Estimated for thumbnail analysis
        self.metrics.api_calls_made = len(optimized_variants)
        self.metrics.estimated_cost = self.estimate_cost(self.metrics.tokens_used, "gpt-4")
        
        self.logger.info(f"Thumbnail optimization complete: {len(optimized_variants)} variants created")
        return thumbnail_suite
    
    async def _analyze_video_content(
        self, 
        video: AssembledVideo, 
        market_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze video content to inform thumbnail design."""
        await asyncio.sleep(0.4)
        
        # Extract key information from video
        content_analysis = {
            "video_title": video.title,
            "duration": video.duration,
            "primary_theme": self._extract_primary_theme(video.title),
            "target_audience": self._determine_target_audience(video.title, market_data),
            "emotional_tone": self._analyze_emotional_tone(video.title),
            "key_benefits": self._extract_key_benefits(video.title),
            "competitive_landscape": self._analyze_competition(market_data) if market_data else {}
        }
        
        return content_analysis
    
    def _extract_primary_theme(self, title: str) -> str:
        """Extract primary theme from video title."""
        theme_keywords = {
            "sleep": "sleep_meditation",
            "anxiety": "anxiety_relief",
            "stress": "stress_reduction",
            "focus": "focus_enhancement",
            "mindfulness": "mindfulness_practice",
            "relaxation": "relaxation",
            "healing": "emotional_healing"
        }
        
        title_lower = title.lower()
        for keyword, theme in theme_keywords.items():
            if keyword in title_lower:
                return theme
        
        return "general_meditation"
    
    def _determine_target_audience(self, title: str, market_data: Optional[Dict[str, Any]]) -> str:
        """Determine target audience based on title and market data."""
        if market_data and hasattr(market_data, 'audience_demographics'):
            # Use market intelligence data
            return "data_driven_audience"
        
        # Fallback to title analysis
        if "beginner" in title.lower():
            return "beginners"
        elif "advanced" in title.lower():
            return "advanced_practitioners"
        else:
            return "general_audience"
    
    def _analyze_emotional_tone(self, title: str) -> str:
        """Analyze emotional tone of the video."""
        tone_keywords = {
            "calm": "peaceful",
            "soothing": "peaceful",
            "energizing": "uplifting",
            "healing": "therapeutic",
            "powerful": "empowering",
            "gentle": "nurturing"
        }
        
        title_lower = title.lower()
        for keyword, tone in tone_keywords.items():
            if keyword in title_lower:
                return tone
        
        return "peaceful"  # Default for meditation
    
    def _extract_key_benefits(self, title: str) -> List[str]:
        """Extract key benefits from video title."""
        benefit_keywords = {
            "sleep": "better_sleep",
            "anxiety": "reduced_anxiety",
            "stress": "stress_relief",
            "focus": "improved_focus",
            "calm": "inner_calm",
            "peace": "inner_peace",
            "healing": "emotional_healing",
            "confidence": "increased_confidence"
        }
        
        title_lower = title.lower()
        benefits = []
        
        for keyword, benefit in benefit_keywords.items():
            if keyword in title_lower:
                benefits.append(benefit)
        
        return benefits if benefits else ["general_wellbeing"]
    
    def _analyze_competition(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze competitive landscape for thumbnail optimization."""
        return {
            "common_styles": ["minimalist", "nature_focused"],
            "oversaturated_approaches": ["text_heavy"],
            "opportunities": ["abstract_artistic", "meditation_symbols"]
        }
    
    async def _generate_thumbnail_variants(self, content_analysis: Dict[str, Any]) -> List[ThumbnailVariant]:
        """Generate multiple thumbnail variants for A/B testing."""
        await asyncio.sleep(0.8)
        
        variants = []
        
        # Generate variants for top-performing styles
        top_styles = [
            ThumbnailStyle.NATURE_FOCUSED,
            ThumbnailStyle.MINIMALIST,
            ThumbnailStyle.ABSTRACT_ARTISTIC,
            ThumbnailStyle.MEDITATION_SYMBOLS
        ]
        
        for i, style in enumerate(top_styles):
            variant = await self._create_thumbnail_variant(content_analysis, style, f"variant_{i+1}")
            variants.append(variant)
        
        return variants
    
    async def _create_thumbnail_variant(
        self, 
        content_analysis: Dict[str, Any], 
        style: ThumbnailStyle, 
        variant_id: str
    ) -> ThumbnailVariant:
        """Create individual thumbnail variant."""
        await asyncio.sleep(0.3)
        
        # Select optimal color scheme
        color_scheme = self._select_color_scheme(content_analysis, style)
        
        # Create elements based on style
        elements = await self._create_thumbnail_elements(content_analysis, style, color_scheme)
        
        # Generate thumbnail file
        file_path = await self._render_thumbnail(elements, style, color_scheme, variant_id)
        
        # Predict performance
        predicted_ctr = self._predict_variant_ctr(style, color_scheme, elements)
        engagement_score = self._calculate_engagement_score(elements, content_analysis)
        
        # Identify psychological principles
        psychological_principles = self._identify_psychological_principles(elements, style)
        
        return ThumbnailVariant(
            variant_id=variant_id,
            title=content_analysis["video_title"],
            style=style,
            color_scheme=color_scheme,
            elements=elements,
            file_path=file_path,
            predicted_ctr=predicted_ctr,
            engagement_score=engagement_score,
            psychological_principles=psychological_principles,
            target_audience=content_analysis["target_audience"],
            test_group="A" if "1" in variant_id or "2" in variant_id else "B"
        )
    
    def _select_color_scheme(self, content_analysis: Dict[str, Any], style: ThumbnailStyle) -> ColorScheme:
        """Select optimal color scheme for thumbnail."""
        theme = content_analysis["primary_theme"]
        emotional_tone = content_analysis["emotional_tone"]
        
        # Map themes to optimal color schemes
        theme_colors = {
            "sleep_meditation": ColorScheme.PURPLE_MYSTICAL,
            "anxiety_relief": ColorScheme.CALMING_BLUES,
            "stress_reduction": ColorScheme.NATURE_GREENS,
            "focus_enhancement": ColorScheme.GOLDEN_HOUR,
            "general_meditation": ColorScheme.WARM_SUNSET
        }
        
        # Consider emotional tone
        if emotional_tone == "uplifting":
            return ColorScheme.GOLDEN_HOUR
        elif emotional_tone == "therapeutic":
            return ColorScheme.CALMING_BLUES
        
        return theme_colors.get(theme, ColorScheme.WARM_SUNSET)
    
    async def _create_thumbnail_elements(
        self, 
        content_analysis: Dict[str, Any], 
        style: ThumbnailStyle, 
        color_scheme: ColorScheme
    ) -> List[ThumbnailElement]:
        """Create thumbnail elements based on style and content."""
        await asyncio.sleep(0.4)
        
        elements = []
        
        # Background element
        background = ThumbnailElement(
            element_id="background",
            element_type="gradient",
            content=f"{color_scheme.value}_gradient",
            position=(0, 0),
            size=self.thumbnail_dimensions,
            color=self._get_primary_color(color_scheme),
            effects=["subtle_texture"]
        )
        elements.append(background)
        
        # Title text
        title_text = await self._create_title_element(content_analysis, style, color_scheme)
        elements.append(title_text)
        
        # Style-specific elements
        if style == ThumbnailStyle.NATURE_FOCUSED:
            nature_element = ThumbnailElement(
                element_id="nature_backdrop",
                element_type="image",
                content="meditation_nature_scene",
                position=(0, 0),
                size=self.thumbnail_dimensions,
                color="#ffffff",
                opacity=0.7,
                effects=["soft_blur", "color_overlay"]
            )
            elements.append(nature_element)
        
        elif style == ThumbnailStyle.MEDITATION_SYMBOLS:
            symbol_element = ThumbnailElement(
                element_id="meditation_symbol",
                element_type="icon",
                content="lotus_flower",
                position=(960, 360),  # Center-right
                size=(200, 200),
                color=self._get_accent_color(color_scheme),
                opacity=0.8,
                effects=["subtle_glow"]
            )
            elements.append(symbol_element)
        
        elif style == ThumbnailStyle.MINIMALIST:
            # Add subtle geometric shapes
            geometric_element = ThumbnailElement(
                element_id="geometric_accent",
                element_type="shape",
                content="circle",
                position=(100, 100),
                size=(80, 80),
                color=self._get_accent_color(color_scheme),
                opacity=0.3,
                effects=["soft_glow"]
            )
            elements.append(geometric_element)
        
        # Benefit indicator
        if content_analysis.get("key_benefits"):
            benefit_element = await self._create_benefit_element(content_analysis, color_scheme)
            elements.append(benefit_element)
        
        return elements
    
    async def _create_title_element(
        self, 
        content_analysis: Dict[str, Any], 
        style: ThumbnailStyle, 
        color_scheme: ColorScheme
    ) -> ThumbnailElement:
        """Create optimized title text element."""
        await asyncio.sleep(0.2)
        
        # Optimize title text for thumbnail
        optimized_title = self._optimize_title_text(content_analysis["video_title"])
        
        # Position based on style
        if style == ThumbnailStyle.MINIMALIST:
            position = (640, 200)  # Upper center
        elif style == ThumbnailStyle.NATURE_FOCUSED:
            position = (640, 500)  # Lower center
        else:
            position = (640, 360)  # Center
        
        return ThumbnailElement(
            element_id="title_text",
            element_type="text",
            content=optimized_title,
            position=position,
            size=(800, 120),
            color=self._get_text_color(color_scheme),
            font_family="Montserrat",
            font_size=54,
            opacity=1.0,
            effects=["shadow", "outline"]
        )
    
    def _optimize_title_text(self, original_title: str) -> str:
        """Optimize title text for thumbnail readability."""
        # Shorten for thumbnail display
        if len(original_title) > 30:
            # Extract key words
            key_words = self._extract_key_words(original_title)
            return " ".join(key_words[:4])  # Limit to 4 key words
        
        return original_title
    
    def _extract_key_words(self, title: str) -> List[str]:
        """Extract key words from title for thumbnail."""
        # Remove common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        
        words = title.lower().split()
        key_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        return key_words
    
    async def _create_benefit_element(
        self, 
        content_analysis: Dict[str, Any], 
        color_scheme: ColorScheme
    ) -> ThumbnailElement:
        """Create benefit indicator element."""
        await asyncio.sleep(0.1)
        
        primary_benefit = content_analysis["key_benefits"][0]
        
        return ThumbnailElement(
            element_id="benefit_indicator",
            element_type="text",
            content=primary_benefit.replace("_", " ").title(),
            position=(100, 600),
            size=(300, 60),
            color=self._get_accent_color(color_scheme),
            font_family="Open Sans",
            font_size=24,
            opacity=0.9,
            effects=["subtle_background", "rounded_corners"]
        )
    
    def _get_primary_color(self, color_scheme: ColorScheme) -> str:
        """Get primary color for color scheme."""
        color_map = {
            ColorScheme.CALMING_BLUES: "#2E5BBA",
            ColorScheme.WARM_SUNSET: "#FF6B35",
            ColorScheme.NATURE_GREENS: "#4CAF50",
            ColorScheme.PURPLE_MYSTICAL: "#7B1FA2",
            ColorScheme.GOLDEN_HOUR: "#FFB300",
            ColorScheme.MONOCHROME_PEACE: "#424242"
        }
        return color_map.get(color_scheme, "#2E5BBA")
    
    def _get_accent_color(self, color_scheme: ColorScheme) -> str:
        """Get accent color for color scheme."""
        accent_map = {
            ColorScheme.CALMING_BLUES: "#E3F2FD",
            ColorScheme.WARM_SUNSET: "#FFF3E0",
            ColorScheme.NATURE_GREENS: "#E8F5E8",
            ColorScheme.PURPLE_MYSTICAL: "#F3E5F5",
            ColorScheme.GOLDEN_HOUR: "#FFFDE7",
            ColorScheme.MONOCHROME_PEACE: "#F5F5F5"
        }
        return accent_map.get(color_scheme, "#E3F2FD")
    
    def _get_text_color(self, color_scheme: ColorScheme) -> str:
        """Get optimal text color for color scheme."""
        # Always use high contrast colors for text
        return "#FFFFFF"  # White text with shadow/outline for readability
    
    async def _render_thumbnail(
        self, 
        elements: List[ThumbnailElement], 
        style: ThumbnailStyle, 
        color_scheme: ColorScheme, 
        variant_id: str
    ) -> str:
        """Render thumbnail image from elements."""
        await asyncio.sleep(0.5)
        
        thumbnail_file = self.temp_dir / f"thumbnail_{variant_id}.jpg"
        
        # Mock thumbnail rendering - in production, would use actual image generation
        with open(thumbnail_file, 'wb') as f:
            f.write(b'mock_thumbnail_image_data')
        
        return str(thumbnail_file)
    
    def _predict_variant_ctr(
        self, 
        style: ThumbnailStyle, 
        color_scheme: ColorScheme, 
        elements: List[ThumbnailElement]
    ) -> float:
        """Predict click-through rate for thumbnail variant."""
        base_ctr = self.a_b_testing_data["best_performing_styles"].get(style, 0.07)
        color_multiplier = self.a_b_testing_data["color_performance"].get(color_scheme, 0.08) / 0.08
        
        # Adjust based on elements
        element_bonus = 0.0
        for element in elements:
            if element.element_type == "text" and len(element.effects) > 0:
                element_bonus += 0.005
            if element.element_type == "icon":
                element_bonus += 0.003
        
        return base_ctr * color_multiplier + element_bonus
    
    def _calculate_engagement_score(
        self, 
        elements: List[ThumbnailElement], 
        content_analysis: Dict[str, Any]
    ) -> float:
        """Calculate engagement score based on design elements."""
        score = 0.5  # Base score
        
        # Text readability
        text_elements = [e for e in elements if e.element_type == "text"]
        if text_elements:
            primary_text = text_elements[0]
            if primary_text.font_size and primary_text.font_size >= 48:
                score += 0.1
            if "shadow" in primary_text.effects:
                score += 0.05
        
        # Visual hierarchy
        if len(elements) >= 3:
            score += 0.1
        
        # Color harmony
        score += 0.1  # Assume good color harmony from scheme selection
        
        # Emotional appeal
        emotional_tone = content_analysis.get("emotional_tone", "peaceful")
        if emotional_tone in ["uplifting", "empowering"]:
            score += 0.1
        
        return min(score, 1.0)
    
    def _identify_psychological_principles(
        self, 
        elements: List[ThumbnailElement], 
        style: ThumbnailStyle
    ) -> List[str]:
        """Identify psychological principles applied in thumbnail."""
        principles = []
        
        # Visual hierarchy
        if len(elements) >= 3:
            principles.append("visual_hierarchy")
        
        # Color psychology
        principles.append("color_psychology")
        
        # Contrast and readability
        text_elements = [e for e in elements if e.element_type == "text"]
        if text_elements and any("shadow" in e.effects for e in text_elements):
            principles.append("contrast_optimization")
        
        # Emotional appeal
        if style in [ThumbnailStyle.NATURE_FOCUSED, ThumbnailStyle.MEDITATION_SYMBOLS]:
            principles.append("emotional_resonance")
        
        # Minimalism
        if style == ThumbnailStyle.MINIMALIST:
            principles.append("cognitive_ease")
        
        return principles
    
    async def _optimize_thumbnail_variants(self, variants: List[ThumbnailVariant]) -> List[ThumbnailVariant]:
        """Optimize thumbnail variants based on performance predictions."""
        await asyncio.sleep(0.6)
        
        # Sort by predicted CTR
        variants.sort(key=lambda v: v.predicted_ctr, reverse=True)
        
        # Apply optimizations to top variants
        for i, variant in enumerate(variants[:2]):  # Optimize top 2
            # Enhance text contrast
            text_elements = [e for e in variant.elements if e.element_type == "text"]
            for text_element in text_elements:
                if "outline" not in text_element.effects:
                    text_element.effects.append("outline")
            
            # Re-render with optimizations
            variant.file_path = await self._render_thumbnail(
                variant.elements, variant.style, variant.color_scheme, f"{variant.variant_id}_optimized"
            )
        
        return variants
    
    async def _predict_thumbnail_performance(self, variants: List[ThumbnailVariant]) -> Dict[str, float]:
        """Predict performance metrics for thumbnail variants."""
        await asyncio.sleep(0.3)
        
        predictions = {}
        
        for variant in variants:
            predictions[variant.variant_id] = {
                "predicted_ctr": variant.predicted_ctr,
                "engagement_score": variant.engagement_score,
                "brand_alignment": 0.85,  # Mock brand alignment score
                "platform_optimization": 0.90,  # Mock platform optimization score
                "psychological_impact": len(variant.psychological_principles) * 0.1
            }
        
        return predictions
    
    async def _select_primary_thumbnail(
        self, 
        variants: List[ThumbnailVariant], 
        predictions: Dict[str, float]
    ) -> ThumbnailVariant:
        """Select primary thumbnail based on performance predictions."""
        await asyncio.sleep(0.1)
        
        # Score each variant
        scored_variants = []
        for variant in variants:
            score = (
                predictions[variant.variant_id]["predicted_ctr"] * 0.4 +
                predictions[variant.variant_id]["engagement_score"] * 0.3 +
                predictions[variant.variant_id]["brand_alignment"] * 0.2 +
                predictions[variant.variant_id]["platform_optimization"] * 0.1
            )
            scored_variants.append((score, variant))
        
        # Return highest scoring variant
        scored_variants.sort(key=lambda x: x[0], reverse=True)
        return scored_variants[0][1]
    
    async def _generate_recommendations(
        self, 
        variants: List[ThumbnailVariant], 
        predictions: Dict[str, float]
    ) -> Dict[str, str]:
        """Generate optimization recommendations."""
        await asyncio.sleep(0.2)
        
        recommendations = {}
        
        # Analyze performance patterns
        best_variant = max(variants, key=lambda v: v.predicted_ctr)
        
        recommendations["primary_choice"] = f"Use {best_variant.variant_id} as primary thumbnail"
        recommendations["a_b_testing"] = f"Test {best_variant.variant_id} against variant with different color scheme"
        recommendations["style_optimization"] = f"Best performing style: {best_variant.style.value}"
        recommendations["color_optimization"] = f"Best performing color scheme: {best_variant.color_scheme.value}"
        
        # Performance insights
        if best_variant.predicted_ctr > 0.08:
            recommendations["performance_outlook"] = "High CTR potential - excellent thumbnail design"
        elif best_variant.predicted_ctr > 0.06:
            recommendations["performance_outlook"] = "Good CTR potential - solid thumbnail design"
        else:
            recommendations["performance_outlook"] = "Consider additional optimization for better performance"
        
        return recommendations
    
    async def _create_thumbnail_suite(
        self, 
        video: AssembledVideo, 
        primary_thumbnail: ThumbnailVariant,
        variants: List[ThumbnailVariant],
        recommendations: Dict[str, str],
        predictions: Dict[str, float]
    ) -> ThumbnailSuite:
        """Create final thumbnail suite."""
        await asyncio.sleep(0.2)
        
        # Calculate performance predictions summary
        performance_summary = {
            "average_predicted_ctr": sum(v.predicted_ctr for v in variants) / len(variants),
            "best_predicted_ctr": max(v.predicted_ctr for v in variants),
            "engagement_potential": sum(v.engagement_score for v in variants) / len(variants)
        }
        
        # Generate optimization notes
        optimization_notes = [
            f"Generated {len(variants)} A/B testing variants",
            f"Primary thumbnail uses {primary_thumbnail.style.value} style",
            f"Best predicted CTR: {primary_thumbnail.predicted_ctr:.3f}",
            f"Applied {len(primary_thumbnail.psychological_principles)} psychological principles",
            "Optimized for YouTube thumbnail requirements"
        ]
        
        return ThumbnailSuite(
            video_title=video.title,
            primary_thumbnail=primary_thumbnail,
            variants=variants,
            recommendations=recommendations,
            performance_predictions=performance_summary,
            optimization_notes=optimization_notes,
            creation_timestamp=datetime.now(),
            total_variants=len(variants)
        )
    
    async def validate_output(self, output: ThumbnailSuite) -> bool:
        """
        Validate thumbnail suite quality and completeness.
        
        Args:
            output: ThumbnailSuite to validate
            
        Returns:
            True if thumbnail suite meets quality standards
        """
        validation_checks = [
            output.primary_thumbnail is not None,
            len(output.variants) >= 2,
            output.primary_thumbnail.predicted_ctr > 0.05,
            Path(output.primary_thumbnail.file_path).exists(),
            len(output.recommendations) > 0,
            output.performance_predictions is not None,
            all(Path(variant.file_path).exists() for variant in output.variants),
            all(len(variant.elements) >= 2 for variant in output.variants),
            output.total_variants == len(output.variants)
        ]
        
        validation_passed = all(validation_checks)
        
        if not validation_passed:
            self.logger.warning("Thumbnail suite validation failed")
            
        return validation_passed
    
    def get_required_context_keys(self) -> List[str]:
        """Define required context keys for thumbnail optimization."""
        return ["assembled_video"]
    
    def get_output_key(self) -> str:
        """Define output key for thumbnail suite."""
        return "thumbnail_suite"
    
    def cleanup_temp_files(self):
        """Clean up temporary files created during processing."""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
            self.logger.info("Temporary files cleaned up")
        except Exception as e:
            self.logger.warning(f"Error cleaning up temp files: {e}")
    
    def __del__(self):
        """Cleanup on object destruction."""
        self.cleanup_temp_files()