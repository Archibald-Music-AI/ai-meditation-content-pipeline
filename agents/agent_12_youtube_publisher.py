"""
Agent 12: YouTube Publisher Agent

This agent handles the complete YouTube publishing pipeline, including video upload,
metadata optimization, thumbnail application, and post-publication analytics setup.
It manages all aspects of YouTube API integration and content optimization.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta
import hashlib

from .base import BaseAgent, AgentConfig
from .agent_10_video_assembly import AssembledVideo
from .agent_11_thumbnail_optimizer import ThumbnailSuite


class PrivacyStatus(Enum):
    """YouTube video privacy status options."""
    PRIVATE = "private"
    UNLISTED = "unlisted"
    PUBLIC = "public"


class CategoryId(Enum):
    """YouTube video category IDs for meditation content."""
    PEOPLE_BLOGS = "22"
    EDUCATION = "27"
    HOWTO_STYLE = "26"
    ENTERTAINMENT = "24"


@dataclass
class VideoMetadata:
    """Optimized video metadata for YouTube."""
    title: str
    description: str
    tags: List[str]
    category_id: CategoryId
    default_language: str = "en"
    privacy_status: PrivacyStatus = PrivacyStatus.PUBLIC
    made_for_kids: bool = False
    embeddable: bool = True
    license: str = "youtube"
    public_stats_viewable: bool = True
    
    
@dataclass
class SEOOptimization:
    """SEO optimization data for YouTube."""
    primary_keywords: List[str]
    secondary_keywords: List[str]
    search_volume_data: Dict[str, int]
    competition_analysis: Dict[str, Any]
    optimal_posting_time: datetime
    hashtags: List[str]
    end_screen_elements: List[Dict[str, Any]]
    cards: List[Dict[str, Any]]


@dataclass
class PublishingSchedule:
    """Publishing schedule and timing optimization."""
    scheduled_publish_time: datetime
    optimal_day_of_week: str
    optimal_hour: int
    timezone: str
    pre_publish_actions: List[str]
    post_publish_actions: List[str]


@dataclass
class YouTubePublication:
    """Complete YouTube publication with metadata and tracking."""
    video_id: str
    video_url: str
    title: str
    upload_timestamp: datetime
    metadata: VideoMetadata
    seo_optimization: SEOOptimization
    publishing_schedule: PublishingSchedule
    thumbnail_url: str
    analytics_setup: Dict[str, Any]
    performance_tracking: Dict[str, Any]
    publication_status: str
    estimated_reach: int
    content_fingerprint: str


class YouTubePublisherAgent(BaseAgent):
    """
    Agent 12: YouTube Publisher and Optimization
    
    This agent handles the complete YouTube publishing pipeline, including
    video upload, metadata optimization, SEO enhancement, and analytics setup.
    It manages all aspects of YouTube API integration and content optimization.
    
    Capabilities:
    - YouTube API integration
    - Video upload and processing
    - Metadata optimization
    - SEO keyword optimization
    - Thumbnail management
    - Publishing schedule optimization
    - Analytics and tracking setup
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the YouTube Publisher Agent."""
        if config is None:
            config = AgentConfig(
                name="youtube_publisher",
                phase=5,
                timeout_seconds=900,  # Publishing can take longer
                max_retries=2,
                required_capabilities=[
                    "youtube_api",
                    "video_upload",
                    "seo_optimization",
                    "metadata_management",
                    "analytics_setup"
                ]
            )
        
        super().__init__(config)
        self.youtube_api_endpoint = "https://www.googleapis.com/youtube/v3"
        self.max_title_length = 100
        self.max_description_length = 5000
        self.max_tags = 500  # characters
        
    async def execute(self, context: Dict[str, Any]) -> YouTubePublication:
        """
        Publish meditation video to YouTube with full optimization.
        
        Args:
            context: Pipeline context containing video and thumbnail data
            
        Returns:
            YouTubePublication with complete publication details
        """
        self.logger.info("Starting YouTube publication process")
        
        # Extract required components
        assembled_video = context.get("assembled_video")
        thumbnail_suite = context.get("thumbnail_suite")
        market_intelligence = context.get("market_intelligence")
        
        if not assembled_video:
            raise ValueError("Assembled video required for YouTube publication")
        
        # Phase 1: Optimize video metadata
        self.logger.info("Optimizing video metadata for YouTube")
        optimized_metadata = await self._optimize_video_metadata(
            assembled_video, market_intelligence
        )
        
        # Phase 2: Perform SEO optimization
        self.logger.info("Performing SEO keyword optimization")
        seo_optimization = await self._perform_seo_optimization(
            optimized_metadata, market_intelligence
        )
        
        # Phase 3: Calculate optimal publishing schedule
        self.logger.info("Calculating optimal publishing schedule")
        publishing_schedule = await self._calculate_publishing_schedule(
            market_intelligence
        )
        
        # Phase 4: Prepare video for upload
        self.logger.info("Preparing video file for upload")
        upload_ready_video = await self._prepare_video_for_upload(assembled_video)
        
        # Phase 5: Upload video to YouTube
        self.logger.info("Uploading video to YouTube")
        video_upload_result = await self._upload_video_to_youtube(
            upload_ready_video, optimized_metadata, publishing_schedule
        )
        
        # Phase 6: Apply thumbnail
        self.logger.info("Applying optimized thumbnail")
        thumbnail_result = await self._apply_thumbnail(
            video_upload_result["video_id"], thumbnail_suite
        )
        
        # Phase 7: Setup analytics and tracking
        self.logger.info("Setting up analytics and performance tracking")
        analytics_setup = await self._setup_analytics_tracking(
            video_upload_result, seo_optimization
        )
        
        # Phase 8: Configure end screens and cards
        self.logger.info("Configuring end screens and cards")
        engagement_setup = await self._setup_engagement_features(
            video_upload_result["video_id"], seo_optimization
        )
        
        # Phase 9: Finalize publication
        self.logger.info("Finalizing YouTube publication")
        publication = await self._finalize_publication(
            video_upload_result, optimized_metadata, seo_optimization,
            publishing_schedule, thumbnail_result, analytics_setup
        )
        
        # Update metrics
        self.metrics.tokens_used = 1200  # Estimated for metadata optimization
        self.metrics.api_calls_made = 5  # YouTube API calls
        self.metrics.estimated_cost = 0.0  # No direct cost for YouTube API
        
        self.logger.info(f"YouTube publication complete: {publication.video_url}")
        return publication
    
    async def _optimize_video_metadata(
        self, 
        video: AssembledVideo, 
        market_data: Optional[Dict[str, Any]]
    ) -> VideoMetadata:
        """Optimize video metadata for YouTube algorithm."""
        await asyncio.sleep(0.5)
        
        # Optimize title
        optimized_title = await self._optimize_title(video.title, market_data)
        
        # Generate optimized description
        optimized_description = await self._generate_optimized_description(
            video, market_data
        )
        
        # Generate relevant tags
        optimized_tags = await self._generate_optimized_tags(
            video.title, market_data
        )
        
        # Select optimal category
        category = self._select_optimal_category(video.title)
        
        return VideoMetadata(
            title=optimized_title,
            description=optimized_description,
            tags=optimized_tags,
            category_id=category,
            default_language="en",
            privacy_status=PrivacyStatus.PUBLIC,
            made_for_kids=False,
            embeddable=True,
            license="youtube",
            public_stats_viewable=True
        )
    
    async def _optimize_title(self, original_title: str, market_data: Optional[Dict[str, Any]]) -> str:
        """Optimize video title for YouTube search and engagement."""
        await asyncio.sleep(0.2)
        
        # Extract key elements
        base_title = original_title
        
        # Add trending keywords if available
        if market_data and hasattr(market_data, 'trending_topics'):
            trending_keywords = market_data.trending_topics[:2]  # Top 2
            for keyword in trending_keywords:
                if keyword.lower() not in base_title.lower():
                    base_title = f"{base_title} | {keyword.replace('_', ' ').title()}"
        
        # Add emotional hooks
        emotional_hooks = [
            "🧘", "✨", "🌟", "💫", "🕉️"
        ]
        
        # Add duration if beneficial
        if "10 min" not in base_title.lower() and "minutes" not in base_title.lower():
            base_title = f"{base_title} (10 Minutes)"
        
        # Ensure under character limit
        if len(base_title) > self.max_title_length:
            base_title = base_title[:self.max_title_length-3] + "..."
        
        return base_title
    
    async def _generate_optimized_description(
        self, 
        video: AssembledVideo, 
        market_data: Optional[Dict[str, Any]]
    ) -> str:
        """Generate SEO-optimized video description."""
        await asyncio.sleep(0.4)
        
        description_parts = []
        
        # Hook (first 125 characters are crucial)
        hook = f"🧘 Transform your day with this {int(video.duration/60)}-minute guided meditation. "
        description_parts.append(hook)
        
        # Benefits section
        benefits = [
            "✨ Reduce stress and anxiety",
            "🌟 Improve focus and clarity",
            "💫 Better sleep quality",
            "🕉️ Inner peace and calm",
            "🌿 Emotional balance"
        ]
        
        description_parts.append("\\n\\nBenefits of this meditation:")
        for benefit in benefits[:3]:  # Top 3 benefits
            description_parts.append(f"• {benefit}")
        
        # Instructions
        instructions = [
            "\\n\\nHow to use this meditation:",
            "1. Find a comfortable position",
            "2. Close your eyes or soften your gaze",
            "3. Follow the gentle guidance",
            "4. Allow yourself to relax completely"
        ]
        description_parts.extend(instructions)
        
        # SEO keywords section
        if market_data and hasattr(market_data, 'trending_topics'):
            keywords_section = "\\n\\nTopics covered: " + ", ".join(
                market_data.trending_topics[:5]
            ).replace("_", " ")
            description_parts.append(keywords_section)
        
        # Call to action
        cta_section = [
            "\\n\\n🔔 Subscribe for more guided meditations",
            "💬 Share your experience in the comments",
            "👍 Like if this meditation helped you",
            "🔄 Share with someone who needs peace"
        ]
        description_parts.extend(cta_section)
        
        # Hashtags
        hashtags = [
            "#GuidedMeditation", "#Mindfulness", "#Relaxation",
            "#StressRelief", "#InnerPeace", "#Meditation",
            "#SelfCare", "#MentalHealth", "#Wellness"
        ]
        description_parts.append("\\n\\n" + " ".join(hashtags[:8]))
        
        # Legal/credits
        credits = [
            "\\n\\n---",
            "🎵 Background music: Original composition",
            "🎙️ Voice: AI-generated with ElevenLabs",
            "🎬 Video: Enhanced Archibald v5.1",
            "\\n© 2024 - Created with love for your wellbeing"
        ]
        description_parts.extend(credits)
        
        full_description = "".join(description_parts)
        
        # Ensure under character limit
        if len(full_description) > self.max_description_length:
            full_description = full_description[:self.max_description_length-3] + "..."
        
        return full_description
    
    async def _generate_optimized_tags(
        self, 
        title: str, 
        market_data: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate optimized tags for YouTube search."""
        await asyncio.sleep(0.3)
        
        tags = []
        
        # Core meditation tags
        core_tags = [
            "guided meditation",
            "mindfulness",
            "relaxation",
            "stress relief",
            "anxiety relief",
            "sleep meditation",
            "meditation music",
            "deep relaxation",
            "inner peace",
            "calm",
            "peaceful"
        ]
        
        # Extract keywords from title
        title_keywords = title.lower().split()
        title_tags = [word for word in title_keywords if len(word) > 3]
        
        # Add trending tags if available
        trending_tags = []
        if market_data and hasattr(market_data, 'trending_topics'):
            trending_tags = [
                topic.replace("_", " ") for topic in market_data.trending_topics[:3]
            ]
        
        # Duration-based tags
        duration_tags = ["10 minute meditation", "short meditation", "quick relaxation"]
        
        # Combine all tags
        all_tags = core_tags + title_tags + trending_tags + duration_tags
        
        # Remove duplicates and limit total character count
        unique_tags = list(dict.fromkeys(all_tags))  # Preserve order
        
        tags_string = ""
        for tag in unique_tags:
            if len(tags_string + tag + ",") <= self.max_tags:
                tags.append(tag)
                tags_string += tag + ","
            else:
                break
        
        return tags
    
    def _select_optimal_category(self, title: str) -> CategoryId:
        """Select optimal YouTube category for meditation content."""
        # Analyze title for category hints
        title_lower = title.lower()
        
        if any(word in title_lower for word in ["how to", "guide", "tutorial"]):
            return CategoryId.HOWTO_STYLE
        elif any(word in title_lower for word in ["learn", "education", "training"]):
            return CategoryId.EDUCATION
        else:
            return CategoryId.PEOPLE_BLOGS  # Default for meditation content
    
    async def _perform_seo_optimization(
        self, 
        metadata: VideoMetadata, 
        market_data: Optional[Dict[str, Any]]
    ) -> SEOOptimization:
        """Perform comprehensive SEO optimization."""
        await asyncio.sleep(0.6)
        
        # Extract primary keywords
        primary_keywords = await self._extract_primary_keywords(metadata, market_data)
        
        # Generate secondary keywords
        secondary_keywords = await self._generate_secondary_keywords(primary_keywords)
        
        # Mock search volume data
        search_volume_data = {
            keyword: 1000 + hash(keyword) % 5000 
            for keyword in primary_keywords + secondary_keywords
        }
        
        # Competition analysis
        competition_analysis = await self._analyze_competition(primary_keywords)
        
        # Calculate optimal posting time
        optimal_posting_time = await self._calculate_optimal_posting_time(market_data)
        
        # Generate hashtags
        hashtags = await self._generate_hashtags(primary_keywords)
        
        # Create end screen elements
        end_screen_elements = await self._create_end_screen_elements()
        
        # Create cards
        cards = await self._create_cards(primary_keywords)
        
        return SEOOptimization(
            primary_keywords=primary_keywords,
            secondary_keywords=secondary_keywords,
            search_volume_data=search_volume_data,
            competition_analysis=competition_analysis,
            optimal_posting_time=optimal_posting_time,
            hashtags=hashtags,
            end_screen_elements=end_screen_elements,
            cards=cards
        )
    
    async def _extract_primary_keywords(
        self, 
        metadata: VideoMetadata, 
        market_data: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Extract primary keywords for SEO."""
        await asyncio.sleep(0.2)
        
        keywords = []
        
        # From title
        title_words = metadata.title.lower().split()
        keywords.extend([word for word in title_words if len(word) > 3])
        
        # From tags
        keywords.extend(metadata.tags[:5])  # Top 5 tags
        
        # From market data
        if market_data and hasattr(market_data, 'trending_topics'):
            keywords.extend(market_data.trending_topics[:3])
        
        return list(dict.fromkeys(keywords))[:10]  # Top 10 unique keywords
    
    async def _generate_secondary_keywords(self, primary_keywords: List[str]) -> List[str]:
        """Generate secondary keywords for long-tail SEO."""
        await asyncio.sleep(0.2)
        
        secondary_keywords = []
        
        # Generate variations
        for keyword in primary_keywords[:3]:  # Top 3
            variations = [
                f"{keyword} for beginners",
                f"best {keyword}",
                f"{keyword} techniques",
                f"how to {keyword}",
                f"{keyword} benefits"
            ]
            secondary_keywords.extend(variations)
        
        return secondary_keywords[:15]  # Limit to 15
    
    async def _analyze_competition(self, keywords: List[str]) -> Dict[str, Any]:
        """Analyze competition for target keywords."""
        await asyncio.sleep(0.3)
        
        return {
            "competition_level": "medium",
            "top_competitors": [
                "Meditation Channel A",
                "Mindfulness Studio B",
                "Relaxation Hub C"
            ],
            "content_gaps": [
                "10-minute sessions",
                "workplace meditation",
                "beginner-friendly content"
            ],
            "optimization_opportunities": [
                "Better thumbnail design",
                "Improved description SEO",
                "More engaging titles"
            ]
        }
    
    async def _calculate_optimal_posting_time(self, market_data: Optional[Dict[str, Any]]) -> datetime:
        """Calculate optimal posting time based on audience data."""
        await asyncio.sleep(0.1)
        
        # Default to Sunday 7 PM EST (high engagement for meditation content)
        base_time = datetime.now().replace(hour=19, minute=0, second=0, microsecond=0)
        
        # Adjust to next Sunday
        days_ahead = 6 - base_time.weekday()  # Sunday is 6
        if days_ahead <= 0:
            days_ahead += 7
        
        optimal_time = base_time + timedelta(days=days_ahead)
        
        # Adjust based on market data if available
        if market_data and hasattr(market_data, 'optimal_timing'):
            # Would adjust based on market intelligence
            pass
        
        return optimal_time
    
    async def _generate_hashtags(self, keywords: List[str]) -> List[str]:
        """Generate relevant hashtags for social media."""
        await asyncio.sleep(0.1)
        
        hashtags = []
        
        # Core meditation hashtags
        core_hashtags = [
            "#GuidedMeditation",
            "#Mindfulness",
            "#StressRelief",
            "#InnerPeace",
            "#Relaxation",
            "#SelfCare",
            "#MentalHealth",
            "#Wellness"
        ]
        
        # Generate from keywords
        keyword_hashtags = [
            f"#{keyword.replace(' ', '').replace('_', '').title()}" 
            for keyword in keywords[:5]
        ]
        
        hashtags.extend(core_hashtags)
        hashtags.extend(keyword_hashtags)
        
        return list(dict.fromkeys(hashtags))[:15]  # Limit to 15 unique hashtags
    
    async def _create_end_screen_elements(self) -> List[Dict[str, Any]]:
        """Create end screen elements for engagement."""
        await asyncio.sleep(0.1)
        
        return [
            {
                "type": "subscribe",
                "position": {"x": 0.6, "y": 0.1},
                "duration": 20,
                "message": "Subscribe for daily meditations"
            },
            {
                "type": "playlist",
                "position": {"x": 0.1, "y": 0.1},
                "duration": 20,
                "playlist_id": "meditation_playlist",
                "title": "More Meditations"
            },
            {
                "type": "video",
                "position": {"x": 0.1, "y": 0.6},
                "duration": 20,
                "video_type": "recent_upload",
                "title": "Latest Meditation"
            }
        ]
    
    async def _create_cards(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Create cards for enhanced engagement."""
        await asyncio.sleep(0.1)
        
        return [
            {
                "type": "playlist",
                "timing": 60,  # 1 minute in
                "message": "Explore more meditations",
                "playlist_id": "related_meditations"
            },
            {
                "type": "poll",
                "timing": 300,  # 5 minutes in
                "message": "How are you feeling?",
                "options": ["Relaxed", "Peaceful", "Centered", "Calm"]
            },
            {
                "type": "link",
                "timing": 480,  # 8 minutes in
                "message": "Download our meditation app",
                "url": "https://example.com/app"
            }
        ]
    
    async def _calculate_publishing_schedule(self, market_data: Optional[Dict[str, Any]]) -> PublishingSchedule:
        """Calculate optimal publishing schedule."""
        await asyncio.sleep(0.2)
        
        # Calculate optimal timing
        optimal_publish_time = await self._calculate_optimal_posting_time(market_data)
        
        return PublishingSchedule(
            scheduled_publish_time=optimal_publish_time,
            optimal_day_of_week="Sunday",
            optimal_hour=19,  # 7 PM
            timezone="EST",
            pre_publish_actions=[
                "thumbnail_optimization",
                "description_finalization",
                "tags_verification",
                "end_screen_setup"
            ],
            post_publish_actions=[
                "social_media_promotion",
                "analytics_monitoring",
                "engagement_tracking",
                "performance_analysis"
            ]
        )
    
    async def _prepare_video_for_upload(self, video: AssembledVideo) -> str:
        """Prepare video file for YouTube upload."""
        await asyncio.sleep(0.5)
        
        # Verify video format and quality
        if not Path(video.file_path).exists():
            raise FileNotFoundError(f"Video file not found: {video.file_path}")
        
        # In production, would perform additional checks:
        # - Video format compatibility
        # - File size limits
        # - Duration limits
        # - Quality verification
        
        return video.file_path
    
    async def _upload_video_to_youtube(
        self, 
        video_path: str, 
        metadata: VideoMetadata,
        schedule: PublishingSchedule
    ) -> Dict[str, Any]:
        """Upload video to YouTube."""
        await asyncio.sleep(3.0)  # Simulate upload time
        
        # Mock YouTube upload - in production, would use actual YouTube API
        video_id = self._generate_video_id(video_path)
        
        upload_result = {
            "video_id": video_id,
            "video_url": f"https://www.youtube.com/watch?v={video_id}",
            "upload_status": "processed",
            "processing_progress": 100,
            "upload_timestamp": datetime.now(),
            "file_size": Path(video_path).stat().st_size,
            "duration": metadata.title,  # Would extract actual duration
            "privacy_status": metadata.privacy_status.value
        }
        
        return upload_result
    
    def _generate_video_id(self, video_path: str) -> str:
        """Generate unique video ID."""
        # Create hash from video path and timestamp
        content = f"{video_path}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:11]  # YouTube-style ID
    
    async def _apply_thumbnail(self, video_id: str, thumbnail_suite: Optional[ThumbnailSuite]) -> Dict[str, Any]:
        """Apply optimized thumbnail to YouTube video."""
        await asyncio.sleep(0.5)
        
        if not thumbnail_suite:
            return {"status": "no_thumbnail", "message": "No thumbnail suite provided"}
        
        # Use primary thumbnail
        primary_thumbnail = thumbnail_suite.primary_thumbnail
        
        # Mock thumbnail upload - in production, would use YouTube API
        thumbnail_result = {
            "video_id": video_id,
            "thumbnail_url": f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg",
            "thumbnail_status": "set",
            "variant_used": primary_thumbnail.variant_id,
            "predicted_ctr": primary_thumbnail.predicted_ctr,
            "upload_timestamp": datetime.now()
        }
        
        return thumbnail_result
    
    async def _setup_analytics_tracking(
        self, 
        upload_result: Dict[str, Any], 
        seo_optimization: SEOOptimization
    ) -> Dict[str, Any]:
        """Setup analytics and performance tracking."""
        await asyncio.sleep(0.3)
        
        return {
            "video_id": upload_result["video_id"],
            "tracking_enabled": True,
            "metrics_to_track": [
                "views",
                "watch_time",
                "engagement_rate",
                "click_through_rate",
                "subscriber_conversion",
                "comment_sentiment"
            ],
            "keyword_tracking": seo_optimization.primary_keywords,
            "performance_thresholds": {
                "good_ctr": 0.05,
                "good_engagement": 0.08,
                "good_retention": 0.6
            },
            "reporting_schedule": "daily",
            "alert_conditions": [
                "ctr_below_threshold",
                "engagement_drop",
                "negative_sentiment_spike"
            ]
        }
    
    async def _setup_engagement_features(self, video_id: str, seo_optimization: SEOOptimization) -> Dict[str, Any]:
        """Setup end screens, cards, and other engagement features."""
        await asyncio.sleep(0.2)
        
        return {
            "video_id": video_id,
            "end_screens": seo_optimization.end_screen_elements,
            "cards": seo_optimization.cards,
            "community_features": {
                "comments_enabled": True,
                "community_tab_promotion": True,
                "premieres_enabled": False
            },
            "interactive_elements": {
                "polls_enabled": True,
                "live_chat_enabled": False,
                "super_chat_enabled": False
            }
        }
    
    async def _finalize_publication(
        self, 
        upload_result: Dict[str, Any],
        metadata: VideoMetadata,
        seo_optimization: SEOOptimization,
        schedule: PublishingSchedule,
        thumbnail_result: Dict[str, Any],
        analytics_setup: Dict[str, Any]
    ) -> YouTubePublication:
        """Finalize YouTube publication with complete metadata."""
        await asyncio.sleep(0.2)
        
        # Generate content fingerprint
        content_fingerprint = self._generate_content_fingerprint(
            upload_result["video_id"], metadata.title
        )
        
        # Estimate reach based on SEO optimization
        estimated_reach = self._estimate_video_reach(seo_optimization, metadata)
        
        return YouTubePublication(
            video_id=upload_result["video_id"],
            video_url=upload_result["video_url"],
            title=metadata.title,
            upload_timestamp=upload_result["upload_timestamp"],
            metadata=metadata,
            seo_optimization=seo_optimization,
            publishing_schedule=schedule,
            thumbnail_url=thumbnail_result.get("thumbnail_url", ""),
            analytics_setup=analytics_setup,
            performance_tracking={
                "initial_metrics": {
                    "views": 0,
                    "likes": 0,
                    "comments": 0,
                    "shares": 0
                },
                "tracking_start": datetime.now(),
                "next_review": datetime.now() + timedelta(hours=24)
            },
            publication_status="published",
            estimated_reach=estimated_reach,
            content_fingerprint=content_fingerprint
        )
    
    def _generate_content_fingerprint(self, video_id: str, title: str) -> str:
        """Generate unique content fingerprint."""
        content = f"{video_id}_{title}_{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _estimate_video_reach(self, seo_optimization: SEOOptimization, metadata: VideoMetadata) -> int:
        """Estimate potential video reach based on optimization."""
        base_reach = 1000  # Base reach for new channel
        
        # Adjust based on SEO factors
        keyword_boost = len(seo_optimization.primary_keywords) * 100
        tag_boost = len(metadata.tags) * 50
        description_boost = len(metadata.description) // 10
        
        # Adjust based on competition
        competition_factor = 0.8  # Assume medium competition
        
        estimated_reach = (base_reach + keyword_boost + tag_boost + description_boost) * competition_factor
        
        return int(estimated_reach)
    
    async def validate_output(self, output: YouTubePublication) -> bool:
        """
        Validate YouTube publication completeness and quality.
        
        Args:
            output: YouTubePublication to validate
            
        Returns:
            True if publication meets quality standards
        """
        validation_checks = [
            output.video_id is not None and len(output.video_id) > 0,
            output.video_url is not None and "youtube.com" in output.video_url,
            output.title is not None and len(output.title) > 0,
            output.metadata is not None,
            len(output.metadata.tags) > 0,
            len(output.metadata.description) > 100,
            output.seo_optimization is not None,
            len(output.seo_optimization.primary_keywords) > 0,
            output.analytics_setup is not None,
            output.publication_status == "published",
            output.estimated_reach > 0,
            len(output.content_fingerprint) > 0
        ]
        
        validation_passed = all(validation_checks)
        
        if not validation_passed:
            self.logger.warning("YouTube publication validation failed")
            
        return validation_passed
    
    def get_required_context_keys(self) -> List[str]:
        """Define required context keys for YouTube publication."""
        return ["assembled_video"]
    
    def get_output_key(self) -> str:
        """Define output key for YouTube publication."""
        return "youtube_publication"