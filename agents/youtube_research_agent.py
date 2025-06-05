import os
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import time
from datetime import datetime, timezone

class YouTubeResearchAgent:
    """
    An agent responsible for researching competitive YouTube content using
    the YouTube Data API v3. Takes search terms from Step 4 and returns
    competitive analysis data for Step 6 content optimization.
    """

    def __init__(self):
        """
        Initializes the YouTube Research Agent with API credentials.
        """
        self._initialize_youtube_api()
        self.logger = logging.getLogger(__name__)

    def _initialize_youtube_api(self):
        """
        Initializes YouTube Data API v3 client using API key from environment.
        """
        load_dotenv()
        api_key = os.getenv("YOUTUBE_API_KEY")
        
        if not api_key:
            raise ValueError(
                "YOUTUBE_API_KEY environment variable not set. "
                "Please get an API key from Google Cloud Console and add it to your .env file."
            )
        
        try:
            self.youtube = build('youtube', 'v3', developerKey=api_key)
            self.logger.info("✅ YouTubeResearchAgent: YouTube Data API v3 initialized successfully")
        except Exception as e:
            self.logger.exception("Failed to initialize YouTube API.") # .exception includes stack trace
            raise RuntimeError(f"Failed to initialize YouTube API: {e}")

    def research_competitive_content(self, search_terms: List[str], max_results_per_term: int = 5) -> Dict:
        """
        Research competitive content on YouTube using the provided search terms.
        
        Args:
            search_terms: List of search terms from Step 4 content analysis
            max_results_per_term: Maximum number of videos to analyze per search term
            
        Returns:
            Dictionary containing research data for Step 6 content optimization
        """
        if not search_terms:
            self.logger.warning("⚠️ No search terms provided for YouTube research")
            return self._empty_research_result()
        
        self.logger.info("🔍 YouTubeResearchAgent: Starting research for %d search terms", len(search_terms))
        
        all_research_data = {
            "research_summary": {
                "total_search_terms": len(search_terms),
                "total_videos_analyzed": 0,
                "search_terms_used": search_terms
            },
            "term_results": {},
            "competitive_analysis": {}
        }
        
        all_videos = []
        
        for term in search_terms:
            self.logger.info("🔎 Researching: '%s'", term)
            try:
                term_data = self._research_single_term(term, max_results_per_term)
                all_research_data["term_results"][term] = term_data
                all_videos.extend(term_data["videos"])
                
                # Rate limiting - YouTube API has quotas
                time.sleep(0.5) # No logging for sleep
                
            except Exception as e:
                self.logger.error("❌ Error researching term '%s': %s", term, e, exc_info=True)
                all_research_data["term_results"][term] = self._empty_term_result()
        
        # Generate competitive analysis
        all_research_data["competitive_analysis"] = self._analyze_competitive_data(all_videos) # This method uses self.logger internally
        all_research_data["research_summary"]["total_videos_analyzed"] = len(all_videos)
        
        self.logger.info("✅ YouTubeResearchAgent: Research complete - analyzed %d videos", len(all_videos))
        return all_research_data

    def _research_single_term(self, search_term: str, max_results: int) -> Dict:
        """
        Research videos for a single search term.
        
        Args:
            search_term: The search term to research
            max_results: Maximum number of results to fetch
            
        Returns:
            Dictionary containing videos and metadata for this search term
        """
        try:
            # Search for videos
            search_response = self.youtube.search().list(
                q=search_term,
                part='id,snippet',
                type='video',
                order='relevance',  # Get most relevant/popular videos
                maxResults=max_results,
                relevanceLanguage='en',  # Focus on English content
                safeSearch='moderate'
            ).execute()
            
            video_ids = [item['id']['videoId'] for item in search_response['items']]
            
            if not video_ids:
                return self._empty_term_result()
            
            # Get detailed video statistics
            videos_response = self.youtube.videos().list(
                part='statistics,snippet,contentDetails',
                id=','.join(video_ids)
            ).execute()
            
            videos = []
            for video in videos_response['items']:
                video_data = self._extract_video_data(video)
                if video_data:
                    videos.append(video_data)
            
            return {
                "search_term": search_term,
                "total_results": len(videos),
                "videos": videos
            }
            
        except HttpError as e:
            self.logger.error("YouTube API error for term '%s': %s", search_term, e, exc_info=True)
            return self._empty_term_result()

    def _extract_video_data(self, video_item: Dict) -> Optional[Dict]:
        """
        Extract relevant data from a YouTube video API response.
        
        Args:
            video_item: Raw video data from YouTube API
            
        Returns:
            Cleaned video data dictionary or None if extraction fails
        """
        try:
            snippet = video_item['snippet']
            statistics = video_item.get('statistics', {})
            
            # Parse view count safely
            view_count = int(statistics.get('viewCount', 0))
            like_count = int(statistics.get('likeCount', 0))
            comment_count = int(statistics.get('commentCount', 0))
            
            # Calculate engagement rate
            engagement_rate = 0
            if view_count > 0:
                engagement_rate = ((like_count + comment_count) / view_count) * 100
            
            return {
                "video_id": video_item['id'],
                "title": snippet['title'],
                "description": snippet.get('description', '')[:500],  # First 500 chars
                "channel_name": snippet['channelTitle'],
                "published_at": snippet['publishedAt'],
                "view_count": view_count,
                "like_count": like_count,
                "comment_count": comment_count,
                "engagement_rate": round(engagement_rate, 4),
                "thumbnail_url": snippet['thumbnails'].get('high', {}).get('url', '')
            }
            
        except (KeyError, ValueError, TypeError) as e:
            self.logger.warning("Error extracting video data for video ID '%s': %s", video_item.get('id', 'Unknown ID'), e)
            return None

    def _analyze_competitive_data(self, all_videos: List[Dict]) -> Dict:
        """
        Analyze all collected video data to generate competitive insights.
        
        Args:
            all_videos: List of all video data from research
            
        Returns:
            Competitive analysis insights
        """
        if not all_videos:
            return {
                "total_videos_analyzed": 0, # Ensure this key is present for consistency
                "average_views": 0,
                "average_engagement_rate": 0,
                "top_performing_videos": [],
                "common_title_words": [],
                "channel_analysis": {}
            }

        # --- Data Preparation and Recency Calculation ---
        view_counts = []
        engagement_rates = []
        days_since_published_list = []

        for video in all_videos:
            try:
                # Parse published_at string to datetime object
                published_date_str = video.get('published_at')
                if published_date_str:
                    # Handle 'Z' for UTC and make it offset-aware
                    published_date = datetime.fromisoformat(published_date_str.replace('Z', '+00:00'))
                    # Calculate days_since_published
                    days_diff = (datetime.now(timezone.utc) - published_date).days
                    video['days_since_published'] = days_diff
                    days_since_published_list.append(days_diff)
                else:
                    # Assign a high value for days_since_published if no date, making it less recent
                    video['days_since_published'] = 365 * 10 # Default to 10 years old
                    days_since_published_list.append(365 * 10)

                view_counts.append(video.get('view_count', 0))
                engagement_rates.append(video.get('engagement_rate', 0.0))
            except Exception as e:
                self.logger.error(f"Error processing video data for recency/normalization: {video.get('video_id')}, {e}")
                # Assign default values if processing fails for a video
                video['days_since_published'] = 365 * 10
                if video.get('view_count') is None : view_counts.append(0) # Check if key exists before appending
                else: view_counts.append(video.get('view_count',0))

                if video.get('engagement_rate') is None : engagement_rates.append(0.0)
                else: engagement_rates.append(video.get('engagement_rate',0.0))

                days_since_published_list.append(365*10) # Ensure list has an entry

        # Calculate overall averages (remains based on all videos)
        total_views_sum = sum(video['view_count'] for video in all_videos) # Renamed for clarity
        total_engagement_sum = sum(video['engagement_rate'] for video in all_videos) # Renamed for clarity
        
        average_views = total_views_sum / len(all_videos) if all_videos else 0
        average_engagement_rate = total_engagement_sum / len(all_videos) if all_videos else 0

        # --- Normalization (values needed for next block) ---
        min_views = min(view_counts) if view_counts else 0
        max_views = max(view_counts) if view_counts else 0
        min_engagement = min(engagement_rates) if engagement_rates else 0
        max_engagement = max(engagement_rates) if engagement_rates else 0
        min_days = min(days_since_published_list) if days_since_published_list else 0
        max_days = max(days_since_published_list) if days_since_published_list else 0
        
        # --- Normalization Loop ---
        for video in all_videos:
            # Views
            if max_views > min_views:
                video['normalized_views'] = (video.get('view_count', 0) - min_views) / (max_views - min_views)
            else:
                video['normalized_views'] = 0.5 # Neutral value if all views are same or no views

            # Engagement
            if max_engagement > min_engagement:
                video['normalized_engagement'] = (video.get('engagement_rate', 0.0) - min_engagement) / (max_engagement - min_engagement)
            else:
                video['normalized_engagement'] = 0.5 # Neutral value

            # Recency (1 - normalized_days, so higher is better/more recent)
            days_published = video.get('days_since_published', max_days) # Default to max_days if somehow missing
            if max_days > min_days:
                video['normalized_recency'] = 1 - ((days_published - min_days) / (max_days - min_days))
            else:
                video['normalized_recency'] = 0.5 # Neutral value

        # --- Scoring and Sorting ---
        w_views = 0.4
        w_engagement = 0.4
        w_recency = 0.2

        for video in all_videos:
            video['composite_score'] = (video.get('normalized_views', 0) * w_views) + \
                                     (video.get('normalized_engagement', 0) * w_engagement) + \
                                     (video.get('normalized_recency', 0) * w_recency)
        
        # Sort by composite_score to find top performing videos
        sorted_videos = sorted(all_videos, key=lambda x: x.get('composite_score', 0), reverse=True)
        top_videos = sorted_videos[:3] # Select the top 3 videos based on composite score
        
        # Analyze title patterns (uses all_videos, which is correct)
        title_words = []
        for video in all_videos: # This loop should remain to analyze all titles
            words = video['title'].lower().split()
            title_words.extend(words)
        
        # Count word frequency for common patterns
        word_count = {}
        for word in title_words:
            # Clean word and count only meaningful words
            clean_word = ''.join(c for c in word if c.isalnum())
            if len(clean_word) > 3:  # Only words longer than 3 characters
                word_count[clean_word] = word_count.get(clean_word, 0) + 1
        
        # Get most common title words
        common_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Channel analysis
        channel_stats = {}
        for video in all_videos:
            channel = video['channel_name']
            if channel not in channel_stats:
                channel_stats[channel] = {
                    "video_count": 0,
                    "total_views": 0,
                    "avg_engagement": 0
                }
            
            channel_stats[channel]["video_count"] += 1
            channel_stats[channel]["total_views"] += video['view_count']
            channel_stats[channel]["avg_engagement"] += video['engagement_rate']
        
        # Calculate channel averages
        for channel, stats in channel_stats.items():
            stats["avg_engagement"] = stats["avg_engagement"] / stats["video_count"]
        
        return {
            "total_videos_analyzed": len(all_videos),
            "average_views": int(average_views),
            "average_engagement_rate": round(average_engagement_rate, 4),
            "top_performing_videos": [
                {
                    "title": video['title'],
                    "channel": video['channel_name'],
                    "views": video['view_count'],
                    "engagement_rate": video['engagement_rate']
                }
                for video in top_videos
            ],
            "common_title_words": [word for word, count in common_words],
            "channel_analysis": channel_stats
        }

    def _empty_research_result(self) -> Dict:
        """Return empty research result structure for error cases."""
        return {
            "research_summary": {
                "total_search_terms": 0,
                "total_videos_analyzed": 0,
                "search_terms_used": []
            },
            "term_results": {},
            "competitive_analysis": {
                "average_views": 0,
                "average_engagement_rate": 0,
                "top_performing_videos": [],
                "common_title_words": [], # Changed for consistency
                "channel_analysis": {}
            }
        }

    def _empty_term_result(self) -> Dict:
        """Return empty term result structure for error cases."""
        return {
            "search_term": "",
            "total_results": 0,
            "videos": []
        }


# For testing the YouTube Research Agent
if __name__ == '__main__':
    # Basic logging for the test script itself
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main_logger = logging.getLogger(__name__) # Use a logger for the test script

    main_logger.info("🧪 Testing YouTubeResearchAgent...")

    load_dotenv() # Ensure .env is loaded for the test script
    youtube_api_key = os.getenv("YOUTUBE_API_KEY")

    if not youtube_api_key:
        main_logger.error("🔴 YOUTUBE_API_KEY environment variable not set. Skipping YouTubeResearchAgent test.")
        main_logger.info("Ensure .env file exists and YOUTUBE_API_KEY is set.")
    else:
        main_logger.info("✅ YOUTUBE_API_KEY is set. Proceeding with agent test.")
        try:
            agent = YouTubeResearchAgent()

            # Test with sample search terms (like those from Step 4)
        test_search_terms = [
            "AI tutorial for beginners",
            "machine learning explained",
            "python programming guide"
        ]
        
        research_results = agent.research_competitive_content(test_search_terms, max_results_per_term=3)
        
        main_logger.info("✅ Research completed successfully!")
        main_logger.info("📊 Total videos analyzed: %d", research_results['research_summary']['total_videos_analyzed'])
        main_logger.info("📈 Average views: %s", f"{research_results['competitive_analysis']['average_views']:,}")
        main_logger.info("🎯 Average engagement rate: %.4f%%", research_results['competitive_analysis']['average_engagement_rate'])
        
    except Exception as e:
            main_logger.exception("❌ Test failed: %s", e) # Use logger.exception for errors in test
            main_logger.info("Make sure YOUTUBE_API_KEY is set in your .env file and is valid.")