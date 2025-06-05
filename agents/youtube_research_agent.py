import os
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import time

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
            print(f"✅ YouTubeResearchAgent: YouTube Data API v3 initialized successfully")
        except Exception as e:
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
            print("⚠️ No search terms provided for YouTube research")
            return self._empty_research_result()
        
        print(f"🔍 YouTubeResearchAgent: Starting research for {len(search_terms)} search terms")
        
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
            print(f"🔎 Researching: '{term}'")
            try:
                term_data = self._research_single_term(term, max_results_per_term)
                all_research_data["term_results"][term] = term_data
                all_videos.extend(term_data["videos"])
                
                # Rate limiting - YouTube API has quotas
                time.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Error researching term '{term}': {e}")
                all_research_data["term_results"][term] = self._empty_term_result()
        
        # Generate competitive analysis
        all_research_data["competitive_analysis"] = self._analyze_competitive_data(all_videos)
        all_research_data["research_summary"]["total_videos_analyzed"] = len(all_videos)
        
        print(f"✅ YouTubeResearchAgent: Research complete - analyzed {len(all_videos)} videos")
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
            print(f"YouTube API error for term '{search_term}': {e}")
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
            print(f"Error extracting video data: {e}")
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
                "average_views": 0,
                "average_engagement_rate": 0,
                "top_performing_videos": [],
                "common_title_patterns": [],
                "channel_analysis": {}
            }
        
        # Calculate averages
        total_views = sum(video['view_count'] for video in all_videos)
        total_engagement = sum(video['engagement_rate'] for video in all_videos)
        
        average_views = total_views / len(all_videos)
        average_engagement_rate = total_engagement / len(all_videos)
        
        # Find top performing videos (by engagement rate)
        top_videos = sorted(all_videos, key=lambda x: x['engagement_rate'], reverse=True)[:3]
        
        # Analyze title patterns
        title_words = []
        for video in all_videos:
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
                "common_title_patterns": [],
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
    print("🧪 Testing YouTubeResearchAgent...")
    
    try:
        agent = YouTubeResearchAgent()
        
        # Test with sample search terms (like those from Step 4)
        test_search_terms = [
            "AI tutorial for beginners",
            "machine learning explained",
            "python programming guide"
        ]
        
        research_results = agent.research_competitive_content(test_search_terms, max_results_per_term=3)
        
        print(f"✅ Research completed successfully!")
        print(f"📊 Total videos analyzed: {research_results['research_summary']['total_videos_analyzed']}")
        print(f"📈 Average views: {research_results['competitive_analysis']['average_views']:,}")
        print(f"🎯 Average engagement rate: {research_results['competitive_analysis']['average_engagement_rate']:.4f}%")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Make sure YOUTUBE_API_KEY is set in your .env file") 