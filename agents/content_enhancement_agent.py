import os
from typing import Dict, List
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
import re
import json
from collections import Counter # Added for n-gram analysis
import logging
import time # For retry delay
from typing import Optional # For retry return type
    by combining original video content analysis with competitive research data.
    This agent implements Step 6 of the content optimization pipeline.
    """

    def __init__(self):
        """
        Initializes the agent and sets up the LlamaIndex LLM configuration.
        """
        self.logger = logging.getLogger(__name__)
        self._initialize_llamaindex_ollama()

    def _initialize_llamaindex_ollama(self):
        """
        Initializes LlamaIndex settings to use Ollama with endpoint and model
        from environment variables.
        """
        load_dotenv()
        model_endpoint = os.getenv("MODEL_ENDPOINT")
        model_name = os.getenv("MODEL_NAME")

        if not model_endpoint:
            self.logger.error("MODEL_ENDPOINT environment variable not set for ContentEnhancementAgent.")
            raise ValueError("MODEL_ENDPOINT environment variable not set for ContentEnhancementAgent.")
        if not model_name:
            self.logger.warning("MODEL_NAME environment variable not set for ContentEnhancementAgent. Using a default or letting Ollama decide.")

        llm = Ollama(base_url=model_endpoint, model=model_name, request_timeout=120.0)
        Settings.llm = llm
        self.logger.info("✅ ContentEnhancementAgent: LlamaIndex configured to use Ollama: endpoint=%s, model=%s", model_endpoint, model_name or 'default')

    def _call_llm_with_retry(self, prompt: str, max_retries: int = 2, delay_seconds: int = 5) -> Optional[str]:
        """
        Calls the LLM with a given prompt and retries on failure.
        """
        for attempt in range(max_retries):
            try:
                self.logger.info("LLM call attempt %d/%d for ContentEnhancementAgent", attempt + 1, max_retries)
                response = Settings.llm.complete(prompt)
                return response.text.strip()
            except Exception as e: # Broad exception catch
                self.logger.error(
                    "LLM call failed in ContentEnhancementAgent on attempt %d/%d: %s",
                    attempt + 1,
                    max_retries,
                    e,
                    exc_info=True
                )
                if attempt < max_retries - 1:
                    self.logger.info("Waiting %d seconds before next retry...", delay_seconds)
                    time.sleep(delay_seconds)
                else:
                    self.logger.error("All LLM retry attempts failed for prompt in ContentEnhancementAgent.")
                    return None
        return None

    def enhance_content(self, video_summary: str, search_terms: List[str], research_data: Dict) -> Dict:
        """
        Generate optimized YouTube title and description using video content and research data.
        
        Args:
            video_summary: Video summary from Step 4 content analysis
            search_terms: Search terms from Step 4
            research_data: Competitive research data from Step 5
            
        Returns:
            Dictionary containing optimized title and description
        """
        if not Settings.llm:
            self.logger.error("🚨 LLM not initialized in ContentEnhancementAgent. Re-initializing...")
            self._initialize_llamaindex_ollama() # Attempt re-init

        self.logger.info("🚀 ContentEnhancementAgent: Starting content enhancement...")
        
        # Extract competitive insights
        competitive_insights = self._extract_competitive_insights(research_data)
        
        # Generate optimized title
        optimized_title = self._generate_optimized_title(
            video_summary, search_terms, competitive_insights
        )
        
        # Generate optimized description
        optimized_description = self._generate_optimized_description(
            video_summary, search_terms, competitive_insights, research_data
        )
        
        return {
            "optimized_title": optimized_title,
            "optimized_description": optimized_description,
            "enhancement_metadata": {
                "search_terms_used": search_terms,
                "competitive_videos_analyzed": competitive_insights.get("total_videos_analyzed", 0),
                "avg_competitor_views": competitive_insights.get("average_views", 0),
                "avg_engagement_rate": competitive_insights.get("average_engagement_rate", 0),
                "top_keywords": competitive_insights.get("common_keywords", [])[:5],
                "successful_patterns": competitive_insights.get("successful_patterns", []),
                "top_performing_titles": competitive_insights.get("top_performing_titles", [])[:3]
            }
        }

    def _extract_competitive_insights(self, research_data: Dict) -> Dict:
        """
        Extract key insights from competitive research data for content optimization.
        
        Args:
            research_data: Raw research data from YouTubeResearchAgent
            
        Returns:
            Processed insights for content enhancement
        """
        if not research_data or "competitive_analysis" not in research_data:
            return {
                "total_videos_analyzed": 0,
                "average_views": 0,
                "average_engagement_rate": 0,
                "top_performing_titles": [],
                "common_keywords": [],
                "successful_patterns": []
            }
        
        analysis = research_data["competitive_analysis"]
        
        # Extract top performing titles
        top_titles = []
        if "top_performing_videos" in analysis:
            top_titles = [video["title"] for video in analysis["top_performing_videos"]]
        
        # Process common title words into meaningful keywords
        common_keywords = analysis.get("common_title_words", [])[:10]
        
        # Identify successful patterns from top titles
        patterns = self._identify_title_patterns(top_titles)
        
        return {
            "total_videos_analyzed": analysis.get("total_videos_analyzed", 0),
            "average_views": analysis.get("average_views", 0),
            "average_engagement_rate": analysis.get("average_engagement_rate", 0),
            "top_performing_titles": top_titles,
            "common_keywords": common_keywords,
            "successful_patterns": patterns
        }

    def _identify_title_patterns(self, titles: List[str]) -> List[str]:
        """
        Identify common patterns in successful video titles.
        
        Args:
            titles: List of top performing video titles
            
        Returns:
            List of identified successful patterns
        """
        if not titles:
            return []

        stopwords = set([
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "should",
            "can", "could", "may", "might", "must", "and", "but", "or", "nor",
            "for", "so", "yet", "in", "on", "at", "by", "from", "to", "with", "about",
            "of", "how", "what", "when", "where", "why", "which", "my", "your", "its",
            "tutorial", "guide", "video", "explained", "review", "vs", "new", "update",
            "tips", "tricks", "secret", "hack", "learn", "build", "create", "discover", "best", "top"
            # Added some common YouTube specific words that might not be general patterns
        ])

        bigram_counts = Counter()
        trigram_counts = Counter()

        for title in titles:
            # Basic tokenization: lowercase, split by space, remove non-alphanumeric chars from ends of words
            words = [re.sub(r'[^a-z0-9#+-]', '', word.strip().lower()) for word in title.split()]
            # Filter stopwords and very short words (potential noise from punctuation removal)
            tokens = [word for word in words if word and word not in stopwords and len(word) > 2]

            # Generate n-grams
            bigrams = list(zip(tokens, tokens[1:]))
            trigrams = list(zip(tokens, tokens[1:], tokens[2:]))

            bigram_counts.update(bigrams)
            trigram_counts.update(trigrams)

        # Identify frequent n-grams (appearing more than once)
        # And also consider n-grams that make up a significant portion of a short list of titles
        min_frequency = 1
        if len(titles) >= 5 : # If we have at least 5 titles, require more than 1 occurrence
             min_frequency = 2
        if len(titles) < 3 and len(titles) > 0: # For very few titles, allow single occurrences of longer phrases
            min_frequency = 1


        frequent_bigrams = {bg: count for bg, count in bigram_counts.items() if count >= min_frequency}
        frequent_trigrams = {tg: count for tg, count in trigram_counts.items() if count >= min_frequency}

        # Sort by frequency
        sorted_bigrams = sorted(frequent_bigrams.items(), key=lambda item: item[1], reverse=True)
        sorted_trigrams = sorted(frequent_trigrams.items(), key=lambda item: item[1], reverse=True)

        # Format output (top N, e.g., 3-5 combined)
        identified_patterns = []
        for bg, count in sorted_bigrams[:3]: # Top 3 bigrams
            identified_patterns.append(" ".join(bg))

        for tg, count in sorted_trigrams[:2]: # Top 2 trigrams
            # Avoid adding trigrams that are just extensions of already added bigrams if they are too similar
            trigram_str = " ".join(tg)
            is_covered = False
            for pattern in identified_patterns:
                if trigram_str.startswith(pattern) or trigram_str.endswith(pattern):
                    is_covered = True
                    break
            if not is_covered:
                 identified_patterns.append(trigram_str)

        # Limit total patterns to avoid overwhelming the prompt
        return identified_patterns[:5]

    def _generate_optimized_title(self, video_summary: str, search_terms: List[str], competitive_insights: Dict) -> str:
        """
        Generate an optimized YouTube title using AI analysis.
        
        Args:
            video_summary: Original video summary
            search_terms: YouTube search terms
            competitive_insights: Processed competitive data
            
        Returns:
            Optimized title string
        """
        # Prepare competitive context
        top_titles = competitive_insights.get("top_performing_titles", [])
        common_keywords = competitive_insights.get("common_keywords", [])
        patterns = competitive_insights.get("successful_patterns", [])
        
        competitive_context = ""
        if top_titles:
            competitive_context += f"\nTop performing titles in this niche:\n"
            for i, title in enumerate(top_titles[:3], 1):
                competitive_context += f"{i}. {title}\n"
        
        if common_keywords:
            competitive_context += f"\nPopular keywords: {', '.join(common_keywords[:8])}\n"
        
        if patterns:
            competitive_context += f"\nSuccessful title patterns: {', '.join(patterns)}\n"
        
        title_prompt = f"""You are a YouTube optimization expert specializing in creating high-performing video titles.

Create an engaging, click-worthy title for a video with this content:

VIDEO SUMMARY:
{video_summary}

TARGET SEARCH TERMS: {', '.join(search_terms)}

COMPETITIVE RESEARCH DATA:{competitive_context}

TITLE OPTIMIZATION REQUIREMENTS:
1. Maximum 70 characters is ideal for YouTube display.
2. Naturally incorporate the primary keyword from search terms.
3. Use natural language and avoid keyword stuffing.
4. Balance click-worthiness with accurate content representation.
5. Create curiosity and urgency where appropriate.
6. If competitive analysis suggests proven patterns (e.g., "How to...", "Top 5..."), consider using them if fitting.
7. Avoid clickbait - the title must be accurate to the video's content.
8. Consider emotional triggers (e.g., excitement, problem-solving, learning).

Generate ONE optimized title that balances SEO optimization with high engagement. Return only the title, nothing else."""

        self.logger.info("🔄 ContentEnhancementAgent: Generating optimized title with retry...")
        optimized_title_raw = self._call_llm_with_retry(title_prompt)

        primary_term = search_terms[0] if search_terms else "Video Content"
        fallback_title = f"Complete Guide to {primary_term} - Everything You Need to Know"

        if optimized_title_raw is not None and optimized_title_raw.strip():
            optimized_title = self._clean_generated_text(optimized_title_raw)
            if len(optimized_title) > 100: # Max YouTube title length is 100
                self.logger.warning("Generated title exceeded 100 chars, truncating: %s", optimized_title)
                optimized_title = optimized_title[:97] + "..."
            
            if not optimized_title.strip(): # Check if cleaning resulted in empty string
                self.logger.warning("⚠️ Title became empty after cleaning. Using fallback.")
                optimized_title = fallback_title
            else:
                self.logger.info("✅ ContentEnhancementAgent: Optimized title generated: %s", optimized_title)
        else:
            self.logger.error("❌ ContentEnhancementAgent: Failed to generate title after all retries or got empty response. Using fallback.")
            optimized_title = fallback_title
            self.logger.info("Using fallback title: %s", optimized_title)

        return optimized_title

    def _generate_optimized_description(self, video_summary: str, search_terms: List[str], competitive_insights: Dict, research_data: Dict) -> str:
        """
        Generate an optimized YouTube description using AI analysis.
        
        Args:
            video_summary: Original video summary
            search_terms: YouTube search terms
            competitive_insights: Processed competitive data
            research_data: Full research data for additional context
            
        Returns:
            Optimized description string
        """
        # Extract top performing descriptions for pattern analysis
        top_descriptions = []
        if "term_results" in research_data:
            for term_data in research_data["term_results"].values():
                for video in term_data.get("videos", [])[:2]:  # Top 2 per term
                    if video.get("description"):
                        top_descriptions.append(video["description"][:200])  # First 200 chars
        
        competitive_context = ""
        if top_descriptions:
            competitive_context += "\nExamples of successful descriptions in this niche:\n"
            for i, desc in enumerate(top_descriptions[:3], 1):
                competitive_context += f"{i}. {desc}...\n"
        
        description_prompt = f"""You are a YouTube SEO expert creating optimized video descriptions that rank well and engage viewers.

Create a comprehensive, SEO-optimized description for this video:

VIDEO SUMMARY:
{video_summary}

TARGET SEARCH TERMS: {', '.join(search_terms)}

COMPETITIVE EXAMPLES:{competitive_context}

DESCRIPTION REQUIREMENTS:
1. Hook: Start with a compelling hook (first 1-2 sentences, crucial for viewer retention).
2. Keyword Integration: Naturally incorporate the primary search term early and secondary search terms throughout the text. Avoid keyword stuffing.
3. Value Proposition: Clearly state what viewers will learn or gain from watching.
4. Readability: Use bullet points or short paragraphs for key information.
5. Call to Action: Include a call-to-action (e.g., subscribe, like, comment, check out a link).
6. Hashtags: Include 3-5 relevant hashtags at the end.
7. Authenticity: Maintain an authentic and engaging tone.
8. Length: Aim for 200-300 words.

Structure the description with the following sections:
- Engaging opening paragraph (the hook).
- A brief overview of what the video covers.
- A "In this video, you'll learn:" or "What to expect:" section with 2-4 bullet points highlighting key takeaways or topics covered.
- Additional details or context if necessary.
- Call to action.
- Relevant hashtags (e.g., #Keyword1 #Keyword2 #VideoTopic).

Generate a complete, ready-to-use YouTube description."""

        self.logger.info("🔄 ContentEnhancementAgent: Generating optimized description with retry...")
        optimized_description_raw = self._call_llm_with_retry(description_prompt)

        if optimized_description_raw is not None and optimized_description_raw.strip():
            optimized_description = self._clean_generated_text(optimized_description_raw)
            
            if not optimized_description.strip() or len(optimized_description) < 50: # Basic validity check for description
                 self.logger.warning("⚠️ Description too short or became empty after cleaning. Length: %d. Using fallback.", len(optimized_description))
                 optimized_description = self._generate_fallback_description(video_summary, search_terms)
                 self.logger.info("Generated fallback description for too short/empty main generation.")
            else:
                optimized_description = self._enhance_description_formatting(optimized_description, search_terms)
                self.logger.info("✅ ContentEnhancementAgent: Optimized description generated (length: %d chars)", len(optimized_description))
        else:
            self.logger.error("❌ ContentEnhancementAgent: Failed to generate description after all retries or got empty response. Using fallback.")
            optimized_description = self._generate_fallback_description(video_summary, search_terms)
            self.logger.info("Generated fallback description due to retry failure.")
            
        return optimized_description

    def _clean_generated_text(self, text: str) -> str:
        """
        Clean up AI-generated text by removing unwanted formatting.
        
        Args:
            text: Raw AI-generated text
            
        Returns:
            Cleaned text
        """
        # Remove quotes if the entire text is wrapped in them
        text = text.strip()
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            text = text[1:-1]
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        
        # Clean up extra whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double
        text = text.strip()
        
        return text

    def _enhance_description_formatting(self, description: str, search_terms: List[str]) -> str:
        """
        Enhance description formatting and ensure proper structure.
        
        Args:
            description: Generated description
            search_terms: Search terms to potentially add as hashtags
            
        Returns:
            Enhanced and formatted description
        """
        # Ensure description has proper call-to-action if missing
        if "subscribe" not in description.lower() and "like" not in description.lower():
            description += "\n\n👍 If you found this helpful, please like and subscribe for more content!"
        
        # Add hashtags if not present
        if "#" not in description:
            hashtags = []
            for term in search_terms[:3]:  # Max 3 hashtags from search terms
                # Convert to hashtag format
                hashtag = "#" + "".join(word.capitalize() for word in term.split() if word.isalnum())
                if len(hashtag) > 3 and len(hashtag) < 25:  # Reasonable hashtag length
                    hashtags.append(hashtag)
            
            if hashtags:
                description += f"\n\n{' '.join(hashtags)}"
        
        return description

    def _generate_fallback_description(self, video_summary: str, search_terms: List[str]) -> str:
        """
        Generate a fallback description when AI generation fails.
        
        Args:
            video_summary: Original video summary
            search_terms: Search terms
            
        Returns:
            Basic but functional description
        """
        primary_term = search_terms[0] if search_terms else "this topic"
        
        description = f"""In this video, we dive deep into {primary_term} and provide you with comprehensive insights and practical information.

{video_summary}

What you'll learn:
• Key concepts and fundamentals
• Practical applications and examples
• Tips and best practices

If you found this content valuable, please like this video and subscribe to our channel for more educational content!

{' '.join('#' + ''.join(term.split()) for term in search_terms[:3])}"""
        
        return description


# For testing the Content Enhancement Agent
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main_logger = logging.getLogger(__name__)
    main_logger.info("🧪 Testing ContentEnhancementAgent...")

    load_dotenv() # Ensure .env is loaded for the test script
    model_endpoint = os.getenv("MODEL_ENDPOINT")
    model_name = os.getenv("MODEL_NAME")
    # YOUTUBE_API_KEY is not directly used by this agent for its core LLM calls,
    # but its input (research_data) comes from YouTubeResearchAgent.
    # For a focused unit test of ContentEnhancementAgent, only LLM vars are strictly needed.

    if not model_endpoint or not model_name:
        main_logger.error("🔴 MODEL_ENDPOINT and/or MODEL_NAME environment variables not set. Skipping ContentEnhancementAgent test.")
        main_logger.info("Ensure .env file exists and these variables are set (e.g., MODEL_ENDPOINT=http://localhost:11434, MODEL_NAME=llama3)")
    else:
        main_logger.info("✅ MODEL_ENDPOINT and MODEL_NAME are set. Proceeding with agent test.")
        try:
            agent = ContentEnhancementAgent()

            # Test data (simulating Step 4 and Step 5 outputs)
        test_video_summary = """This video provides a comprehensive introduction to artificial intelligence and machine learning. 
        The content covers fundamental concepts, practical applications, and real-world examples of AI implementation. 
        Key topics include neural networks, data processing, and the future potential of AI technology."""
        
        test_search_terms = ["AI tutorial for beginners", "machine learning explained", "artificial intelligence guide"]
        
        test_research_data = {
            "competitive_analysis": {
                "total_videos_analyzed": 15,
                "average_views": 125000,
                "average_engagement_rate": 3.5,
                "top_performing_videos": [
                    {"title": "Complete AI Tutorial for Beginners in 2024", "channel": "Tech Channel", "views": 500000, "engagement_rate": 5.2},
                    {"title": "Machine Learning Explained Simply", "channel": "AI Academy", "views": 300000, "engagement_rate": 4.8}
                ],
                "common_title_words": ["tutorial", "explained", "beginners", "complete", "guide", "2024", "simple", "learn"]
            }
        }
        
        results = agent.enhance_content(test_video_summary, test_search_terms, test_research_data)
        
        main_logger.info("✅ Content enhancement completed successfully!")
        main_logger.info("📝 Optimized Title: %s", results['optimized_title'])
        main_logger.info("📄 Description length: %d characters", len(results['optimized_description']))
        main_logger.info("📊 Metadata: %s", results['enhancement_metadata'])
        
    except Exception as e:
            main_logger.exception("❌ Test failed: %s", e)
            main_logger.info("Make sure MODEL_ENDPOINT and MODEL_NAME are set in your .env file (and YOUTUBE_API_KEY if testing full flow).")