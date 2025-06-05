import os
from typing import Dict, List
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
import re
import json

class ContentEnhancementAgent:
    """
    An agent responsible for generating optimized YouTube titles and descriptions
    by combining original video content analysis with competitive research data.
    This agent implements Step 6 of the content optimization pipeline.
    """

    def __init__(self):
        """
        Initializes the agent and sets up the LlamaIndex LLM configuration.
        """
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
            raise ValueError("MODEL_ENDPOINT environment variable not set for ContentEnhancementAgent.")
        if not model_name:
            print("Warning: MODEL_NAME environment variable not set for ContentEnhancementAgent. Using a default or letting Ollama decide.")

        llm = Ollama(base_url=model_endpoint, model=model_name, request_timeout=120.0)
        Settings.llm = llm
        print(f"✅ ContentEnhancementAgent: LlamaIndex configured to use Ollama: endpoint={model_endpoint}, model={model_name or 'default'}")

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
            print("🚨 LLM not initialized in ContentEnhancementAgent. Re-initializing...")
            self._initialize_llamaindex_ollama()

        print(f"🚀 ContentEnhancementAgent: Starting content enhancement...")
        
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
        patterns = []
        
        if not titles:
            return patterns
        
        # Check for common title patterns
        patterns_to_check = [
            ("Question format", r"\?"),
            ("How to format", r"(?i)how\s+to"),
            ("Number format", r"^\d+"),
            ("Year/Date", r"202[0-9]"),
            ("Tutorial/Guide", r"(?i)(tutorial|guide|walkthrough)"),
            ("Best/Top", r"(?i)(best|top|ultimate)"),
            ("Complete/Full", r"(?i)(complete|full|comprehensive)"),
            ("Quick/Fast", r"(?i)(quick|fast|easy|simple)"),
            ("Review/Explained", r"(?i)(review|explained|breakdown)")
        ]
        
        for pattern_name, regex in patterns_to_check:
            count = sum(1 for title in titles if re.search(regex, title))
            if count > 0:
                percentage = (count / len(titles)) * 100
                if percentage >= 30:  # If 30% or more titles use this pattern
                    patterns.append(f"{pattern_name} ({percentage:.0f}%)")
        
        return patterns

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
1. Maximum 60 characters (optimal for YouTube display)
2. Include primary keyword from search terms naturally
3. Create curiosity and urgency
4. Use proven patterns from competitive analysis
5. Avoid clickbait - be accurate to content
6. Consider emotional triggers (excitement, solving problems, etc.)

Generate ONE optimized title that balances SEO optimization with engagement. Return only the title, nothing else."""

        try:
            print("🔄 ContentEnhancementAgent: Generating optimized title...")
            title_response = Settings.llm.complete(title_prompt)
            optimized_title = title_response.text.strip()
            
            # Clean up title (remove quotes, extra formatting)
            optimized_title = self._clean_generated_text(optimized_title)
            
            # Ensure title length is reasonable
            if len(optimized_title) > 100:
                optimized_title = optimized_title[:97] + "..."
            
            print(f"✅ ContentEnhancementAgent: Optimized title generated")
            return optimized_title
            
        except Exception as e:
            print(f"❌ ContentEnhancementAgent: Error generating title: {e}")
            # Fallback title
            primary_term = search_terms[0] if search_terms else "Video Content"
            return f"Complete Guide to {primary_term} - Everything You Need to Know"

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
1. Start with a compelling hook (first 125 characters are crucial)
2. Naturally incorporate all search terms throughout the text
3. Include clear value proposition and what viewers will learn
4. Use bullet points or sections for readability
5. Add call-to-action (subscribe, like, comment)
6. Include relevant hashtags at the end
7. Maintain authentic tone while being SEO-optimized
8. Length: 200-300 words optimal

Structure the description with:
- Engaging opening paragraph
- What the video covers
- Key takeaways/benefits
- Call to action
- Relevant hashtags

Generate a complete, ready-to-use YouTube description."""

        try:
            print("🔄 ContentEnhancementAgent: Generating optimized description...")
            description_response = Settings.llm.complete(description_prompt)
            optimized_description = description_response.text.strip()
            
            # Clean up description
            optimized_description = self._clean_generated_text(optimized_description)
            
            # Ensure proper formatting and add engagement elements
            optimized_description = self._enhance_description_formatting(optimized_description, search_terms)
            
            print(f"✅ ContentEnhancementAgent: Optimized description generated")
            return optimized_description
            
        except Exception as e:
            print(f"❌ ContentEnhancementAgent: Error generating description: {e}")
            # Fallback description
            return self._generate_fallback_description(video_summary, search_terms)

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
    print("🧪 Testing ContentEnhancementAgent...")
    
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
        
        print(f"✅ Content enhancement completed successfully!")
        print(f"📝 Optimized Title: {results['optimized_title']}")
        print(f"📄 Description length: {len(results['optimized_description'])} characters")
        print(f"📊 Metadata: {results['enhancement_metadata']}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Make sure MODEL_ENDPOINT and MODEL_NAME are set in your .env file") 