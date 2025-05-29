#!/usr/bin/env python3
"""
AI-Powered Twitter Bot
======================
A fully automated Twitter bot that scrapes tweets, analyzes sentiment,
generates AI responses using Gemini, and posts replies, quote tweets, and daily threads.

Author: AI Assistant
License: MIT
"""

import os
import json
import time
import random
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import requests
from textblob import TextBlob
import tweepy
import snscrape.modules.twitter as sntwitter
import google.generativeai as genai
from urllib.parse import urlparse
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TwitterBot:
    """Main Twitter bot class handling all operations."""
    
    def __init__(self):
        """Initialize the bot with API credentials and configuration."""
        self.setup_credentials()
        self.setup_twitter_api()
        self.setup_gemini_api()
        self.tweet_count_today = 0
        self.max_daily_tweets = 50  # Stay well under 1500/month limit
        self.processed_tweets = set()
        self.personalities = ['friendly', 'informative', 'supportive', 'sarcastic', 'poetic', 'funny']
        
    def setup_credentials(self):
        """Load API credentials from environment variables."""
        try:
            self.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
            self.twitter_consumer_key = os.getenv('TWITTER_CONSUMER_KEY')
            self.twitter_consumer_secret = os.getenv('TWITTER_CONSUMER_SECRET')
            self.twitter_access_token = os.getenv('TWITTER_ACCESS_TOKEN')
            self.twitter_access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
            self.gemini_api_key = os.getenv('GEMINI_API_KEY')
            
            # Validate credentials
            required_vars = [
                'TWITTER_BEARER_TOKEN', 'TWITTER_CONSUMER_KEY', 'TWITTER_CONSUMER_SECRET',
                'TWITTER_ACCESS_TOKEN', 'TWITTER_ACCESS_TOKEN_SECRET', 'GEMINI_API_KEY'
            ]
            
            for var in required_vars:
                if not os.getenv(var):
                    raise ValueError(f"Missing required environment variable: {var}")
                    
            logger.info("✅ All credentials loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading credentials: {e}")
            raise
    
    def setup_twitter_api(self):
        """Initialize Twitter API client."""
        try:
            # Initialize Twitter API v2 client
            self.twitter_client = tweepy.Client(
                bearer_token=self.twitter_bearer_token,
                consumer_key=self.twitter_consumer_key,
                consumer_secret=self.twitter_consumer_secret,
                access_token=self.twitter_access_token,
                access_token_secret=self.twitter_access_token_secret,
                wait_on_rate_limit=True
            )
            
            # Initialize Twitter API v1.1 for media upload
            auth = tweepy.OAuth1UserHandler(
                self.twitter_consumer_key,
                self.twitter_consumer_secret,
                self.twitter_access_token,
                self.twitter_access_token_secret
            )
            self.twitter_api_v1 = tweepy.API(auth, wait_on_rate_limit=True)
            
            # Test authentication
            me = self.twitter_client.get_me()
            logger.info(f"✅ Twitter API authenticated as: @{me.data.username}")
            
        except Exception as e:
            logger.error(f"❌ Error setting up Twitter API: {e}")
            raise
    
    def setup_gemini_api(self):
        """Initialize Gemini AI API."""
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-pro')
            logger.info("✅ Gemini API configured successfully")
            
        except Exception as e:
            logger.error(f"❌ Error setting up Gemini API: {e}")
            raise
    
    def scrape_tweets(self, query: str, max_tweets: int = 20) -> List[Dict]:
        """
        Scrape tweets using snscrape.
        
        Args:
            query: Search query or username
            max_tweets: Maximum number of tweets to scrape
            
        Returns:
            List of tweet dictionaries
        """
        tweets = []
        try:
            logger.info(f"🔍 Scraping tweets for query: {query}")
            
            # Create search query with filters for recent, non-retweet content
            search_query = f"{query} -filter:retweets -filter:replies since:{datetime.now().date() - timedelta(days=1)}"
            
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_query).get_items()):
                if i >= max_tweets:
                    break
                
                # Skip old tweets (older than 24 hours)
                if tweet.date < datetime.now(tweet.date.tzinfo) - timedelta(hours=24):
                    continue
                
                # Skip tweets we've already processed
                if tweet.id in self.processed_tweets:
                    continue
                
                tweet_data = {
                    'id': tweet.id,
                    'text': tweet.rawContent,
                    'username': tweet.user.username,
                    'timestamp': tweet.date,
                    'url': tweet.url,
                    'media': [],
                    'hashtags': tweet.hashtags or []
                }
                
                # Extract media URLs if present
                if tweet.media:
                    for media in tweet.media:
                        if hasattr(media, 'fullUrl'):
                            tweet_data['media'].append({
                                'type': 'photo' if 'photo' in media.fullUrl else 'video',
                                'url': media.fullUrl
                            })
                
                tweets.append(tweet_data)
                self.processed_tweets.add(tweet.id)
                
            logger.info(f"✅ Scraped {len(tweets)} tweets")
            return tweets
            
        except Exception as e:
            logger.error(f"❌ Error scraping tweets: {e}")
            return []
    
    def download_media(self, media_url: str, tweet_id: str) -> Optional[str]:
        """
        Download media from URL and save locally.
        
        Args:
            media_url: URL of the media to download
            tweet_id: Tweet ID for filename
            
        Returns:
            Local file path if successful, None otherwise
        """
        try:
            response = requests.get(media_url, timeout=30)
            response.raise_for_status()
            
            # Determine file extension
            parsed_url = urlparse(media_url)
            file_ext = os.path.splitext(parsed_url.path)[1] or '.jpg'
            
            # Create media directory if it doesn't exist
            os.makedirs('media', exist_ok=True)
            
            # Save file
            filename = f"media/tweet_{tweet_id}{file_ext}"
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"📁 Downloaded media: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"❌ Error downloading media: {e}")
            return None
    
    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Analyze sentiment of tweet text using TextBlob.
        
        Args:
            text: Tweet text to analyze
            
        Returns:
            Tuple of (sentiment_label, polarity_score)
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                return "positive", polarity
            elif polarity < -0.1:
                return "negative", polarity
            else:
                return "neutral", polarity
                
        except Exception as e:
            logger.error(f"❌ Error analyzing sentiment: {e}")
            return "neutral", 0.0
    
    def get_personality_for_sentiment(self, sentiment: str) -> str:
        """
        Get personality based on sentiment and time of day.
        
        Args:
            sentiment: Sentiment analysis result
            
        Returns:
            Personality string
        """
        current_hour = datetime.now().hour
        
        # Base personality on sentiment
        sentiment_personalities = {
            'positive': ['friendly', 'funny', 'poetic'],
            'neutral': ['informative', 'sarcastic'],
            'negative': ['supportive', 'friendly']
        }
        
        # Add time-based variation
        if 6 <= current_hour < 12:  # Morning
            base_personalities = sentiment_personalities.get(sentiment, ['friendly'])
        elif 12 <= current_hour < 18:  # Afternoon
            base_personalities = sentiment_personalities.get(sentiment, ['informative'])
        else:  # Evening/Night
            base_personalities = sentiment_personalities.get(sentiment, ['supportive'])
        
        return random.choice(base_personalities)
    
    def generate_ai_response(self, tweet_text: str, sentiment: str, response_type: str = "reply") -> Optional[str]:
        """
        Generate AI response using Gemini.
        
        Args:
            tweet_text: Original tweet text
            sentiment: Sentiment analysis result
            response_type: Type of response (reply, quote, thread)
            
        Returns:
            Generated response text or None if failed
        """
        try:
            personality = self.get_personality_for_sentiment(sentiment)
            
            # Create personality-specific prompts
            personality_prompts = {
                'friendly': "Respond in a warm, friendly, and encouraging tone. Be positive and supportive.",
                'informative': "Provide helpful, factual information. Be educational and precise.",
                'supportive': "Be empathetic and supportive. Offer comfort and understanding.",
                'sarcastic': "Use light sarcasm and wit, but keep it clever and not mean-spirited.",
                'poetic': "Respond with creative, poetic language. Use metaphors and beautiful imagery.",
                'funny': "Be humorous and entertaining. Use appropriate jokes or witty observations."
            }
            
            if response_type == "reply":
                prompt = f"""
                You are a helpful Twitter bot with a {personality} personality. 
                {personality_prompts[personality]}
                
                Original tweet: "{tweet_text}"
                Sentiment: {sentiment}
                
                Generate a short, engaging reply (max 280 characters) that:
                1. Responds naturally to the tweet content
                2. Matches the {personality} personality
                3. Adds value to the conversation
                4. Uses appropriate hashtags if relevant
                
                Reply:
                """
            
            elif response_type == "quote":
                prompt = f"""
                You are a helpful Twitter bot with a {personality} personality.
                {personality_prompts[personality]}
                
                Original tweet: "{tweet_text}"
                Sentiment: {sentiment}
                
                Generate a short quote tweet comment (max 200 characters) that:
                1. Adds your perspective on the original tweet
                2. Matches the {personality} personality
                3. Encourages engagement
                
                Quote tweet:
                """
            
            elif response_type == "thread":
                prompt = f"""
                You are a helpful Twitter bot with a {personality} personality.
                {personality_prompts[personality]}
                
                Generate a 3-part Twitter thread about an interesting topic related to: "{tweet_text}"
                
                Each tweet should be:
                1. Max 280 characters
                2. Engaging and informative
                3. Connected to the next tweet
                4. Include relevant hashtags
                
                Format as:
                TWEET 1: [content]
                TWEET 2: [content]  
                TWEET 3: [content]
                """
            
            # Generate response with retry mechanism
            for attempt in range(3):
                try:
                    response = self.gemini_model.generate_content(prompt)
                    
                    if response.text:
                        generated_text = response.text.strip()
                        
                        # Clean up the response
                        if response_type == "thread":
                            # Parse thread format
                            thread_tweets = []
                            lines = generated_text.split('\n')
                            current_tweet = ""
                            
                            for line in lines:
                                line = line.strip()
                                if line.startswith('TWEET'):
                                    if current_tweet:
                                        thread_tweets.append(current_tweet.strip())
                                    current_tweet = line.split(':', 1)[1].strip() if ':' in line else line
                                elif line and not line.startswith('TWEET'):
                                    current_tweet += " " + line
                            
                            if current_tweet:
                                thread_tweets.append(current_tweet.strip())
                            
                            return thread_tweets[:3]  # Ensure only 3 tweets
                        
                        else:
                            # For replies and quotes, ensure length limit
                            max_length = 280 if response_type == "reply" else 200
                            if len(generated_text) > max_length:
                                generated_text = generated_text[:max_length-3] + "..."
                            
                            return generated_text
                    
                except Exception as e:
                    logger.warning(f"⚠️  Gemini attempt {attempt + 1} failed: {e}")
                    if attempt < 2:
                        time.sleep(2)
                    continue
            
            logger.error("❌ All Gemini attempts failed")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error generating AI response: {e}")
            return None
    
    def get_trending_hashtags(self) -> List[str]:
        """
        Get trending hashtags (simplified version).
        In a real implementation, you might scrape trending topics.
        
        Returns:
            List of trending hashtags
        """
        # Simplified trending hashtags - in production, you'd scrape these
        trending = [
            "#AI", "#Tech", "#Innovation", "#Future", "#Learning",
            "#Inspiration", "#Motivation", "#Success", "#Growth", "#Ideas"
        ]
        return random.sample(trending, k=min(3, len(trending)))
    
    def post_reply(self, original_tweet_id: str, reply_text: str) -> bool:
        """
        Post a reply to a tweet.
        
        Args:
            original_tweet_id: ID of tweet to reply to
            reply_text: Reply text
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.tweet_count_today >= self.max_daily_tweets:
                logger.warning("⚠️  Daily tweet limit reached")
                return False
            
            response = self.twitter_client.create_tweet(
                text=reply_text,
                in_reply_to_tweet_id=original_tweet_id
            )
            
            if response.data:
                self.tweet_count_today += 1
                logger.info(f"✅ Posted reply: {reply_text[:50]}...")
                return True
            else:
                logger.error("❌ Failed to post reply - no response data")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error posting reply: {e}")
            return False
    
    def post_quote_tweet(self, original_tweet_url: str, quote_text: str) -> bool:
        """
        Post a quote tweet.
        
        Args:
            original_tweet_url: URL of original tweet
            quote_text: Quote tweet text
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.tweet_count_today >= self.max_daily_tweets:
                logger.warning("⚠️  Daily tweet limit reached")
                return False
            
            full_text = f"{quote_text} {original_tweet_url}"
            
            response = self.twitter_client.create_tweet(text=full_text)
            
            if response.data:
                self.tweet_count_today += 1
                logger.info(f"✅ Posted quote tweet: {quote_text[:50]}...")
                return True
            else:
                logger.error("❌ Failed to post quote tweet - no response data")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error posting quote tweet: {e}")
            return False
    
    def post_thread(self, thread_tweets: List[str]) -> bool:
        """
        Post a Twitter thread.
        
        Args:
            thread_tweets: List of tweet texts for the thread
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.tweet_count_today + len(thread_tweets) > self.max_daily_tweets:
                logger.warning("⚠️  Not enough daily tweets remaining for thread")
                return False
            
            previous_tweet_id = None
            
            for i, tweet_text in enumerate(thread_tweets):
                # Add thread numbering
                numbered_text = f"{i+1}/{len(thread_tweets)} {tweet_text}"
                
                if previous_tweet_id:
                    response = self.twitter_client.create_tweet(
                        text=numbered_text,
                        in_reply_to_tweet_id=previous_tweet_id
                    )
                else:
                    response = self.twitter_client.create_tweet(text=numbered_text)
                
                if response.data:
                    previous_tweet_id = response.data['id']
                    self.tweet_count_today += 1
                    logger.info(f"✅ Posted thread tweet {i+1}: {tweet_text[:50]}...")
                    time.sleep(1)  # Small delay between thread tweets
                else:
                    logger.error(f"❌ Failed to post thread tweet {i+1}")
                    return False
            
            logger.info(f"✅ Successfully posted {len(thread_tweets)}-tweet thread")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error posting thread: {e}")
            return False
    
    def should_create_thread_today(self) -> bool:
        """
        Determine if we should create a daily thread.
        
        Returns:
            True if thread should be created
        """
        # Check if we've already posted a thread today
        thread_file = f"thread_posted_{datetime.now().date()}.flag"
        
        if os.path.exists(thread_file):
            return False
        
        # Create flag file
        with open(thread_file, 'w') as f:
            f.write(str(datetime.now()))
        
        return True
    
    def cleanup_old_files(self):
        """Clean up old media files and flag files."""
        try:
            # Clean up media files older than 7 days
            if os.path.exists('media'):
                for filename in os.listdir('media'):
                    filepath = os.path.join('media', filename)
                    if os.path.isfile(filepath):
                        file_age = time.time() - os.path.getmtime(filepath)
                        if file_age > 7 * 24 * 3600:  # 7 days
                            os.remove(filepath)
                            logger.info(f"🗑️  Cleaned up old media file: {filename}")
            
            # Clean up old flag files
            for filename in os.listdir('.'):
                if filename.startswith('thread_posted_') and filename.endswith('.flag'):
                    filepath = filename
                    file_age = time.time() - os.path.getmtime(filepath)
                    if file_age > 2 * 24 * 3600:  # 2 days
                        os.remove(filepath)
                        logger.info(f"🗑️  Cleaned up old flag file: {filename}")
                        
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
    
    def run(self):
        """Main bot execution method."""
        try:
            logger.info("🤖 Starting Twitter Bot")
            
            # Clean up old files
            self.cleanup_old_files()
            
            # Define search queries/topics
            search_queries = [
                "artificial intelligence",
                "machine learning", 
                "technology trends",
                "future tech",
                "innovation"
            ]
            
            # Scrape and process tweets
            all_tweets = []
            for query in search_queries:
                tweets = self.scrape_tweets(query, max_tweets=5)
                all_tweets.extend(tweets)
            
            logger.info(f"📊 Processing {len(all_tweets)} tweets total")
            
            # Process each tweet
            for tweet in all_tweets:
                if self.tweet_count_today >= self.max_daily_tweets:
                    logger.warning("⚠️  Daily tweet limit reached, stopping")
                    break
                
                try:
                    # Analyze sentiment
                    sentiment, polarity = self.analyze_sentiment(tweet['text'])
                    logger.info(f"📝 Tweet sentiment: {sentiment} ({polarity:.2f})")
                    
                    # Download media if present
                    local_media_files = []
                    for media in tweet['media']:
                        if media_file := self.download_media(media['url'], str(tweet['id'])):
                            local_media_files.append(media_file)
                    
                    # Decide action (reply, quote, or skip)
                    action = random.choice(['reply', 'quote', 'skip', 'skip'])  # 25% reply, 25% quote, 50% skip
                    
                    if action == 'reply':
                        # Generate and post reply
                        if reply_text := self.generate_ai_response(tweet['text'], sentiment, 'reply'):
                            # Add trending hashtags occasionally
                            if random.random() < 0.3:  # 30% chance
                                hashtags = self.get_trending_hashtags()
                                hashtag_text = ' '.join(hashtags[:2])  # Max 2 hashtags
                                if len(reply_text) + len(hashtag_text) + 1 <= 280:
                                    reply_text += f" {hashtag_text}"
                            
                            self.post_reply(str(tweet['id']), reply_text)
                    
                    elif action == 'quote':
                        # Generate and post quote tweet
                        if quote_text := self.generate_ai_response(tweet['text'], sentiment, 'quote'):
                            self.post_quote_tweet(tweet['url'], quote_text)
                    
                    # Small delay between actions
                    time.sleep(random.uniform(5, 15))
                    
                except Exception as e:
                    logger.error(f"❌ Error processing tweet {tweet['id']}: {e}")
                    continue
            
            # Create daily thread if needed
            if self.should_create_thread_today() and self.tweet_count_today < self.max_daily_tweets - 3:
                logger.info("📝 Creating daily thread")
                
                # Generate thread about a random topic
                thread_topics = [
                    "the future of artificial intelligence",
                    "emerging technology trends", 
                    "innovation in the digital age",
                    "the impact of AI on society",
                    "sustainable technology solutions"
                ]
                
                topic = random.choice(thread_topics)
                if thread_content := self.generate_ai_response(topic, 'neutral', 'thread'):
                    self.post_thread(thread_content)
            
            logger.info(f"✅ Bot run completed. Posted {self.tweet_count_today} tweets today.")
            
        except Exception as e:
            logger.error(f"❌ Error in bot execution: {e}")
            raise

if __name__ == "__main__":
    try:
        bot = TwitterBot()
        bot.run()
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"💥 Bot crashed: {e}")
        raise
