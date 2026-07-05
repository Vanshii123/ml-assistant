"""
Zero-cost YouTube suggestion helper.
Generates a direct YouTube search deep-link — no API key, no quota, no cost.
"""
import urllib.parse


def get_youtube_search_link(topic: str, subtopic: str = "", college: str = "", lang: str = "hindi") -> str:
    parts = [p.strip() for p in [
        topic, subtopic, college, "explanation", "in detail", lang
    ] if p and p.strip()]
    query = " ".join(parts)
    encoded_query = urllib.parse.quote_plus(query)
    return f"https://www.youtube.com/results?search_query={encoded_query}"