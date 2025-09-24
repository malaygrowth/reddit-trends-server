# app.py
"""
Reddit trends server (flask)
- Fetch posts + top comments via PRAW
- Basic analysis: sentiment (VADER), word frequency, bigrams, entities (spaCy if available)
- Optional OpenAI "writer" summary if OPENAI_API_KEY present

Endpoints:
- GET /health
- GET /api/search?q=<query>&subreddits=r/xyz,r/abc&limit=10  (or POST JSON)
- POST /api/analyze   (body {"text": "..."} ) -> sentiment + entities
"""

import os
import re
import time
import math
import logging
from collections import Counter, defaultdict
from itertools import islice
from typing import List, Dict

from flask import Flask, request, jsonify, abort

# PRAW for Reddit API
import praw

# VADER sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Optional OpenAI usage (simple requests call) - only used if key present
import requests

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ----------------------------
# Config from environment
# ----------------------------
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "reddit-trends-server/0.1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # optional

if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET):
    logging.warning("REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET not set — Reddit calls will fail.")

# Initialize PRAW (read-only)
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID or "",
    client_secret=REDDIT_CLIENT_SECRET or "",
    user_agent=REDDIT_USER_AGENT,
    check_for_async=False,
)


# VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


# ----------------------------
# Lazy spaCy loader (safe)
# ----------------------------
import threading
_spacy_lock = threading.Lock()
_nlp = None
_spacy_available = False

def ensure_spacy_loaded():
    """Attempt to import spaCy and load en_core_web_sm (only if already installed).
       If spaCy or model is missing (or import fails), don't raise — mark unavailable."""
    global _nlp, _spacy_available
    if _nlp is not None or _spacy_available:
        return
    with _spacy_lock:
        if _nlp is not None or _spacy_available:
            return
        try:
            import spacy
            try:
                _nlp = spacy.load("en_core_web_sm")
                _spacy_available = True
                logging.info("spaCy loaded successfully")
            except Exception as e:
                logging.warning("spaCy model not available / failed to load: %s", e)
                _nlp = None
                _spacy_available = False
        except Exception as e:
            logging.warning("spaCy not installed or failed to import: %s", e)
            _nlp = None
            _spacy_available = False

def spacy_entities(text: str):
    ensure_spacy_loaded()
    if not _spacy_available or _nlp is None:
        return []
    doc = _nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


# ----------------------------
# Utilities: text cleaning, ngrams, top comments
# ----------------------------
WORD_RE = re.compile(r"\b[a-zA-Z0-9']+\b")

# Minimal fallback stopwords if spaCy unavailable
FALLBACK_STOPWORDS = {
    "the","and","to","of","a","in","is","it","i","for","on","this","that","with","you","are","as","be",
    "was","but","not","have","they","we","so","if","at","or","from","by","an","my","has","what","about"
}

def tokenize(text: str):
    words = WORD_RE.findall(text.lower())
    return words

def top_words(texts: List[str], top_n=25):
    all_words = []
    for t in texts:
        all_words.extend(tokenize(t))
    # remove stopwords (use spaCy stopwords if available)
    stopwords = FALLBACK_STOPWORDS.copy()
    try:
        ensure_spacy_loaded()
        if _spacy_available and _nlp is not None:
            stopwords = set([w.lower() for w in _nlp.Defaults.stop_words])
    except Exception:
        pass

    filtered = [w for w in all_words if w not in stopwords and len(w) > 2]
    c = Counter(filtered)
    return c.most_common(top_n)

def top_bigrams(texts: List[str], top_n=20):
    from collections import Counter
    tokens_all = []
    for t in texts:
        tokens_all.extend(tokenize(t))
    # basic bigrams
    bigrams = zip(tokens_all, tokens_all[1:])
    # filter bigrams containing stopwords
    stopwords = FALLBACK_STOPWORDS
    bigram_list = [" ".join(b) for b in bigrams if b[0] not in stopwords and b[1] not in stopwords]
    return Counter(bigram_list).most_common(top_n)

def sentiment_summary(texts: List[str]):
    if not texts:
        return {"count": 0, "avg_compound": 0.0, "distribution": {}}
    scores = []
    dist = {"positive": 0, "neutral": 0, "negative": 0}
    for t in texts:
        s = analyzer.polarity_scores(t)
        scores.append(s["compound"])
        if s["compound"] >= 0.05:
            dist["positive"] += 1
        elif s["compound"] <= -0.05:
            dist["negative"] += 1
        else:
            dist["neutral"] += 1
    avg = sum(scores) / len(scores) if scores else 0.0
    return {"count": len(texts), "avg_compound": avg, "distribution": dist}

def get_top_comments(submission, top_n=5):
    """
    Returns top_n comments (by score) for a submission.
    Uses replace_more(limit=0) to load current comments.
    """
    try:
        submission.comments.replace_more(limit=0)
    except Exception:
        # network or replaced_more might fail; continue with what's available
        pass
    comments = []
    for c in submission.comments:
        try:
            if isinstance(c.score, int):
                comments.append({"body": c.body, "score": c.score, "id": getattr(c, 'id', None)})
        except Exception:
            continue
    # sort by score desc
    comments_sorted = sorted(comments, key=lambda x: x.get("score", 0), reverse=True)
    return comments_sorted[:top_n]


# ----------------------------
# Reddit fetching logic
# ----------------------------
def fetch_from_subreddit(subreddit_name: str, limit=10):
    """
    Fetches 'hot' and 'top (day)' posts from a subreddit.
    Returns combined list (deduped).
    """
    posts = []
    try:
        sub = reddit.subreddit(subreddit_name)
    except Exception as e:
        logging.warning("Failed to get subreddit %s: %s", subreddit_name, e)
        return posts

    seen_ids = set()

    # hot posts
    try:
        for s in islice(sub.hot(limit=limit), limit):
            if s.id in seen_ids:
                continue
            seen_ids.add(s.id)
            posts.append(s)
    except Exception as e:
        logging.debug("sub.hot failed for %s: %s", subreddit_name, e)

    # top of day
    try:
        for s in islice(sub.top(time_filter="day", limit=limit), limit):
            if s.id in seen_ids:
                continue
            seen_ids.add(s.id)
            posts.append(s)
    except Exception as e:
        logging.debug("sub.top failed for %s: %s", subreddit_name, e)

    return posts

def search_reddit(query: str, subreddits: List[str]=None, limit=25):
    """
    Search across subreddits or reddit-wide if subreddits is None.
    Returns list of praw.Submission
    """
    results = []
    try:
        if subreddits:
            # search each subreddit
            for sr in subreddits:
                try:
                    sub = reddit.subreddit(sr)
                    for s in sub.search(query, limit=limit, sort="relevance"):
                        results.append(s)
                except Exception as e:
                    logging.debug("search failed in %s: %s", sr, e)
        else:
            for s in reddit.subreddit("all").search(query, limit=limit, sort="relevance"):
                results.append(s)
    except Exception as e:
        logging.warning("Reddit search failed: %s", e)
    return results

def serialize_submission(s):
    """Return simple JSON-friendly dict for a submission"""
    return {
        "id": getattr(s, "id", None),
        "title": getattr(s, "title", None),
        "url": getattr(s, "url", None),
        "permalink": ("https://reddit.com" + s.permalink) if getattr(s, "permalink", None) else None,
        "subreddit": getattr(s, "subreddit", {}).display_name if getattr(s, "subreddit", None) else None,
        "score": getattr(s, "score", None),
        "num_comments": getattr(s, "num_comments", None),
        "created_utc": getattr(s, "created_utc", None),
    }

# ----------------------------
# Analysis pipeline
# ----------------------------
def analyze_posts(submissions: List):
    """
    For given list of praw.Submission objects:
    - extracts top comments
    - analyzes comment sentiment and top words/phrases
    - extracts entities if spaCy available
    - returns analysis summary + per-post info
    """
    post_results = []
    all_comments_texts = []
    for s in submissions:
        try:
            top_comments = get_top_comments(s, top_n=5)
        except Exception:
            top_comments = []
        # add to global pool for trends
        for c in top_comments:
            all_comments_texts.append(c["body"])

        post_results.append({
            "post": serialize_submission(s),
            "top_comments": top_comments
        })

    # analysis across all top comments
    sentiment = sentiment_summary(all_comments_texts)
    top_words_list = top_words(all_comments_texts, top_n=25)
    top_bigrams_list = top_bigrams(all_comments_texts, top_n=20)

    # entities via spaCy if available
    entities_counter = Counter()
    entities_sample = []
    try:
        ensure_spacy_loaded()
        if _spacy_available and _nlp is not None:
            # aggregate named entities
            for t in all_comments_texts:
                doc = _nlp(t)
                for ent in doc.ents:
                    entities_counter[(ent.text.strip(), ent.label_)] += 1
                    if len(entities_sample) < 30:
                        entities_sample.append((ent.text, ent.label_))
    except Exception as e:
        logging.debug("spaCy entity extraction failed: %s", e)

    top_entities = [{"text": text, "label": label, "count": count} for (text, label), count in entities_counter.most_common(30)]

    analysis = {
        "sentiment": sentiment,
        "top_words": [{"word": w, "count": c} for w, c in top_words_list],
        "top_bigrams": [{"bigram": b, "count": c} for b, c in top_bigrams_list],
        "top_entities": top_entities,
        "sample_entities": entities_sample,
        "total_comments_analyzed": len(all_comments_texts),
    }

    # small "insights" heuristics
    insights = []
    if sentiment["count"] > 0:
        if sentiment["avg_compound"] > 0.2:
            insights.append("Overall tone across top comments is positive.")
        elif sentiment["avg_compound"] < -0.2:
            insights.append("Overall tone across top comments is negative.")
        else:
            insights.append("Overall tone is mixed/neutral across top comments.")
    if top_words_list:
        insights.append(f"Top terms: {', '.join([w for w, _ in top_words_list[:6]])}")
    if top_bigrams_list:
        insights.append(f"Common phrases: {', '.join([b for b, _ in top_bigrams_list[:6]])}")

    return {"posts": post_results, "analysis": analysis, "insights": insights}


# ----------------------------
# Optional OpenAI writer summary (very simple integration)
# ----------------------------
def generate_writer_summary(insights_summary_text: str, max_tokens=200):
    """
    If OPENAI_API_KEY is set, calls OpenAI (Chat Completions API) to create a short human-friendly summary.
    This function uses the HTTP API minimally — adapt if you use official SDK.
    """
    if not OPENAI_API_KEY:
        return None

    # example prompt - keep short
    prompt = (
        "You are a concise analyst. Summarize the following reddit insights in 2-3 sentences, "
        "state what's gaining traction, why it resonates, and how people are engaging:\n\n"
        f"{insights_summary_text}"
    )

    # NOTE: adjust endpoint to the model / API you want. This example uses the OpenAI completions endpoint.
    # Replace with official newer endpoints / libraries if you prefer.
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",  # change model to available one for you
                "messages": [{"role": "system", "content": "You are a concise analyst."},
                             {"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.3,
            },
            timeout=15,
        )
        if resp.status_code == 200:
            j = resp.json()
            # chat completion response shape: choices[0].message.content
            return j["choices"][0]["message"]["content"].strip()
        else:
            logging.warning("OpenAI call failed %s: %s", resp.status_code, resp.text)
            return None
    except Exception as e:
        logging.warning("OpenAI call exception: %s", e)
        return None


# ----------------------------
# Flask endpoints
# ----------------------------
@app.route("/health", methods=["GET"])
def health():
    ensure_spacy_loaded()
    return jsonify({"ok": True, "spacy": bool(_spacy_available)})


@app.route("/api/analyze", methods=["POST"])
def analyze_text():
    """Analyze a single text for sentiment + entities."""
    payload = request.get_json(force=True, silent=True)
    if not payload or "text" not in payload:
        return jsonify({"error": "provide JSON body with 'text' field"}), 400
    text = payload["text"]
    sent = analyzer.polarity_scores(text)
    ents = spacy_entities(text)
    return jsonify({"sentiment": sent, "entities": ents})


@app.route("/api/search", methods=["GET", "POST"])
def api_search():
    """
    Query params (GET):
      - q (required)
      - subreddits (optional): comma separated e.g. "python,MachineLearning"
      - limit (optional): posts per subreddit or search limit
    POST: JSON body with same keys allowed.
    """
    if request.method == "GET":
        q = request.args.get("q", "").strip()
        subreddits = request.args.get("subreddits", "")
        limit = int(request.args.get("limit", 15))
    else:
        data = request.get_json(force=True, silent=True) or {}
        q = (data.get("q") or "").strip()
        subreddits = data.get("subreddits", "")
        limit = int(data.get("limit", 15))

    if not q:
        return jsonify({"error": "query parameter 'q' is required"}), 400

    subreddit_list = [s.strip() for s in subreddits.split(",") if s.strip()] if subreddits else []

    # If user provided subreddit_list, fetch hot/top from those subreddits; otherwise search across reddit
    submissions = []
    try:
        if subreddit_list:
            for sr in subreddit_list:
                subs = fetch_from_subreddit(sr, limit=limit)
                submissions.extend(subs)
        else:
            submissions = search_reddit(q, subreddits=None, limit=limit)
    except Exception as e:
        logging.warning("Error fetching reddit submissions: %s", e)

    # If search returned 0 items (rare), try a fallback: search reddit-wide
    if not submissions:
        submissions = search_reddit(q, subreddits=None, limit=limit)

    # analyze posts and comments
    analysis_result = analyze_posts(submissions)

    # assemble text for writer summary
    insights_text = "\n".join(analysis_result.get("insights", []))
    writer_summary = None
    if OPENAI_API_KEY and insights_text:
        writer_summary = generate_writer_summary(insights_text)

    response = {
        "query": q,
        "subreddits": subreddit_list,
        "count_posts": len(analysis_result["posts"]),
        "results": analysis_result,
        "writer_summary": writer_summary,
    }
    return jsonify(response)


# ----------------------------
# Run locally (flask dev) - Railway/Gunicorn uses Procfile
# ----------------------------
if __name__ == "__main__":
    # Local dev
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
