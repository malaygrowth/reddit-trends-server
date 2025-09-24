# app.py  — Reddit fetch + comment analysis for Custom GPT
import os, re, json
from flask import Flask, request, jsonify, abort
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from collections import Counter

# --- Config (from Railway environment variables) ---
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET")
REDDIT_USERNAME = os.environ.get("REDDIT_USERNAME") or None
REDDIT_PASSWORD = os.environ.get("REDDIT_PASSWORD") or None
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "reddit-trends-bot/0.1 by example")
API_KEY = os.environ.get("API_KEY")  # secret that Custom GPT will send
PORT = int(os.environ.get("PORT", 5000))

# --- Basic checks ---
if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
    raise RuntimeError("Missing Reddit API credentials in environment variables.")

# --- Initialize PRAW (Reddit client) ---
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    username=REDDIT_USERNAME,
    password=REDDIT_PASSWORD,
    user_agent=REDDIT_USER_AGENT
)

# --- spaCy model: try to load, otherwise download at first run ---
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

sia = SentimentIntensityAnalyzer()
app = Flask(__name__)

# --- helpers ---
def clean_text(s):
    return re.sub(r'\s+', ' ', s).strip()

def analyze_comments(comments):
    results = []
    for c in comments:
        if not getattr(c, "body", None):
            continue
        text = clean_text(c.body)
        sent = sia.polarity_scores(text)
        doc = nlp(text)
        ents = [ent.text for ent in doc.ents]
        tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
        results.append({
            "text": text,
            "upvotes": getattr(c, "score", 0),
            "sentiment": sent,
            "entities": ents,
            "tokens": tokens
        })
    all_tokens = [t for r in results for t in r["tokens"]]
    common_tokens = Counter(all_tokens).most_common(25)
    avg_compound = sum(r["sentiment"]["compound"] for r in results) / (len(results) or 1)
    return {"comments": results, "common_tokens": common_tokens, "avg_compound": avg_compound}

def fetch_posts(subreddits, q, sort="hot", limit=10):
    subs = "+".join(subreddits) if subreddits else "all"
    subreddit = reddit.subreddit(subs)
    # choose iterator
    if q:
        submissions = subreddit.search(q, sort="relevance", limit=limit)
    else:
        if sort == "hot":
            submissions = subreddit.hot(limit=limit)
        elif sort == "top":
            submissions = subreddit.top(time_filter="day", limit=limit)
        else:
            submissions = subreddit.new(limit=limit)
    out = []
    for s in submissions:
        try:
            s.comments.replace_more(limit=0)
        except Exception:
            pass
        top_comments = sorted(s.comments, key=lambda c: getattr(c, "score", 0), reverse=True)[:5]
        analysis = analyze_comments(top_comments)
        out.append({
            "id": s.id,
            "url": "https://reddit.com" + s.permalink,
            "title": s.title,
            "subreddit": str(s.subreddit),
            "score": s.score,
            "num_comments": s.num_comments,
            "top_comments": [{"text": clean_text(c.body), "upvotes": c.score} for c in top_comments],
            "comment_analysis": analysis
        })
    return out

# --- endpoints ---
@app.route("/")
def index():
    return jsonify({"ok": True, "msg": "reddit-trends-server is running"})

@app.route("/api/search")
def api_search():
    # security: require x-api-key header
    header_key = request.headers.get("x-api-key", "")
    if not API_KEY or header_key != API_KEY:
        return abort(401, description="Unauthorized: missing or invalid x-api-key header")

    q = request.args.get("q", "")
    subs = request.args.get("subs", "")   # comma separated
    sort = request.args.get("sort", "hot")
    limit = int(request.args.get("limit", 10))
    sub_list = [s.strip() for s in subs.split(",") if s.strip()] if subs else []
    posts = fetch_posts(sub_list, q, sort=sort, limit=limit)
    return jsonify({"query": q, "count": len(posts), "posts": posts})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
