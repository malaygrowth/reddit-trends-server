\# reddit-trends-server

Simple Flask server to fetch top/hot Reddit posts and top comments, returning JSON for a Custom GPT.



Endpoints:

GET /api/search?q=KEYWORD\&subs=startups,sales\&sort=hot\&limit=6

Header required: x-api-key: <my\_super\_secret\_123>



Environment variables:

\- REDDIT\_CLIENT\_ID

\- REDDIT\_CLIENT\_SECRET

\- REDDIT\_USER\_AGENT

\- API\_KEY



