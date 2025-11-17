from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from search import SearchEngine

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.json.ensure_ascii = False
CORS(app)


def _fix_query_encoding(value: str) -> str:
    if not value:
        return value
    try:
        if any("Ã" <= ch <= "Ý" or "Ð" <= ch <= "ß" for ch in value):
            return value.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        pass
    return value

es_host = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
use_embeddings = os.getenv("USE_EMBEDDINGS", "true").lower() == "true"

try:
    search_engine = SearchEngine(es_host, use_embeddings=use_embeddings)
except Exception as e:
    print(f"Warning: Could not initialize search engine: {e}")
    search_engine = None


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


@app.route("/search", methods=["GET", "POST"])
def search():
    """Search endpoint."""
    if search_engine is None:
        return jsonify({"error": "Search engine not initialized"}), 503
    
    if request.method == "POST":
        data = request.get_json()
        query = data.get("query", "")
        top_k = data.get("top_k", 5)
        use_embeddings = data.get("use_embeddings")
    else:
        query = request.args.get("q", "")
        top_k = int(request.args.get("top_k", 5))
        use_embeddings = request.args.get("use_embeddings")
        use_embeddings = use_embeddings.lower() == "true" if use_embeddings else None

    # Fix potential mojibake from mis-decoded UTF-8 query parameters
    query = _fix_query_encoding(query)

    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400
    
    try:
        print(query)
        results = search_engine.search(query, top_k=top_k, use_embeddings=use_embeddings)
        return jsonify({
            "query": query,
            "results": results,
            "count": len(results)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

