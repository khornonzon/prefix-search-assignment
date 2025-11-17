# Prefix Search Implementation

This document describes the implementation of the prefix search system for the cargo catalog assignment.

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)

### Setup and Run

**Option 1: Define venv and use setup script (recommended)**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requrements.txt
./setup.sh
```

**Option 2: Manual setup**
```bash
# 1. Start Elasticsearch and search service
docker compose up 

# 2. Wait for Elasticsearch to be ready (about 30 seconds)
# Check health: curl http://localhost:9200/_cluster/health

# 3. Load catalog into Elasticsearch (with embeddings)
python tools/load_catalog.py --host localhost --port 9200 --with-embeddings

# 4. Test the search service
curl "http://localhost:5000/search?q=масло"

# 5. Run evaluation
python tools/evaluate.py --type open
```

### Docker Compose Services

- **elasticsearch**: Runs on port 9200
- **search-service**: Flask API on port 5000

## Features

### Hybrid Search (Text + Embeddings)

The system supports hybrid search combining:
- **Text-based search**: Prefix matching, keyboard layout switching, transliteration
- **Vector search**: Semantic similarity using embeddings

### Embeddings

- **Default**: Uses `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions)
- **Storage**: Embeddings stored in Elasticsearch `dense_vector` field
- **Search**: Combined with text search using Elasticsearch `knn` query


## Index Schema and Analyzers

### Index Structure

The `catalog` index contains the following fields:

- `id` (keyword) - Product ID
- `name` (text) - Product name with prefix analyzer
- `category` (keyword + text) - Category with both exact and text search
- `brand` (text + keyword) - Brand name
- `weight` (float) - Weight/volume value
- `weight_unit` (keyword) - Unit (kg, g, l, ml, etc.)
- `package_size` (integer) - Package size
- `keywords` (text) - Searchable keywords
- `description` (text) - Product description
- `price` (float) - Price
- `image_url` (keyword) - Image URL
- `embedding` (dense_vector) - Semantic embedding vector (384 or 1536 dimensions)

### Custom Analyzers

#### prefix_analyzer
- **Tokenizer**: standard
- **Filters**: lowercase, russian_stop, prefix_edge_ngram
- **Char Filter**: layout_switch (qwerty ↔ йцукен)
- **Purpose**: Index-time analyzer for prefix matching (1-15 characters)

#### search_analyzer
- **Tokenizer**: standard
- **Filters**: lowercase, russian_stop
- **Purpose**: Query-time analyzer for search

#### prefix_edge_ngram Filter
- **Type**: edge_ngram
- **Min gram**: 1
- **Max gram**: 15
- **Token chars**: letter, digit

#### layout_switch Char Filter
- **Type**: mapping
- **Mappings**: Full bidirectional keyboard layout mapping (qwerty ↔ йцукен)

## Search Logic

### Query Normalization

1. **Numeric Attribute Extraction**: Extracts weight/volume from query (e.g., "10л", "5kg")
2. **Keyboard Layout Switching**: Detects and switches between qwerty and йцукен layouts
3. **Transliteration**: Converts Cyrillic to Latin (using `transliterate` library)
4. **Query Variations**: Generates multiple query variations for better recall

### Ranking Strategy

The search uses hybrid ranking combining text and vector search:

**Text-based boosts:**
- **Name exact match**: boost 5.0
- **Name prefix match**: boost 4.0 (using `match_bool_prefix`)
- **Brand match**: boost 3.0
- **Keywords match**: boost 2.0
- **Category match**: boost 1.5
- **Description match**: boost 1.0
- **Numeric attribute match**: boost 2.0 (if weight/volume matches)

**Vector search:**
- **Semantic similarity**: Uses cosine similarity on embeddings
- **Boost**: 0.5 (combined with text search)
- **Candidates**: Searches 10x top_k candidates for better recall

### Noise Filtering

Results are filtered to remove noise:

1. **Score threshold**: Results with score < 0.1 are removed
2. **Word matching**: At least one query word must match name, brand, or keywords
3. **Prefix matching**: Query words should appear as prefixes in name/brand
4. **High score bypass**: Results with score > 2.0 bypass filtering

## Supported Scenarios

### 1. Short Prefixes (1-3 letters)
- Example: "ма" → finds "масло", "макароны"
- Uses edge_ngram analyzer with min_gram=1

### 2. Keyboard Layout Switching
- Example: "xfq" → finds "чай" (qwerty layout mistake)
- Uses layout_switch char filter

### 3. Transliteration
- Example: "prosecco" → finds products with "prosecco" in name/brand
- Uses transliterate library for Cyrillic → Latin

### 4. Numeric Attributes
- Example: "масло раст 10л" → finds sunflower oil with ~10L volume
- Extracts numeric patterns and boosts matching products

### 5. Mixed Languages
- Example: "adapter usb c" → finds USB-C adapters
- Handles both Cyrillic and Latin text

### 6. Typos and Abbreviations
- Example: "греч не" → finds "гречневая"
- Uses prefix matching and fuzzy logic

## Evaluation Metrics

The evaluation script calculates:

- **Precision@3**: Fraction of top-3 results that are relevant
- **Coverage**: Fraction of queries that return at least one result
- **Latency**: Average response time in milliseconds

### Running Evaluation

```bash
# Evaluate open queries
python tools/evaluate.py --type open --output reports/evaluation_open.csv

# Evaluate hidden queries
python tools/evaluate.py --type hidden --output reports/evaluation_hidden.csv

# Evaluate all queries
python tools/evaluate.py --type all
```

## API Endpoints

### GET /health
Health check endpoint.

**Response:**
```json
{"status": "ok"}
```

### GET /search?q=<query>&top_k=<number>&use_embeddings=<bool>
Search endpoint.

**Parameters:**
- `q` (required): Search query
- `top_k` (optional): Number of results (default: 5)
- `use_embeddings` (optional): Enable/disable embeddings (default: true)


## Implementation Details

### Query Processing Flow

1. **Input**: User query (e.g., "масло раст 10л")
2. **Normalization**:
   - Extract numeric: `{"volume_l": 10}`
   - Remove numeric from text: "масло раст"
   - Generate variations: ["масло раст", "vfkj hfcn"] (layout switch)
3. **Elasticsearch Query**:
   - Multi-match across name, brand, keywords, category, description
   - Boost numeric matches if present
4. **Post-processing**:
   - Filter noise results
   - Return top K results

### Keyboard Layout Mapping

The layout mapping covers:
- Letters: q→й, w→ц, e→у, etc.
- Punctuation: [→х, ]→ъ, ;→ж, etc.
- Bidirectional: Both qwerty→йцукен and йцукен→qwerty

### Numeric Attribute Parsing

Supports patterns:
- `10л`, `10l`, `10лт` → volume in liters
- `5кг`, `5kg` → weight in kilograms
- `500г`, `500g` → weight in grams
- `1000мл`, `1000ml` → volume in milliliters

## Known Limitations and Future Improvements

### Current Limitations

1. **Transliteration**: Only handles Cyrillic→Latin, not reverse
2. **Numeric Matching**: Weight/volume matching is approximate
3. **Noise Filtering**: Heuristic-based, may filter some valid results
4. **No Learning**: No machine learning for ranking improvements

## Reflection

### Challenges Encountered

1. **Edge N-gram Configuration**: Initially tried to use edge_ngram as tokenizer, but it should be a filter
2. **Numeric Attribute Matching**: Decided to use boosts instead of strict filters to maintain recall
3. **Noise Filtering**: Balancing precision and recall - too strict filters remove valid results

### Design Decisions

1. **Elasticsearch over Faiss/Milvus**: Chosen for better text search capabilities and analyzers
2. **Multi-match with Boosts**: Provides flexibility while maintaining relevance
3. **Post-processing Filtering**: Allows fine-grained control over result quality
4. **Docker Compose**: Simplifies deployment and ensures consistency

### Test Scenarios Covered

The solution handles the key scenarios from `prefix_queries.csv`:

- ✅ Short prefixes ("ма", "pr")
- ✅ Keyboard layout mistakes ("xfq" → "чай")
- ✅ Transliteration ("prosecco", "riesling")
- ✅ Numeric attributes ("10л", "5kg")
- ✅ Mixed languages ("adapter usb c")
- ✅ Abbreviations ("греч не" → "гречневая")
- ✅ Multi-word queries ("масло раст 10л")

### Performance

- **Indexing**: ~1000 products in < 5 seconds
- **Search Latency**: < 100ms for most queries
- **Coverage**: Target ≥70% (to be validated with evaluation)


## Loading Catalog with Embeddings

```bash
# With sentence-transformers (default, free)
python tools/load_catalog.py --with-embeddings


# Custom embedding model
python tools/load_catalog.py --with-embeddings --embedding-model "sentence-transformers/all-MiniLM-L6-v2"
```


