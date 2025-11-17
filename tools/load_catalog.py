import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import time
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from service.embeddings import EmbeddingService


def create_index(es: Elasticsearch, index_name: str = "catalog", embedding_dim: int = 384) -> None:

    layout_map = {
        'q': 'й', 'w': 'ц', 'e': 'у', 'r': 'к', 't': 'е', 'y': 'н', 'u': 'г', 'i': 'ш', 'o': 'щ', 'p': 'з',
        'a': 'ф', 's': 'ы', 'd': 'в', 'f': 'а', 'g': 'п', 'h': 'р', 'j': 'о', 'k': 'л', 'l': 'д',
        'z': 'я', 'x': 'ч', 'c': 'с', 'v': 'м', 'b': 'и', 'n': 'т', 'm': 'ь'
    }
    
    reverse_layout_map = {v: k for k, v in layout_map.items()}
    
    mapping = {
        "settings": {
            "analysis": {
                "filter": {
                    "russian_stop": {
                        "type": "stop",
                        "stopwords": "_russian_"
                    },
                    "prefix_edge_ngram": {
                        "type": "edge_ngram",
                        "min_gram": 2,
                        "max_gram": 15,
                        "token_chars": ["letter", "digit"]
                    }
                },
                "analyzer": {
                    "prefix_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "russian_stop", "prefix_edge_ngram"]
                    },
                    "search_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "russian_stop"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "name": {
                    "type": "text",
                    "analyzer": "prefix_analyzer",
                    "search_analyzer": "search_analyzer",
                    "fields": {
                        "keyword": {"type": "keyword"}
                    }
                },
                "brand": {
                    "type": "text",
                    "analyzer": "prefix_analyzer",
                    "search_analyzer": "search_analyzer",
                    "fields": {
                        "keyword": {"type": "keyword"}
                    }
                },
                "category": {
                    "type": "text",
                    "analyzer": "prefix_analyzer",
                    "search_analyzer": "search_analyzer",
                    "fields": {
                        "keyword": {"type": "keyword"}
                    }
                },
                "keywords": {
                    "type": "text",
                    "analyzer": "prefix_analyzer",
                    "search_analyzer": "search_analyzer"
                },
                "description": {
                    "type": "text",
                    "analyzer": "prefix_analyzer",
                    "search_analyzer": "search_analyzer"
                },
                "weight": {"type": "float"},
                "weight_unit": {"type": "keyword"},
                "package_size": {"type": "integer"},
                "price": {"type": "float"},
                "image_url": {"type": "keyword"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": embedding_dim,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }

    if es.indices.exists(index=index_name):
        print(f"Deleting existing index: {index_name}")
        es.indices.delete(index=index_name)
    
    es.indices.create(index=index_name, settings=mapping["settings"], mappings=mapping["mappings"])
    print(f"Created index: {index_name}")


def parse_weight(weight_str: str, unit: str) -> tuple[float, str]:
    try:
        weight = float(weight_str)
    except (ValueError, TypeError):
        weight = 0.0
    
    unit_map = {
        "g": "g", "kg": "kg", "ml": "ml", "l": "л", "л": "л",
        "pcs": "pcs", "packs": "packs", "sachets": "sachets",
        "caps": "caps", "tabs": "tabs", "bag": "bag"
    }
    normalized_unit = unit_map.get(unit.lower(), unit)
    
    return weight, normalized_unit


def load_catalog(xml_path: Path, es: Elasticsearch, index_name: str = "catalog", 
                 embedding_service: EmbeddingService | None = None) -> None:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    products = root.findall("product")
    
    if embedding_service:
        print("Generating embeddings for products...")
        product_texts = []
        for product in products:
            name = product.findtext("name", default="")
            description = product.findtext("description", default="")
            keywords = product.findtext("keywords", default="")
            product_texts.append((name, description, keywords))
        
        batch_size = 50
        embeddings_list = []
        for i in range(0, len(product_texts), batch_size):
            batch = product_texts[i:i+batch_size]
            batch_texts = [f"{name} {description} {keywords}" for name, description, keywords in batch]
            batch_embeddings = embedding_service.encode(batch_texts)
            embeddings_list.extend(batch_embeddings)
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Generated embeddings for {min(i+batch_size, len(product_texts))}/{len(product_texts)} products")
        
        print(f"Generated {len(embeddings_list)} embeddings")
    else:
        embeddings_list = [None] * len(products)
    
    def generate_docs():
        for idx, product in enumerate(products):
            product_id = product.get("id", "")
            name = product.findtext("name", default="")
            category = product.findtext("category", default="")
            brand = product.findtext("brand", default="")
            
            weight_elem = product.find("weight")
            weight = 0.0
            weight_unit = ""
            if weight_elem is not None:
                weight, weight_unit = parse_weight(weight_elem.text or "0", weight_elem.get("unit", ""))
            
            package_size = int(product.findtext("package_size", default="1") or "1")
            keywords = product.findtext("keywords", default="")
            description = product.findtext("description", default="")
            price = float(product.findtext("price", default="0") or "0")
            image_url = product.findtext("image_url", default="")
            
            doc_source = {
                "id": product_id,
                "name": name,
                "category": category,
                "brand": brand,
                "weight": weight,
                "weight_unit": weight_unit,
                "package_size": package_size,
                "keywords": keywords,
                "description": description,
                "price": price,
                "image_url": image_url
            }
            
            if embeddings_list[idx] is not None:
                doc_source["embedding"] = embeddings_list[idx].tolist()
            
            doc = {
                "_index": index_name,
                "_id": product_id,
                "_source": doc_source
            }
            yield doc
    
    print(f"Loading {len(products)} products into Elasticsearch...")
    success, failed = bulk(es, generate_docs(), chunk_size=100, request_timeout=120)
    print(f"Successfully indexed {success} documents")
    if failed:
        print(f"Failed to index {len(failed)} documents")

    es.indices.refresh(index=index_name)
    print("Index refreshed")


def wait_for_elasticsearch(es: Elasticsearch, max_retries: int = 30) -> bool:
    for i in range(max_retries):
        try:
            if es.ping():
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Load catalog into Elasticsearch")
    parser.add_argument("--catalog", default="data/catalog_products.xml", help="Path to XML catalog")
    parser.add_argument("--host", default="localhost", help="Elasticsearch host")
    parser.add_argument("--port", type=int, default=9200, help="Elasticsearch port")
    parser.add_argument("--index", default="catalog", help="Index name")
    parser.add_argument(
        "--with-embeddings",
        dest="with_embeddings",
        action="store_true",
        help="Generate and store embeddings (default)",
    )
    parser.add_argument(
        "--no-embeddings",
        dest="with_embeddings",
        action="store_false",
        help="Skip embedding generation",
    )
    parser.set_defaults(with_embeddings=True)
    parser.add_argument("--embedding-model", default="paraphrase-multilingual-MiniLM-L12-v2", help="Embedding model name")
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI embeddings (requires API key)")
    args = parser.parse_args()
    
    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        raise SystemExit(f"Catalog not found: {catalog_path}")
    
    es = Elasticsearch([f"http://{args.host}:{args.port}"])
    
    print("Waiting for Elasticsearch...")
    if not wait_for_elasticsearch(es):
        raise SystemExit("Elasticsearch is not available")
    
    embedding_service = None
    embedding_dim = 384
    
    if args.with_embeddings:
        embedding_service = EmbeddingService(
                model_name=args.embedding_model
            )
        embedding_dim = embedding_service.EMBEDDING_DIM
        print(f"Using embedding model with dimension: {embedding_dim}")
    
    print("Creating index...")
    create_index(es, args.index, embedding_dim=embedding_dim)
    
    print("Loading catalog...")
    load_catalog(catalog_path, es, args.index, embedding_service=embedding_service)
    
    count = es.count(index=args.index)["count"]
    print(f"\nCatalog loaded successfully! Total products: {count}")
    if embedding_service:
        print("Embeddings: done")


if __name__ == "__main__":
    main()
