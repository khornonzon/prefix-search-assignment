import re
from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch
from transliterate import translit

from embeddings import EmbeddingService



class SearchEngine:
    TEXT_WEIGHT = 0.7
    VECTOR_WEIGHT = 0.3
    
    LAYOUT_MAP = {
        'q': 'й', 'w': 'ц', 'e': 'у', 'r': 'к', 't': 'е', 'y': 'н', 'u': 'г', 'i': 'ш', 'o': 'щ', 'p': 'з',
        'a': 'ф', 's': 'ы', 'd': 'в', 'f': 'а', 'g': 'п', 'h': 'р', 'j': 'о', 'k': 'л', 'l': 'д',
        'z': 'я', 'x': 'ч', 'c': 'с', 'v': 'м', 'b': 'и', 'n': 'т', 'm': 'ь',
        '[': 'х', ']': 'ъ', ';': 'ж', "'": 'э', ',': 'б', '.': 'ю', '/': '.'
    }
    
    REVERSE_LAYOUT_MAP = {v: k for k, v in LAYOUT_MAP.items()}
    
    NUMERIC_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*(кг|kg|л|l|лт|ml|г|g|мл)', re.IGNORECASE)
    
    def __init__(self, es_host: str = "http://localhost:9200", index_name: str = "catalog",
                 use_embeddings: bool = True):
        self.es = Elasticsearch([es_host])
        self.index_name = index_name
        self.use_embeddings = use_embeddings
        self.embeddings_available = self._detect_embedding_field()
        
        self.embedding_service = None
        if self.use_embeddings and self.embeddings_available:
            self.embedding_service = EmbeddingService()
        elif self.use_embeddings and not self.embeddings_available:
            print("Embedding field not detected in index; vector search disabled.")
            self.use_embeddings = False
        
    
    def switch_keyboard_layout(self, text: str) -> str:
        result = []
        for char in text:
            if char.lower() in self.LAYOUT_MAP:
                mapped = self.LAYOUT_MAP[char.lower()]
                result.append(mapped.upper() if char.isupper() else mapped)
            elif char.lower() in self.REVERSE_LAYOUT_MAP:
                mapped = self.REVERSE_LAYOUT_MAP[char.lower()]
                result.append(mapped.upper() if char.isupper() else mapped)
            else:
                result.append(char)
        return ''.join(result)
    
    def transliterate(self, text: str) -> str:
        try:
            return translit(text, 'ru', reversed=True)
        except Exception:
            return text
    
    def extract_numeric_attributes(self, query: str) -> Dict[str, Any]:
        matches = self.NUMERIC_PATTERN.findall(query)
        attributes = {}
        
        for value, unit in matches:
            num_value = float(value)
            unit_lower = unit.lower()
            if unit_lower in ['кг', 'kg']:
                attributes['weight_kg'] = num_value
            elif unit_lower in ['л', 'l', 'лт']:
                attributes['volume_l'] = num_value
            elif unit_lower in ['г', 'g']:
                attributes['weight_g'] = num_value
            elif unit_lower in ['мл', 'ml']:
                attributes['volume_ml'] = num_value
        
        return attributes
    
    def normalize_query(self, query: str) -> tuple[str, Dict[str, Any], Optional[List[float]]]:

        original_query = query.strip()
        numeric_attrs = self.extract_numeric_attributes(original_query)
        text_query = self.NUMERIC_PATTERN.sub('', query).strip()

        variations = [text_query]
        
        layout_switched = self.switch_keyboard_layout(text_query)
        if layout_switched != text_query:
            variations.append(layout_switched)
        
        if any('\u0400' <= c <= '\u04FF' for c in text_query):
            transliterated = self.transliterate(text_query)
            if transliterated != text_query:
                variations.append(transliterated)
        
        normalized = ' '.join(variations)
        
        query_embedding = None
        if self.embedding_service:
            try:
                query_embedding = self.embedding_service.encode(normalized)[0].tolist()
            except Exception as e:
                print(f"Warning: Could not generate query embedding: {e}")
        
        return normalized, numeric_attrs, query_embedding
    
    def build_search_query(self, query: str, numeric_attrs: Dict[str, Any], 
                          top_k: int = 5) -> Dict[str, Any]:

        should_clauses = []

        should_clauses.append({
            "match": {
                "name": {
                    "query": query,
                    "boost": 5.0
                }
            }
        })
        
        should_clauses.append({
            "match_bool_prefix": {
                "name": {
                    "query": query,
                    "boost": 4.0
                }
            }
        })
        
        should_clauses.append({
            "match": {
                "brand": {
                    "query": query,
                    "boost": 3.0
                }
            }
        })
        
        should_clauses.append({
            "match": {
                "keywords": {
                    "query": query,
                    "boost": 2.0
                }
            }
        })
        
        should_clauses.append({
            "match": {
                "category.text": {
                    "query": query,
                    "boost": 1.5
                }
            }
        })
        
        should_clauses.append({
            "match": {
                "description": {
                    "query": query,
                    "boost": 1.0
                }
            }
        })
        
        es_query = {
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 1
            }
        }
        
        if numeric_attrs:
            numeric_should = []
            if 'weight_kg' in numeric_attrs:
                target_weight = numeric_attrs['weight_kg']
                numeric_should.append({
                    "range": {
                        "weight": {
                            "gte": target_weight * 0.8,
                            "lte": target_weight * 1.2,
                            "boost": 2.0
                        }
                    }
                })
            if 'volume_l' in numeric_attrs:
                volume_ml = numeric_attrs['volume_l'] * 1000
                numeric_should.append({
                    "range": {
                        "weight": {
                            "gte": volume_ml * 0.8,
                            "lte": volume_ml * 1.2,
                            "boost": 2.0
                        }
                    }
                })
            
            if numeric_should:
                es_query["bool"]["should"].extend(numeric_should)
        
        query_body = {
            "size": top_k,
            "query": es_query,
            "_source": ["id", "name", "category", "brand", "weight", "weight_unit", "price", "image_url"],
            "highlight": {
                "fields": {
                    "name": {},
                    "brand": {},
                    "keywords": {}
                }
            }
        }
        
        return query_body
    
    def _detect_embedding_field(self) -> bool:
        try:
            mapping = self.es.indices.get_mapping(index=self.index_name)
            props = mapping.get(self.index_name, {}).get("mappings", {}).get("properties", {})
            embedding_field = props.get("embedding")
            return bool(embedding_field and embedding_field.get("type") == "dense_vector")
        except Exception as exc:
            print(f"Warning: could not inspect index mapping for embeddings: {exc}")
            return False
    
    def run_vector_search(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        if not (self.embedding_service and self.embeddings_available and query_embedding):
            return []
        
        response = self.es.search(
            index=self.index_name,
            knn={
                "field": "embedding",
                "query_vector": query_embedding,
                "k": top_k,
                "num_candidates": max(top_k * 5, 50),
            },
            size=top_k,
            _source=["id", "name", "category", "brand", "weight", "weight_unit", "price", "image_url"],
        )
        return response["hits"]["hits"]
    
    def filter_noise(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        if not results:
            return results
        
        filtered = []
        query_lower = query.lower()
        query_words = [word for word in query_lower.split() if len(word) > 1]
        
        for result in results:
            score = result.get('score', 0) or 0.0
            if not query_words:
                filtered.append(result)
                continue
            
            name = (result.get('name') or '').lower()
            brand = (result.get('brand') or '').lower()
            category = (result.get('category') or '').lower()
            
            fields = " ".join([name, brand, category])
            has_match = any(word in fields for word in query_words)
            
            if has_match or score >= 0.5:
                filtered.append(result)
        
        return filtered
    
    def search(self, query: str, top_k: int = 5, use_embeddings: Optional[bool] = False) -> List[Dict[str, Any]]:
        use_embeddings_flag = self.use_embeddings if use_embeddings is None else use_embeddings
        
        normalized_query, numeric_attrs, query_embedding = self.normalize_query(query)
        
        es_query = self.build_search_query(
            normalized_query, 
            numeric_attrs, 
            top_k=top_k * 2
        )
        response = self.es.search(index=self.index_name, **es_query)
        text_hits = response['hits']['hits']
        
        vector_hits = []
        if (
            use_embeddings_flag
            and self.embedding_service
            and self.embeddings_available
            and query_embedding is not None
        ):
            try:
                vector_hits = self.run_vector_search(query_embedding, top_k * 2)
            except Exception as exc:
                print(f"Vector search failed, falling back to text only: {exc}")
                vector_hits = []
        
        combined = {}
        
        def ensure_entry(hit: Dict[str, Any]) -> Dict[str, Any]:
            source = hit['_source']
            doc_id = source.get('id') or hit.get('_id')
            if doc_id not in combined:
                combined[doc_id] = {
                    "id": doc_id,
                    "name": source.get('name'),
                    "category": source.get('category'),
                    "brand": source.get('brand'),
                    "weight": source.get('weight'),
                    "weight_unit": source.get('weight_unit'),
                    "price": source.get('price'),
                    "image_url": source.get('image_url'),
                    "text_score": 0.0,
                    "vector_score": 0.0,
                }
            return combined[doc_id]
        
        max_text_score = text_hits[0]['_score'] if text_hits else 0.0
        max_vector_score = vector_hits[0]['_score'] if vector_hits else 0.0
        
        for hit in text_hits:
            entry = ensure_entry(hit)
            entry["text_score"] = max(entry["text_score"], hit.get('_score', 0.0) or 0.0)
        
        for hit in vector_hits:
            entry = ensure_entry(hit)
            entry["vector_score"] = max(entry["vector_score"], hit.get('_score', 0.0) or 0.0)
        
        results = []
        vector_weight = self.VECTOR_WEIGHT if vector_hits else 0.0
        text_weight = 1.0 - vector_weight if vector_weight else 1.0
        
        for entry in combined.values():
            norm_text = (entry["text_score"] / max_text_score) if max_text_score > 0 else 0.0
            norm_vector = (entry["vector_score"] / max_vector_score) if max_vector_score > 0 else 0.0
            combined_score = (text_weight * norm_text) + (vector_weight * norm_vector)
            result = {
                "id": entry["id"],
                "name": entry["name"],
                "category": entry["category"],
                "brand": entry["brand"],
                "weight": entry["weight"],
                "weight_unit": entry["weight_unit"],
                "price": entry["price"],
                "image_url": entry["image_url"],
                "score": combined_score,
            }
            results.append(result)
        
        filtered_results = self.filter_noise(sorted(results, key=lambda x: x["score"], reverse=True), normalized_query)
        return filtered_results[:top_k]

