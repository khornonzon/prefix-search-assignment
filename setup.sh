#!/bin/bash
# Setup script for prefix search assignment

set -e

echo "Starting Elasticsearch and search service..."
docker compose up -d 

echo "Waiting for Elasticsearch to be ready..."
max_retries=60
retry_count=0

while [ $retry_count -lt $max_retries ]; do
    if curl -s http://localhost:9200/_cluster/health > /dev/null 2>&1; then
        echo "Elasticsearch is ready!"
        break
    fi
    retry_count=$((retry_count + 1))
    echo "Waiting for Elasticsearch... ($retry_count/$max_retries)"
    sleep 1
done

if [ $retry_count -eq $max_retries ]; then
    echo "Error: Elasticsearch did not become ready in time"
    exit 1
fi

echo "Loading catalog into Elasticsearch..."
python tools/load_catalog.py --host localhost --port 9200

echo "Setup complete!"
echo ""
echo "You can now:"
echo "  1. Test the search API: curl 'http://localhost:5000/search?q=масло'"
echo "  2. Run evaluation: python tools/evaluate.py --type open"
echo "  3. Check service health: curl http://localhost:5000/health"

