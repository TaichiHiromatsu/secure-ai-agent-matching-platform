#!/bin/bash
# Export current database as sample data for development
# Usage: ./deploy/export-sample-data.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Export Sample Data${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Create sample data directory
SAMPLE_DATA_DIR="$PROJECT_ROOT/deploy/sample-data"
mkdir -p "$SAMPLE_DATA_DIR"

# Check if container is running
if ! docker ps | grep -q secure-platform; then
    echo -e "${RED}Error: secure-platform container is not running${NC}"
    echo -e "${YELLOW}Please start the platform first with: ./deploy/run-local.sh${NC}"
    exit 1
fi

echo -e "${YELLOW}Exporting database from container...${NC}"
docker cp secure-platform:/app/trusted_agent_hub/data/agent_store.db "$SAMPLE_DATA_DIR/agent_store.db"

echo -e "${GREEN}✓ Database exported successfully${NC}"
echo ""

echo -e "${YELLOW}Database contents:${NC}"
docker exec secure-platform sqlite3 /app/trusted_agent_hub/data/agent_store.db \
    "SELECT submission_id, agent_id, state, created_at FROM submissions ORDER BY created_at DESC LIMIT 5;"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Export Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Sample database saved to:${NC}"
echo -e "  ${YELLOW}$SAMPLE_DATA_DIR/agent_store.db${NC}"
echo ""
echo -e "${BLUE}This database will be automatically loaded on next startup.${NC}"
echo ""
