# Legal Datasets and RECAP Archive Integration

## Overview

This implementation provides a comprehensive infrastructure for managing and scraping legal datasets, including the RECAP archive, through the MCP dashboard.

## Components

### 1. Dataset Loaders (`ipfs_accelerate_py/legal_datasets_loader.py`)

Base class `LegalDatasetLoader` with specialized loaders for:

- **CaseLawAccessProjectLoader**: Scrapes U.S. court decisions from case.law
- **USCodeFederalRegisterLoader**: Scrapes US Code and Federal Register from govinfo.gov
- **StateLawsLoader**: Scrapes state-level legislation
- **MunicipalLawsLoader**: Scrapes city and county ordinances
- **RECAPArchiveLoader**: Scrapes PACER documents from CourtListener

Each loader provides:
- `get_dataset_info()`: Returns metadata about the dataset
- `scrape_*()`: Methods to scrape data from the source
- `load_dataset()`: Loads cached or sample data

### 2. MCP Server Tools (`ipfs_accelerate_py/mcp/tools/datasets.py`)

MCP server tools that expose dataset operations:

- `list_legal_datasets()`: Lists all available datasets
- `get_dataset_info(dataset_type)`: Gets information about a specific dataset
- `scrape_recap_archive(court, date_filed, limit)`: Scrapes RECAP documents
- `scrape_cap_cases(jurisdiction, court, limit)`: Scrapes CAP cases
- `scrape_us_code(title, limit)`: Scrapes US Code sections
- `scrape_state_laws(state, limit)`: Scrapes state laws
- `scrape_municipal_laws(city, state, limit)`: Scrapes municipal ordinances

### 3. Dashboard Integration

#### New Routes (`ipfs_accelerate_py/mcp_dashboard.py`)

- `GET /mcp/datasets`: Datasets management page
- `GET /api/mcp/datasets`: API to list all datasets
- `GET /api/mcp/datasets/<dataset_type>`: API to get dataset info
- `POST /api/mcp/datasets/recap/scrape`: API to scrape RECAP archive
- `POST /api/mcp/datasets/cap/scrape`: API to scrape CAP cases

#### UI Features

1. **Main Dashboard Tab**: Added "Legal Datasets" tab showing:
   - Summary of all 5 datasets
   - Status badges (currently showing "Empty")
   - Link to full datasets manager

2. **Datasets Management Page** (`/mcp/datasets`):
   - Overview of all datasets with descriptions and URLs
   - Interactive RECAP scraper with form inputs:
     - Court identifier (e.g., ca9, dcd)
     - Date filed (date picker)
     - Limit (number of documents)
   - Interactive CAP scraper with form inputs:
     - Jurisdiction (e.g., us, cal)
     - Court name
     - Limit (number of cases)
   - Real-time feedback on scraping operations

## Usage

### Starting the Dashboard

```python
from ipfs_accelerate_py.mcp_dashboard import MCPDashboard

dashboard = MCPDashboard(port=8900)
dashboard.run()
```

Navigate to: http://127.0.0.1:8900/mcp/datasets

### Using Dataset Loaders Programmatically

```python
from ipfs_accelerate_py.legal_datasets_loader import (
    RECAPArchiveLoader,
    CaseLawAccessProjectLoader,
    get_all_datasets_info
)

# List all datasets
datasets = get_all_datasets_info()
print(f"Found {len(datasets)} datasets")

# Use RECAP loader
recap = RECAPArchiveLoader()
documents = recap.scrape_recap_documents(
    court='ca9',
    limit=100
)

# Use CAP loader
cap = CaseLawAccessProjectLoader()
cases = cap.scrape_cases(
    jurisdiction='us',
    court='Supreme Court',
    limit=50
)
```

### Using MCP Tools

The dataset tools are automatically registered with the MCP server and can be called through the MCP protocol.

### API Examples

```bash
# List all datasets
curl http://127.0.0.1:8900/api/mcp/datasets

# Get RECAP dataset info
curl http://127.0.0.1:8900/api/mcp/datasets/recap_archive

# Scrape RECAP archive
curl -X POST http://127.0.0.1:8900/api/mcp/datasets/recap/scrape \
  -H "Content-Type: application/json" \
  -d '{"court": "ca9", "limit": 100}'

# Scrape CAP cases
curl -X POST http://127.0.0.1:8900/api/mcp/datasets/cap/scrape \
  -H "Content-Type: application/json" \
  -d '{"jurisdiction": "us", "court": "Supreme Court", "limit": 50}'
```

## Dataset Status

All datasets are currently marked as "Empty" and return placeholder data. The infrastructure is in place for implementing the actual scraping logic.

## Next Steps

To fully implement the scraping functionality:

1. **Add API Keys/Authentication**:
   - Case Law Access Project requires API key
   - Some sources may require authentication

2. **Implement Scraping Logic**:
   - Add actual HTTP requests to data sources
   - Handle pagination and rate limiting
   - Parse response data into standardized format

3. **Add Data Storage**:
   - Implement caching mechanism
   - Store scraped data in structured format
   - Add database integration for persistent storage

4. **ETL Pipeline**:
   - Transform raw scraped data
   - Normalize data formats
   - Extract metadata and relationships

5. **Workflow Orchestration**:
   - Create MCP workflows for automated scraping
   - Schedule periodic updates
   - Monitor scraping jobs

## File Structure

```
ipfs_accelerate_py/
├── legal_datasets_loader.py         # Dataset loader classes
├── mcp/
│   └── tools/
│       ├── __init__.py               # Updated to register dataset tools
│       └── datasets.py               # MCP dataset tools
├── mcp_dashboard.py                  # Updated with dataset routes
└── templates/
    └── dashboard.html                # Updated with datasets tab
```

## Dependencies

- Flask and Flask-CORS (already required by dashboard)
- requests (for HTTP requests to data sources)
- Standard library modules (os, json, pathlib, logging)

## Testing

All components have been tested:

✅ Dataset loaders instantiate correctly
✅ API endpoints return proper JSON responses
✅ Dashboard UI renders all datasets
✅ Scraper forms expand/collapse correctly
✅ MCP tools register successfully

## Screenshots

See the PR for screenshots showing:
- Datasets overview page
- RECAP scraper form
- CAP scraper form
- Main dashboard integration
