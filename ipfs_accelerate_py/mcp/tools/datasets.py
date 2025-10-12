"""
Dataset MCP Tools

MCP server tools for managing and scraping legal datasets.
"""

import logging
from typing import Any, Dict, List, Optional

# Set up logging
logger = logging.getLogger("ipfs_accelerate_mcp.tools.datasets")


def register_dataset_tools(mcp: Any) -> None:
    """
    Register dataset management tools with the MCP server.
    
    Args:
        mcp: MCP server instance
    """
    logger.info("Registering dataset management tools")
    
    # Check if this is a FastMCP-style server
    if hasattr(mcp, "tool"):
        register_fastmcp_dataset_tools(mcp)
    else:
        register_standalone_dataset_tools(mcp)


def register_fastmcp_dataset_tools(mcp: Any) -> None:
    """Register dataset tools for FastMCP-style server."""
    
    @mcp.tool()
    def list_legal_datasets() -> Dict[str, Any]:
        """List all available legal datasets.
        
        Returns:
            Dictionary containing list of datasets with their information.
        """
        try:
            from ipfs_accelerate_py.legal_datasets_loader import get_all_datasets_info
            
            datasets = get_all_datasets_info()
            return {
                "success": True,
                "datasets": datasets,
                "total": len(datasets)
            }
        except Exception as e:
            logger.error(f"Error listing datasets: {e}")
            return {
                "success": False,
                "error": str(e),
                "datasets": []
            }
    
    @mcp.tool()
    def get_dataset_info(dataset_type: str) -> Dict[str, Any]:
        """Get information about a specific dataset.
        
        Args:
            dataset_type: Type of dataset (e.g., 'case_law_access_project', 'recap_archive')
            
        Returns:
            Dictionary containing dataset information.
        """
        try:
            from ipfs_accelerate_py.legal_datasets_loader import get_all_dataset_loaders
            
            loaders = get_all_dataset_loaders()
            if dataset_type not in loaders:
                return {
                    "success": False,
                    "error": f"Unknown dataset type: {dataset_type}"
                }
            
            loader = loaders[dataset_type]
            info = loader.get_dataset_info()
            
            return {
                "success": True,
                "dataset_info": info
            }
        except Exception as e:
            logger.error(f"Error getting dataset info: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @mcp.tool()
    def scrape_recap_archive(
        court: Optional[str] = None,
        date_filed: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Scrape documents from the RECAP Archive.
        
        Args:
            court: Filter by court identifier (optional)
            date_filed: Filter by filing date in YYYY-MM-DD format (optional)
            limit: Maximum number of documents to fetch (default: 100)
            
        Returns:
            Dictionary containing scraped documents.
        """
        try:
            from ipfs_accelerate_py.legal_datasets_loader import RECAPArchiveLoader
            
            loader = RECAPArchiveLoader()
            documents = loader.scrape_recap_documents(
                court=court,
                date_filed=date_filed,
                limit=limit
            )
            
            return {
                "success": True,
                "documents": documents,
                "total": len(documents),
                "parameters": {
                    "court": court,
                    "date_filed": date_filed,
                    "limit": limit
                }
            }
        except Exception as e:
            logger.error(f"Error scraping RECAP archive: {e}")
            return {
                "success": False,
                "error": str(e),
                "documents": []
            }
    
    @mcp.tool()
    def scrape_cap_cases(
        jurisdiction: Optional[str] = None,
        court: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Scrape cases from the Case Law Access Project.
        
        Args:
            jurisdiction: Filter by jurisdiction (e.g., 'us', 'cal') (optional)
            court: Filter by court name (optional)
            limit: Maximum number of cases to fetch (default: 100)
            
        Returns:
            Dictionary containing scraped cases.
        """
        try:
            from ipfs_accelerate_py.legal_datasets_loader import CaseLawAccessProjectLoader
            
            loader = CaseLawAccessProjectLoader()
            cases = loader.scrape_cases(
                jurisdiction=jurisdiction,
                court=court,
                limit=limit
            )
            
            return {
                "success": True,
                "cases": cases,
                "total": len(cases),
                "parameters": {
                    "jurisdiction": jurisdiction,
                    "court": court,
                    "limit": limit
                }
            }
        except Exception as e:
            logger.error(f"Error scraping CAP cases: {e}")
            return {
                "success": False,
                "error": str(e),
                "cases": []
            }
    
    @mcp.tool()
    def scrape_us_code(
        title: Optional[int] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Scrape sections from the US Code.
        
        Args:
            title: Filter by US Code title number (optional)
            limit: Maximum number of sections to fetch (default: 100)
            
        Returns:
            Dictionary containing scraped code sections.
        """
        try:
            from ipfs_accelerate_py.legal_datasets_loader import USCodeFederalRegisterLoader
            
            loader = USCodeFederalRegisterLoader()
            sections = loader.scrape_us_code(
                title=title,
                limit=limit
            )
            
            return {
                "success": True,
                "sections": sections,
                "total": len(sections),
                "parameters": {
                    "title": title,
                    "limit": limit
                }
            }
        except Exception as e:
            logger.error(f"Error scraping US Code: {e}")
            return {
                "success": False,
                "error": str(e),
                "sections": []
            }
    
    @mcp.tool()
    def scrape_state_laws(
        state: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Scrape state laws and statutes.
        
        Args:
            state: Filter by state code (e.g., 'CA', 'NY') (optional)
            limit: Maximum number of statutes to fetch (default: 100)
            
        Returns:
            Dictionary containing scraped statutes.
        """
        try:
            from ipfs_accelerate_py.legal_datasets_loader import StateLawsLoader
            
            loader = StateLawsLoader()
            statutes = loader.scrape_state_laws(
                state=state,
                limit=limit
            )
            
            return {
                "success": True,
                "statutes": statutes,
                "total": len(statutes),
                "parameters": {
                    "state": state,
                    "limit": limit
                }
            }
        except Exception as e:
            logger.error(f"Error scraping state laws: {e}")
            return {
                "success": False,
                "error": str(e),
                "statutes": []
            }
    
    @mcp.tool()
    def scrape_municipal_laws(
        city: Optional[str] = None,
        state: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Scrape municipal laws and ordinances.
        
        Args:
            city: Filter by city name (optional)
            state: Filter by state code (optional)
            limit: Maximum number of ordinances to fetch (default: 100)
            
        Returns:
            Dictionary containing scraped ordinances.
        """
        try:
            from ipfs_accelerate_py.legal_datasets_loader import MunicipalLawsLoader
            
            loader = MunicipalLawsLoader()
            ordinances = loader.scrape_municipal_laws(
                city=city,
                state=state,
                limit=limit
            )
            
            return {
                "success": True,
                "ordinances": ordinances,
                "total": len(ordinances),
                "parameters": {
                    "city": city,
                    "state": state,
                    "limit": limit
                }
            }
        except Exception as e:
            logger.error(f"Error scraping municipal laws: {e}")
            return {
                "success": False,
                "error": str(e),
                "ordinances": []
            }


def register_standalone_dataset_tools(mcp: Any) -> None:
    """Register dataset tools for standalone MCP server."""
    
    # For standalone servers, tools need to be registered differently
    # This is a placeholder for the standalone registration pattern
    logger.info("Dataset tools registration for standalone mode (placeholder)")


def register_tools(mcp: Any) -> None:
    """Main entry point for registering dataset tools.
    
    Args:
        mcp: MCP server instance
    """
    register_dataset_tools(mcp)
