# Documentation Exports

This directory contains exported documentation in formats other than Markdown.

## Contents

### HTML Exports
These HTML files are visual exports of documentation, diagrams, or interactive visualizations:

- **Causal_Proximity_Delegation.html** - Visualization of causal proximity delegation patterns
- **Huggingface_Model_Manager.html** - HuggingFace model manager documentation export
- **Huggingface_Model_Manager_with_data.html** - HuggingFace model manager with sample data
- **P2P_Network_Simulation.html** - Interactive P2P network simulation visualization

### PDF Exports
- **kitchen_sink_overview.pdf** - Comprehensive kitchen sink feature overview

## Purpose

These files serve as:
- **Visual Documentation**: Interactive diagrams and visualizations
- **Presentation Materials**: Ready-to-share documentation exports
- **Archive Formats**: Preserved documentation snapshots

## Usage

### Viewing HTML Files
Open HTML files directly in a web browser:
```bash
# Example
open Causal_Proximity_Delegation.html
# or
firefox P2P_Network_Simulation.html
```

### Generating New Exports
To generate new HTML or PDF exports from Markdown documentation, use appropriate tools:
```bash
# Example with pandoc
pandoc ../README.md -o README.html

# Example with markdown-pdf
markdown-pdf ../ARCHITECTURE.md
```

## Maintenance

- **Not Version Controlled**: Large binary files (like detailed PDFs) may not be tracked in git
- **Regenerate as Needed**: These exports should be regenerated when source documentation changes
- **Source of Truth**: Markdown files in the parent directory are the authoritative source

## Related Documentation

- **Source Documentation**: See parent [../README.md](../README.md)
- **Documentation Index**: See [../INDEX.md](../INDEX.md)

---

*Directory Created: January 2026*
*Purpose: Store non-markdown documentation exports*
