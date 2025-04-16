# Drug Crawler

A high-performance web crawler for collecting pharmaceutical drug data.

## Features

- Asynchronous crawling with concurrency control
- Two-phase crawling for improved performance
- Resumable crawling for better error recovery
- Multiple output formats (JSON, CSV, DuckDB)
- Image downloading and search capabilities

## Installation

```bash
pip install -r requirements.txt
```

## Usage

The drug-crawler now supports a two-phase crawling approach for better performance:

1. **Phase 1: Link Discovery** - Quickly gathers all drug URLs without processing details
2. **Phase 2: Link Processing** - Processes the saved URLs for detailed information

### Two-Phase Crawling (Recommended)

For best performance, run the crawler in two separate phases:

```bash
# Phase 1: Discover and save all drug URLs (fast)
python main.py discover

# Phase 2: Process the saved URLs in batches
python main.py process --batch-size 50
```

If your crawling process is interrupted, you can easily resume from where you left off:

```bash
# Resume from the last saved position
python main.py resume
```

### Original Single-Phase Crawling

You can also run the crawler in the original single-phase mode:

```bash
# Run the complete crawling process
python main.py full --max_groups 10 --max_pages 5
```

### Command Line Arguments

#### Common Options:

- `--output-format [json|csv|duckdb]`: Format to save the data (default: json)

#### For 'full' command:

- `--max_groups N`: Maximum number of drug groups to process
- `--max_pages N`: Maximum pages per group to process (default: 5)
- `--max_drugs N`: Maximum number of drugs to process

#### For 'process' command:

- `--batch-size N`: Number of links to process in each batch (default: 50)
- `--start-index N`: Index to start processing from (for manual resuming)

## Performance Improvements

The two-phase approach offers several significant performance benefits:

1. **Faster Link Collection**: Phase 1 is much faster as it only collects links without processing details
2. **Resumable Processing**: If crawling is interrupted, you can resume from where you left off
3. **Batch Processing**: Processes links in controlled batches to manage memory usage
4. **Better Error Recovery**: Errors in one drug don't affect the entire process

## Output Formats

- **JSON**: Simple, readable format for general use
- **CSV**: Tabular format for spreadsheet software
- **DuckDB**: High-performance analytical database

## Example

```bash
# Phase 1: Discover all links
python main.py discover --output-format json

# Phase 2: Process links in larger batches
python main.py process --batch-size 100 --output-format json
```

This approach allows for much better performance on large datasets by separating the fast URL collection from the more intensive data processing.
