import os
import asyncio
import aiohttp
import aiofiles
import time
import json
from pathlib import Path
from async_lru import alru_cache
from aiolimiter import AsyncLimiter
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import List, Optional, Set, Tuple
from dataclasses import dataclass
from duckduckgo_search import DDGS
from .config import (
    GROUP_SELECTORS,
    DRUG_SELECTORS,
    MANUFACTURER_SELECTORS,
    IMAGE_SELECTORS,
    INGREDIENTS_SELECTORS,
    REQUEST_TIMEOUT,
    REQUEST_DELAY,
    PAGE_DELAY,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_OUTPUT_PATH
)
ddgs_images=DDGS()
@dataclass
class CommercialDrug:
    persian_name: str
    english_name: str
    manufacturer: str
    url: str
    image_url: Optional[List[str]] = None

@dataclass
class DrugDetails:
    id: str
    english_name: str
    persian_name: str
    manufacturer: str
    image_path: Optional[str]
    combinations: List[str]
    url: str
    commercial_drugs: List[CommercialDrug] = None

class DrugDataCrawler:
    def __init__(self, base_url: str = "https://www.darooyab.ir", 
                start_url: str = "https://www.darooyab.ir/DrugGroups",
                output_format: str = DEFAULT_OUTPUT_FORMAT,
                output_path: str = DEFAULT_OUTPUT_PATH,
                max_concurrent: int = 10):
        self.base_url = base_url
        self.start_url = start_url
        self.output_format = output_format
        self.output_path = output_path
        self.max_concurrent = max_concurrent
        
        # Performance monitoring
        from .utils import PerformanceMonitor
        self.performance_monitor = PerformanceMonitor()
        
        # Create directories
        self._create_output_dirs()
        
        # Add colorama for Windows color support
        try:
            from colorama import init
            init()
        except ImportError:
            pass
            
    # PHASE 1: Link Discovery Methods
    async def discover_all_drug_links(self):
        """
        Phase 1: Discover and save all drug links without processing details.
        This is much faster than processing everything at once.
        """
        # Start the discovery process
        
        print("🔍 Starting Phase 1: Drug Link Discovery")
        drug_links = []
        discovery_start_time = time.time()
        
        # Create the caching directory if it doesn't exist
        cache_dir = os.path.join(self.output_path, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create HTML cache directory
        html_cache_dir = os.path.join(cache_dir, "html")
        os.makedirs(html_cache_dir, exist_ok=True)
        
        async with await self._create_session() as session:
            # Get all drug group URLs
            print("📋 Fetching drug group URLs...")
            group_start_time = time.time()
            group_urls = await self.get_drug_groups(session)
            group_elapsed = time.time() - group_start_time
            print(f"✅ Found {len(group_urls)} drug groups in {group_elapsed:.2f}s")
            
            # Save group URLs for potential restarts
            group_cache_file = os.path.join(cache_dir, "group_urls.json")
            async with aiofiles.open(group_cache_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(group_urls, ensure_ascii=False, indent=2))
            
            # Get all drug URLs from each group using asyncio.gather for parallel processing
            print("🔎 Fetching drug URLs from groups...")
            
            # Create a progress bar for group processing with ETAs
            pbar = tqdm(total=len(group_urls), desc="Processing drug groups", 
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            
            # Process drug groups with a semaphore to control concurrency
            semaphore = asyncio.Semaphore(self.max_concurrent)
            drug_urls_found = 0
              # Batch process groups for better memory management
            batch_size = 25  # Process groups in batches
            drug_urls_found = 0  # Initialize counter
            
            async def process_group_batch(group_batch):
                nonlocal drug_urls_found
                batch_results = []
                
                # Process each group in the batch with controlled concurrency
                async def process_single_group(group_url):
                    async with semaphore:
                        group_start = time.time()
                        urls = await self.get_drug_urls_from_group(session, group_url)
                        group_time = time.time() - group_start
                        
                        # Log timing for group processing
                        urls_count = len(urls)
                        drug_urls_found += urls_count
                        pbar.set_postfix(urls=drug_urls_found, last_batch=f"{urls_count} in {group_time:.2f}s")
                        pbar.update(1)
                        
                        return urls
                
                # Process all groups in this batch concurrently
                group_results = await asyncio.gather(*[process_single_group(url) for url in group_batch])
                
                # Flatten the results
                for urls in group_results:
                    batch_results.extend(urls)
                    
                return batch_results
            
            # Process all groups in batches
            for i in range(0, len(group_urls), batch_size):
                batch = group_urls[i:min(i+batch_size, len(group_urls))]
                batch_links = await process_group_batch(batch)
                drug_links.extend(batch_links)
                
                # Save intermediate results
                if i % (batch_size * 4) == 0 and drug_links:
                    intermediate_file = os.path.join(cache_dir, f"drug_links_partial_{i}.json")
                    async with aiofiles.open(intermediate_file, "w", encoding="utf-8") as f:
                        await f.write(json.dumps(list(set(drug_links)), ensure_ascii=False, indent=2))
            
            # Remove duplicates
            drug_links = list(set(drug_links))
            pbar.close()
            
            total_time = time.time() - discovery_start_time
            print(f"✅ Found {len(drug_links)} unique drug URLs in {total_time:.2f}s")
            print(f"📊 Average time per drug URL: {total_time/len(drug_links):.4f}s")
            
            # Save to file for later processing
            cache_file = os.path.join(cache_dir, "drug_links.json")
            async with aiofiles.open(cache_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(drug_links, ensure_ascii=False, indent=2))
            
            print(f"💾 Drug links saved to {cache_file}")
            
            # Print performance summary
            self.performance_monitor.print_summary()
            
            return drug_links
      # PHASE 2: Link Processing Methods
    async def process_saved_links(self, batch_size: int = 50, start_index: int = 0, 
                                max_retries: int = 3, delay_between_batches: float = 1.0):
        """
        Phase 2: Process previously saved drug links in batches with improved performance.
        This allows for resumable crawling and better error recovery.
        
        Args:
            batch_size: Number of links to process in each batch
            start_index: Index to start processing from (for resuming)
            max_retries: Maximum number of retries for failed requests
            delay_between_batches: Delay in seconds between processing batches
        """
        from .data_handlers import get_data_handler
        
        print("🔄 Starting Phase 2: Processing Drug Details")
        process_start_time = time.time()
        cache_dir = os.path.join(self.output_path, "cache")
        cache_file = os.path.join(cache_dir, "drug_links.json")
        errors_file = os.path.join(cache_dir, "processing_errors.json")
        failed_urls = []
        processing_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "retried": 0,
            "batch_times": [],
            "start_time": time.time()
        }
        
        try:
            # Load saved links
            async with aiofiles.open(cache_file, "r", encoding="utf-8") as f:
                content = await f.read()
                drug_links = json.loads(content)
            
            # Create a data handler for saving results
            data_handler = get_data_handler(self.output_format, self.output_path)
            
            # Process in batches
            total_links = len(drug_links)
            if start_index >= total_links:
                print(f"⚠️ Start index {start_index} is out of range (total links: {total_links})")
                return
            
            print(f"📊 Processing {total_links} links in batches of {batch_size}, starting from index {start_index}")
            
            # Create a progress bar for the entire processing with ETA
            main_pbar = tqdm(
                total=total_links-start_index, 
                desc="Overall progress", 
                position=0,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
            
            # Track progress for resuming later
            progress_file = os.path.join(cache_dir, "crawl_progress.json")
            
            # Function to process a batch of URLs concurrently with retry logic
            async def process_url_batch(urls):
                batch_results = []
                batch_semaphore = asyncio.Semaphore(min(self.max_concurrent, len(urls)))
                batch_start_time = time.time()
                
                async def process_single_url(url, attempt=1):
                    nonlocal processing_stats
                    processing_stats["total_processed"] += 1
                    url_start_time = time.time()
                    
                    try:
                        async with batch_semaphore:
                            drug_details = await self.extract_drug_details(session, url)
                            url_elapsed = time.time() - url_start_time
                            
                            if drug_details:
                                processing_stats["successful"] += 1
                                batch_results.append(drug_details)
                                # Save each drug immediately after scraping
                                data_handler.save_incremental([drug_details])
                                batch_pbar.set_postfix(success=f"{processing_stats['successful']}/{processing_stats['total_processed']}", 
                                                     time=f"{url_elapsed:.2f}s")
                            else:
                                # If extraction returned None but didn't raise an exception
                                if attempt < max_retries:
                                    processing_stats["retried"] += 1
                                    # Exponential backoff for retries
                                    retry_delay = REQUEST_DELAY * (2 ** (attempt - 1))
                                    await asyncio.sleep(retry_delay)
                                    return await process_single_url(url, attempt + 1)
                                else:
                                    processing_stats["failed"] += 1
                                    failed_urls.append({"url": url, "error": "Extraction returned None"})
                            
                            batch_pbar.update(1)
                            main_pbar.update(1)
                            
                    except Exception as e:
                        url_elapsed = time.time() - url_start_time
                        error_msg = f"{type(e).__name__}: {str(e)}"
                        
                        if attempt < max_retries:
                            processing_stats["retried"] += 1
                            print(f"⚠️ Retry {attempt}/{max_retries} for {url}: {error_msg}")
                            # Exponential backoff for retries
                            retry_delay = REQUEST_DELAY * (2 ** (attempt - 1))
                            await asyncio.sleep(retry_delay)
                            return await process_single_url(url, attempt + 1)
                        else:
                            # Failed after all retries
                            processing_stats["failed"] += 1
                            failed_urls.append({"url": url, "error": error_msg})
                            print(f"❌ Failed after {max_retries} attempts: {url} - {error_msg}")
                            batch_pbar.update(1)
                            main_pbar.update(1)
                
                # Process all URLs in the batch concurrently
                batch_pbar = tqdm(
                    total=len(urls), 
                    desc=f"Batch {len(processing_stats['batch_times'])+1}", 
                    position=1, 
                    leave=False,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}'
                )
                
                await asyncio.gather(*[process_single_url(url) for url in urls])
                batch_elapsed = time.time() - batch_start_time
                processing_stats["batch_times"].append(batch_elapsed)
                
                batch_pbar.close()
                return batch_results
            
            current_index = start_index
            async with await self._create_session() as session:
                while current_index < total_links:
                    # Calculate end index for current batch
                    end_index = min(current_index + batch_size, total_links)
                    batch_links = drug_links[current_index:end_index]
                    
                    # Process the current batch concurrently
                    batch_start = time.time()
                    batch_results = await process_url_batch(batch_links)
                    batch_time = time.time() - batch_start
                    
                    # Save batch results
                    if batch_results:
                        data_handler.save_incremental(batch_results)
                    
                    # Print batch stats
                    success_rate = len(batch_results) / len(batch_links) * 100
                    print(f"✅ Batch completed: {len(batch_results)}/{len(batch_links)} successful ({success_rate:.1f}%) in {batch_time:.2f}s")
                    
                    # Update progress file for potential resuming later
                    async with aiofiles.open(progress_file, "w", encoding="utf-8") as f:
                        await f.write(json.dumps({
                            "current_index": end_index,
                            "total_links": total_links,
                            "stats": processing_stats,
                            "last_update": time.time()
                        }, indent=2))
                    
                    # Save any errors
                    if failed_urls:
                        async with aiofiles.open(errors_file, "w", encoding="utf-8") as f:
                            await f.write(json.dumps(failed_urls, indent=2))
                    
                    # Move to next batch
                    current_index = end_index
                    
                    # Add a small delay between batches to avoid overwhelming the server
                    await asyncio.sleep(delay_between_batches)
            
            # Clean up
            main_pbar.close()
            
            # Calculate and display final statistics
            total_time = time.time() - process_start_time
            success_rate = processing_stats["successful"] / processing_stats["total_processed"] * 100 if processing_stats["total_processed"] > 0 else 0
            
            print("\n📊 Processing Summary:")
            print(f"  • Total Time: {total_time:.2f} seconds")
            print(f"  • Processed: {processing_stats['total_processed']} URLs")
            print(f"  • Successful: {processing_stats['successful']} ({success_rate:.1f}%)")
            print(f"  • Failed: {processing_stats['failed']}")
            print(f"  • Retried: {processing_stats['retried']}")
            
            if processing_stats["batch_times"]:
                avg_batch_time = sum(processing_stats["batch_times"]) / len(processing_stats["batch_times"])
                print(f"  • Average Batch Time: {avg_batch_time:.2f}s")
            
            print(f"\n✅ Processing completed! Data saved in {self.output_format} format.")
            
            # Print performance monitor summary
            self.performance_monitor.print_summary()
            
        except FileNotFoundError:
            print(f"❌ No saved links found at {cache_file}. Run the link discovery phase first.")
        except Exception as e:
            print(f"❌ Error processing saved links: {str(e)}")
            
    async def resume_processing(self):
        """Resume processing from the last saved position."""
        cache_dir = os.path.join(self.output_path, "cache")
        progress_file = os.path.join(cache_dir, "crawl_progress.json")
        
        try:
            async with aiofiles.open(progress_file, "r", encoding="utf-8") as f:
                content = await f.read()
                progress = json.loads(content)
            
            current_index = progress.get("current_index", 0)
            print(f"📋 Resuming from index {current_index}")
            await self.process_saved_links(start_index=current_index)
            
        except FileNotFoundError:
            print("❌ No progress file found. Starting from the beginning...")
            await self.process_saved_links()
        except Exception as e:
            print(f"❌ Error resuming process: {str(e)}")

    # Original methods follow below
    async def _create_session(self) -> aiohttp.ClientSession:
        connector = aiohttp.TCPConnector(limit=self.max_concurrent, force_close=False)
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        return aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9,fa;q=0.8'
            }
        )
        
    def _create_output_dirs(self):
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "images"), exist_ok=True)
        
    @alru_cache(maxsize=1000)
    async def get_page_content(self, session: aiohttp.ClientSession, url: str) -> Optional[BeautifulSoup]:
        """Get page content with caching to avoid redundant requests."""
        # Get page content from server or cache
        
        # Use the cache if available
        cache_dir = os.path.join(self.output_path, "cache", "html")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{url.split('/')[-1]}.html")
        
        # Check if we have a cached version
        if os.path.exists(cache_path) and (time.time() - os.path.getmtime(cache_path) < 86400):  # 24 hour cache
            try:
                async with aiofiles.open(cache_path, 'r', encoding='utf-8') as f:
                    html = await f.read()
                    return BeautifulSoup(html, 'lxml')
            except Exception:
                # If cache read fails, continue with request
                pass
        
        # Log request timing
        start_time = time.time()
        try:
            async with session.get(url) as response:
                end_time = time.time()
                elapsed = end_time - start_time
                
                # Log request timing
                print(f"📊 Request to {url.split('/')[-1]} took {elapsed:.4f}s")
                
                if response.status == 200:
                    html = await response.text()
                    
                    # Cache the HTML content
                    try:
                        async with aiofiles.open(cache_path, 'w', encoding='utf-8') as f:
                            await f.write(html)
                    except Exception as e:
                        print(f"⚠️ Failed to cache HTML: {str(e)}")
                    
                    return BeautifulSoup(html, 'lxml')
                elif response.status in (429, 503):  # Rate limited or service unavailable
                    print(f"⚠️ Rate limited on {url}, retrying after delay...")
                    await asyncio.sleep(REQUEST_DELAY * 3)  # Triple the delay
                    return await self.get_page_content(session, url)
                else:
                    # Log other HTTP errors with status code
                    print(f"❌ HTTP Error {response.status} for {url}: {response.reason}")
                    # Log additional headers that might be useful for debugging
                    if 'Retry-After' in response.headers:
                        print(f"   Retry-After: {response.headers['Retry-After']}")
                return None
        except aiohttp.ClientConnectorError as e:
            print(f"❌ Connection Error fetching {url}: {e.__class__.__name__} - {str(e)}")
            print("   This might indicate a network issue or the site is blocking requests")
            await asyncio.sleep(REQUEST_DELAY)
            return None
        except aiohttp.ClientPayloadError as e:
            print(f"❌ Payload Error fetching {url}: Content download interrupted - {str(e)}")
            await asyncio.sleep(REQUEST_DELAY)
            return None
        except aiohttp.ClientResponseError as e:
            print(f"❌ Response Error fetching {url}: {e.status} - {e.message}")
            await asyncio.sleep(REQUEST_DELAY)
            return None
        except aiohttp.ClientError as e:
            print(f"❌ Client Error fetching {url}: {e.__class__.__name__} - {str(e)}")
            await asyncio.sleep(REQUEST_DELAY)
            return None
        except asyncio.TimeoutError:
            print(f"⏱️ Timeout Error fetching {url}: Request took too long to complete")
            await asyncio.sleep(REQUEST_DELAY * 2)  # Wait longer for timeout errors
            return None
        except Exception as e:
            print(f"❌ Unexpected error fetching {url}: {e.__class__.__name__} - {str(e)}")
            await asyncio.sleep(REQUEST_DELAY)
            return None

    async def get_drug_groups(self, session: aiohttp.ClientSession) -> List[str]:
        """Get list of drug group URLs from the drug groups page."""
        group_urls = []
        soup = await self.get_page_content(session, self.start_url)
        if not soup:
            return group_urls

        for selector in GROUP_SELECTORS:
            group_links = soup.select(selector)
            if group_links:
                for link in group_links:
                    href = link.get('href')
                    if href:
                        group_url = urljoin(self.base_url, href)
                        if group_url not in group_urls:
                            group_urls.append(group_url)
                break
        return group_urls

    async def get_drug_urls_from_group(self, session: aiohttp.ClientSession, 
                                     group_url: str, max_pages: int = 5) -> List[str]:
        """Get drug URLs from a specific drug group."""
        drug_urls = []
        soup = await self.get_page_content(session, group_url)
        if not soup:
            return drug_urls

        drug_urls.extend(self._extract_drug_urls_from_page(soup))

        if max_pages > 1:
            for page in range(2, max_pages + 1):
                page_url = f"{group_url}?page={page}"
                page_soup = await self.get_page_content(session, page_url)
                if page_soup:
                    drug_urls.extend(self._extract_drug_urls_from_page(page_soup))
                await asyncio.sleep(PAGE_DELAY)
        return drug_urls

    def _extract_drug_urls_from_page(self, soup: BeautifulSoup) -> List[str]:
        """Extract drug URLs from a single page."""
        drug_urls = []
        for selector in DRUG_SELECTORS:
            drug_links = soup.select(selector)
            if drug_links:
                for link in drug_links:
                    href = link.get('href')
                    if href:
                        drug_url = urljoin(self.base_url, href)
                        drug_urls.append(drug_url)
                break
        return drug_urls

    async def download_image(self, session: aiohttp.ClientSession, 
                           image_url: str, drug_id: str) -> Optional[str]:
        """Download drug image to local storage using async file operations."""
        if not image_url:
            return None

        try:
            image_url = urljoin(self.base_url, image_url)
            async with session.get(image_url) as response:
                if response.status != 200:
                    return None

                content_type = response.headers.get('content-type', '')
                ext = 'jpg'
                if 'png' in content_type:
                    ext = 'png'
                elif 'jpeg' in content_type or 'jpg' in content_type:
                    ext = 'jpg'

                image_path = os.path.join(self.output_path, "images", f"{drug_id}.{ext}")
                
                # Use async file operations for better performance
                async with aiofiles.open(image_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
                        
                return image_path
        except Exception as e:
            print(f"Error downloading image for drug {drug_id}: {str(e)}")
            return None

    async def extract_drug_details(self, session: aiohttp.ClientSession, 
                                 url: str) -> Optional[DrugDetails]:
        """Extract drug details from a single drug page."""
        try:
            # Get the page content
            soup = await self.get_page_content(session, url)
            if not soup:
                return None

            # Initialize all data structures
            drug_id = url.split('/')[-1]
            combinations = []
            commercial_drugs = []
            generic_persian_name = None
            generic_english_name = None
            generic_manufacturer = None
            image_url = None
            image_path = None

            # Extract names for generic drug
            persian_title = soup.select_one('h1')
            english_title = soup.select_one('.EnglishTopLabel')
            
            if persian_title:
                generic_persian_name = persian_title.text.strip()
            if english_title:
                generic_english_name = english_title.text.strip()

            # Extract manufacturer for generic drug
            for selector in MANUFACTURER_SELECTORS:
                manufacturer_element = soup.select_one(selector)
                if manufacturer_element:
                    generic_manufacturer = manufacturer_element.text.strip()
                    break

            # Extract image
            for selector in IMAGE_SELECTORS:
                image_element = soup.select_one(selector)
                if image_element and 'src' in image_element.attrs:
                    image_url = image_element['src']
                    if image_url:
                        image_path = await self.download_image(session, image_url, drug_id)
                    break

            # Extract combinations
            for selector in INGREDIENTS_SELECTORS:
                ingredients_section = soup.select_one(selector)
                if ingredients_section:
                    ingredient_items = ingredients_section.select('li')
                    if ingredient_items:
                        combinations.extend(item.text.strip() for item in ingredient_items)
                    else:
                        combinations.append(ingredients_section.text.strip())
                    break

            # Extract commercial drugs
            seen_urls = set()
            commercial_links = soup.select('.text-xs-center .ahref_Generic')
            
            if commercial_links:
                for link in commercial_links:
                    comm_url = urljoin(self.base_url, link.get('href'))
                    if comm_url in seen_urls:
                        continue
                    
                    seen_urls.add(comm_url)
                    comm_soup = await self.get_page_content(session, comm_url)
                    
                    if comm_soup:
                        persian_name_element = comm_soup.select_one('#h1PersianName')
                        english_name_element = comm_soup.select_one('#h2EnglishName')
                        manufacturer_element = comm_soup.select_one('.ahref_GenericWhiteColor')
                        
                        comm_persian_name = persian_name_element.text.strip() if persian_name_element else ""
                        comm_english_name = english_name_element.text.strip() if english_name_element else ""
                        comm_manufacturer = manufacturer_element.text.strip() if manufacturer_element else ""
                        
                        # Use DuckDuckGo search to find multiple images for the commercial drug
                        image_search_results = ddgs_images.images(f"{comm_english_name} {comm_manufacturer}", max_results=5)
                        comm_image_urls = [result['image'] for result in image_search_results] if image_search_results else []

                        commercial_drugs.append(CommercialDrug(
                            persian_name=comm_persian_name,
                            english_name=comm_english_name,
                            manufacturer=comm_manufacturer,
                            url=comm_url,
                            image_url=comm_image_urls  # Store multiple image URLs
                        ))
                        
                    await asyncio.sleep(REQUEST_DELAY)

            # Create and return drug details
            return DrugDetails(
                id=drug_id,
                english_name=generic_english_name,
                persian_name=generic_persian_name,
                manufacturer=generic_manufacturer or "",
                image_path=image_path,
                combinations=combinations,
                url=url,
                commercial_drugs=commercial_drugs
            )

        except Exception as e:
            print(f"Error processing drug {url}: {str(e)}")
            return None

    async def _save_state(self, urls: List[str], processed_urls: Set[str]) -> None:
        """Save crawling state to allow for resuming later"""
        state_file = Path(self.output_path) / "crawl_state.json"
        state = {
            "all_urls": urls,
            "processed_urls": list(processed_urls)
        }
        
        async with aiofiles.open(state_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(state, ensure_ascii=False, indent=2))
            
    async def _load_state(self) -> Tuple[List[str], Set[str]]:
        """Load previous crawling state if available"""
        state_file = Path(self.output_path) / "crawl_state.json"
        
        if state_file.exists():
            try:
                async with aiofiles.open(state_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    state = json.loads(content)
                    return state.get("all_urls", []), set(state.get("processed_urls", []))
            except Exception as e:
                print(f"Error loading state: {str(e)}")
                
        return [], set()    
    async def _load_urls_from_file(self) -> List[str]:
        """Load URLs from all_drug_urls.json file if it exists"""
        try:
            all_drug_urls_file = Path("all_drug_urls.json")
            if all_drug_urls_file.exists():
                async with aiofiles.open(all_drug_urls_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    urls = json.loads(content)
                    print(f"📋 Loaded {len(urls)} URLs from all_drug_urls.json")
                    return urls
        except Exception as e:
            print(f"⚠️ Error loading URLs from file: {str(e)}")
        return []

    async def run(self, max_groups: Optional[int] = None, 
                 max_pages: int = 5,
                 max_drugs: Optional[int] = None) -> List[DrugDetails]:
        """Main method to run the crawler."""
        print("\n🔍 Starting Drug Data Crawler...")
        async with await self._create_session() as session:
            # Load previous state if available
            all_drug_urls, processed_urls = await self._load_state()
            
            # Load URLs from all_drug_urls.json file if available
            file_urls = await self._load_urls_from_file()
            if file_urls:
                # Add URLs from file that aren't already in all_drug_urls
                original_count = len(all_drug_urls)
                for url in file_urls:
                    if url not in all_drug_urls:
                        all_drug_urls.append(url)
                print(f"➕ Added {len(all_drug_urls) - original_count} new URLs from all_drug_urls.json")
            
            # Track partial results for incremental saving
            partial_results = []
            last_save_time = time.time()
            save_interval = 300  # Save every 5 minutes

            # Only fetch new URLs if we don't have any from state or file
            if not all_drug_urls:
                # Get all drug URLs
                print("\n📚 Fetching drug groups...")
                group_urls = await self.get_drug_groups(session)
                if not group_urls:
                    print("❌ No drug groups found!")
                    return []

                if max_groups and max_groups < len(group_urls):
                    group_urls = group_urls[:max_groups]

                print(f"\n🔎 Found {len(group_urls)} drug groups")
                
                # Add progress bar for group processing
                with tqdm(total=len(group_urls), desc="Processing drug groups", 
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} groups') as pbar:
                    for group_url in group_urls:
                        drug_urls = await self.get_drug_urls_from_group(session, group_url, max_pages)
                        all_drug_urls.extend(url for url in drug_urls if url not in all_drug_urls)
                        await asyncio.sleep(REQUEST_DELAY)
                        pbar.update(1)

                if max_drugs and max_drugs < len(all_drug_urls):
                    all_drug_urls = all_drug_urls[:max_drugs]

            # Create a rate limiter to avoid overwhelming the server
            rate_limiter = AsyncLimiter(self.max_concurrent, 1)  # max_concurrent requests per second
            
            print(f"\n💊 Processing {len(all_drug_urls)} unique drugs...")
            print(f"👉 Will skip {len(processed_urls)} already processed URLs")
            
            # Create a queue for processing URLs
            queue = asyncio.Queue()
            for url in all_drug_urls:
                if url not in processed_urls:
                    await queue.put(url)
            
            # Initialize progress bar
            total_to_process = queue.qsize()
            pbar = tqdm(total=total_to_process, desc="Extracting drug details",
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} drugs')
              # Worker function for processing URLs from the queue
            async def worker(worker_id: int):
                nonlocal last_save_time
                while not queue.empty():
                    try:
                        url = await queue.get()
                        async with rate_limiter:
                            result = await self.extract_drug_details(session, url)
                            if result:
                                partial_results.append(result)
                                
                            # Mark as processed regardless of success
                            processed_urls.add(url)
                            
                            # Save state periodically
                            current_time = time.time()
                            if current_time - last_save_time > save_interval:
                                await self._save_state(all_drug_urls, processed_urls)
                                last_save_time = current_time
                                
                                # Also save partial results
                                if partial_results:
                                    from .data_handlers import get_data_handler
                                    handler = get_data_handler(self.output_format, self.output_path)
                                    handler.save(partial_results)
                                    print(f"\n💾 Saved {len(partial_results)} drugs (incremental)")
                            
                            pbar.update(1)
                            queue.task_done()
                    except Exception as e:
                        print(f"Worker {worker_id} error: {str(e)}")
                        queue.task_done()

            # Create worker tasks
            workers = [asyncio.create_task(worker(i)) for i in range(self.max_concurrent)]
            
            # Wait for all tasks to complete
            await queue.join()
            
            # Cancel worker tasks
            for task in workers:
                task.cancel()
                
            # Wait for all workers to be cancelled
            await asyncio.gather(*workers, return_exceptions=True)
            pbar.close()

            # Ensure final state is saved
            await self._save_state(all_drug_urls, processed_urls)

            # Save results with visual feedback
            print("\n💾 Saving drug data...")
            from .data_handlers import get_data_handler
            handler = get_data_handler(self.output_format, self.output_path)
            handler.save(partial_results)

            print(f"\n✅ Successfully processed {len(partial_results)} drugs!")
            print(f"📁 Data saved to: {self.output_path}")

            return partial_results
