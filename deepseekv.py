from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import json
import csv
import duckdb
import os
from urllib.parse import urljoin
from typing import List, Dict, Optional, Union
import time
import random

class DrugCrawler:
    def __init__(self, proxy: Optional[Union[str, List[str]]] = None):
        self.base_urls = {
            'drugs_com': 'https://www.drugs.com',
            'iran_drug': 'https://www.darookhaneonline.com',
            'fda': 'https://www.accessdata.fda.gov'
        }
        
        self.browser_config = {
            'headless': False,  # Set to False for debugging
            'proxy': proxy[0] if isinstance(proxy, list) and proxy else proxy,
            'viewport': {'width': 1920, 'height': 1080},
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
        }
        
        self.output_dir = 'drug_data'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize DuckDB
        self.conn = duckdb.connect(os.path.join(self.output_dir, 'drugs.db'))
        self._create_tables()
        
        # Initialize Playwright browser
        self.playwright = None
        self.browser = None
        self.context = None
        
    def __enter__(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=self.browser_config['headless'],
            proxy={'server': self.browser_config['proxy']} if self.browser_config['proxy'] else None
        )
        self.context = self.browser.new_context(
            viewport=self.browser_config['viewport'],
            user_agent=self.browser_config['user_agent']
        )
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    def _create_tables(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS drugs (
            id INTEGER PRIMARY KEY,
            english_name VARCHAR,
            persian_name VARCHAR,
            manufacturer VARCHAR,
            description TEXT,
            image_url VARCHAR,
            combinations TEXT
        )
        """)

    def _get_page_content(self, url: str, params: Dict = None) -> Optional[str]:
        """Get page content using Playwright"""
        if not self.context:
            raise RuntimeError("Browser context not initialized. Use 'with' statement.")
            
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                page = self.context.new_page()
                
                # Set up request interception for common bot detection scripts
                page.route("**/{script.js,analytics.js,tracking.js}", lambda route: route.abort())
                
                # Add random delay and mouse movements to appear more human-like
                page.set_default_timeout(30000)
                
                # Construct final URL with parameters
                final_url = url
                if params:
                    query_string = "&".join(f"{k}={v}" for k, v in params.items())
                    final_url = f"{url}?{query_string}"
                  # Navigate to the page
                response = page.goto(final_url, wait_until="networkidle")
                if not response:
                    raise Exception("Failed to get response")
                
                if response.status == 403:
                    retry_count += 1
                    print(f"403 error, retrying ({retry_count}/{max_retries})...")
                    page.close()
                    time.sleep(random.uniform(5, 10))
                    continue                # Try to handle GDPR consent modal
                try:
                    # Common GDPR accept button selectors
                    gdpr_selectors = [
                        '#onetrust-accept-btn-handler',  # OneTrust
                        '.consent-accept',  # Generic
                        '#accept-cookies',  # Generic
                        'button[contains(text(), "Accept")]',  # Text-based
                        '[aria-label="Accept cookies"]',  # Aria-label based
                        '.gdpr-consent-button'  # Generic class
                    ]
                    
                    for selector in gdpr_selectors:                        
                        try:
                            accept_button = page.wait_for_selector(selector, timeout=5000)
                            if accept_button:
                                accept_button.click()
                                print("Clicked GDPR consent button")
                                time.sleep(1)  # Wait for modal to close
                                break
                        except Exception:
                            continue
                except Exception as e:
                    print(f"No GDPR modal found or error handling it: {e}")

                # Simulate human-like behavior
                page.mouse.move(random.randint(100, 500), random.randint(100, 500))
                page.mouse.wheel(delta_x=0, delta_y=random.randint(-300, -100))
                time.sleep(random.uniform(2, 4))
                
                # Wait for content to load
                page.wait_for_selector('body', timeout=10000)
                
                # Get the page content
                content = page.content()
                page.close()
                return content
                
            except Exception as e:
                print(f"Error fetching {url}: {str(e)}")
                retry_count += 1
                if page:
                    page.close()
                time.sleep(random.uniform(5, 10))
                
        return None
        
    def _get_soup(self, url: str, params: Dict = None) -> Optional[BeautifulSoup]:
        """Get BeautifulSoup object from URL using Playwright"""
        content = self._get_page_content(url, params)
        if content:
            return BeautifulSoup(content, 'html.parser')
        return None

    def scrape_drugs_com(self, limit: int = 100) -> List[Dict]:
        print("Scraping Drugs.com...")
        url = f"{self.base_urls['drugs_com']}/drug_information.html"
        soup = self._get_soup(url)
        if not soup:
            return []
        
        drugs = []
        letters = soup.select('ul.az-list a')
        
        for letter in letters[:min(5, len(letters))]:  # Limit letters for demo
            letter_url = urljoin(self.base_urls['drugs_com'], letter['href'])
            letter_soup = self._get_soup(letter_url)
            if not letter_soup:
                continue
                
            drug_links = letter_soup.select('ul.ddc-list-column-2 li a')
            
            for link in drug_links[:min(limit, len(drug_links))]:
                drug_url = urljoin(self.base_urls['drugs_com'], link['href'])
                drug_soup = self._get_soup(drug_url)
                if not drug_soup:
                    continue
                    
                # Extract drug info
                try:
                    name = drug_soup.find('h1').text.strip()
                    
                    # Manufacturer
                    manufacturer = "Unknown"
                    dt_elements = drug_soup.select('div.ddc-media-body dt')
                    for dt in dt_elements:
                        if 'Manufacturer' in dt.text:
                            manufacturer = dt.find_next('dd').text.strip()
                            break
                    
                    # Image
                    image_url = ""
                    img_tag = drug_soup.select_one('div.ddc-media img')
                    if img_tag and 'src' in img_tag.attrs:
                        image_url = urljoin(self.base_urls['drugs_com'], img_tag['src'])
                    
                    # Description
                    description = ""
                    desc_section = drug_soup.select_one('div.ddc-media-body')
                    if desc_section:
                        description = desc_section.text.strip()
                    
                    drug_data = {
                        'english_name': name,
                        'persian_name': '',  # Will fill from Iranian source
                        'manufacturer': manufacturer,
                        'description': description,
                        'image_url': image_url,
                        'combinations': []
                    }
                    
                    drugs.append(drug_data)
                    print(f"Collected: {name}")
                    
                    # Add delay between requests
                    time.sleep(random.uniform(2, 4))
                    
                except Exception as e:
                    print(f"Error parsing drug page {drug_url}: {e}")
        
        return drugs
    
    def scrape_iran_drugs(self, drugs_list: List[Dict]) -> List[Dict]:
        print("Enhancing with Persian names from Iranian source...")
        search_url = f"{self.base_urls['iran_drug']}/search"
        
        for drug in drugs_list[:10]:  # Limit for demo
            try:
                # Search for the drug
                params = {'s': drug['english_name']}
                search_soup = self._get_soup(search_url, params=params)
                if not search_soup:
                    continue
                    
                # Find first result
                result = search_soup.select_one('div.product-item')
                if result:
                    persian_name = result.select_one('h2 a').text.strip()
                    drug['persian_name'] = persian_name
                    
                    # Try to get more info from product page
                    product_url = urljoin(self.base_urls['iran_drug'], result.select_one('h2 a')['href'])
                    product_soup = self._get_soup(product_url)
                    if product_soup:
                        # Extract manufacturer if not already found
                        if not drug.get('manufacturer') or drug['manufacturer'] == 'Unknown':
                            manufacturer_tag = product_soup.select_one('span.manufacturer')
                            if manufacturer_tag:
                                drug['manufacturer'] = manufacturer_tag.text.strip()
                        
                        # Get image if not already found
                        if not drug.get('image_url'):
                            img_tag = product_soup.select_one('div.product-image img')
                            if img_tag and 'src' in img_tag.attrs:
                                drug['image_url'] = urljoin(self.base_urls['iran_drug'], img_tag['src'])
                    
                    print(f"Found Persian name for {drug['english_name']}: {persian_name}")
                    time.sleep(random.uniform(2, 4))
                    
            except Exception as e:
                print(f"Error searching for {drug['english_name']}: {e}")
                
        return drugs_list
    
    def save_to_json(self, data: List[Dict], filename: str = 'drugs.json'):
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON to {path}")
    
    def save_to_csv(self, data: List[Dict], filename: str = 'drugs.csv'):
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        print(f"Saved CSV to {path}")
    
    def save_to_duckdb(self, data: List[Dict]):
        # Clear existing data for fresh insert
        self.conn.execute("DELETE FROM drugs")
        
        for i, drug in enumerate(data):
            self.conn.execute("""
            INSERT INTO drugs VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                i+1,
                drug['english_name'],
                drug['persian_name'],
                drug['manufacturer'],
                drug['description'],
                drug['image_url'],
                json.dumps(drug['combinations'])
            ])
        
        print(f"Inserted {len(data)} records into DuckDB")
    
    def run(self, limit: int = 50):
        """Run the crawler with context management"""
        with self:  # This ensures proper browser cleanup
            # Step 1: Get international drug data
            drugs = self.scrape_drugs_com(limit=limit)
            
            # Step 2: Enhance with Persian names
            drugs = self.scrape_iran_drugs(drugs)
            
            # Step 3: Save in all formats
            if drugs:
                self.save_to_json(drugs)
                self.save_to_csv(drugs)
                self.save_to_duckdb(drugs)
            else:
                print("No drug data collected")

if __name__ == "__main__":
    crawler = DrugCrawler()
    crawler.run(limit=20)  # Set higher limit for more comprehensive data