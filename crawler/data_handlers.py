import os
import json
import csv
from typing import List, Dict, Optional
from dataclasses import asdict
import logging
from .core import DrugDetails

class DataHandler:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.logger = logging.getLogger("DrugCrawler")

    def save(self, drug_data: List[DrugDetails]):
        raise NotImplementedError
        
    def save_incremental(self, drug_data: List[DrugDetails], merge_with_existing: bool = True):
        """Save data incrementally, merging with existing data if needed"""
        if merge_with_existing:
            existing_data = self.load_existing()
            # Create a dictionary of existing drugs by ID for easy lookup
            existing_dict = {drug.id: drug for drug in existing_data if hasattr(drug, 'id')}
            
            # Merge new data with existing data (new data overrides existing)
            for drug in drug_data:
                existing_dict[drug.id] = drug
                
            # Convert back to list
            merged_data = list(existing_dict.values())
            self.save(merged_data)
        else:
            self.save(drug_data)
    
    def load_existing(self) -> List[DrugDetails]:
        """Load existing data if available"""
        return []

class JsonHandler(DataHandler):
    def __init__(self, output_path: str):
        super().__init__(output_path)
        self.output_file = os.path.join(self.output_path, "drug_data.json")
        # Initialize the file with an empty array
        self._initialize_file()
        self.commercial_drug_count = 0
        
    def _initialize_file(self):
        """Initialize the JSON file with an empty array"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write('[\n]')
            
    def save(self, drug_data: List[DrugDetails]):
        """Save drug data one by one to avoid holding all in memory"""
        had_items = self._has_items()
        
        # Process drugs one by one to minimize memory usage
        for drug in drug_data:
            self._save_single_drug(drug, had_items or self.commercial_drug_count > 0)
            
        self.logger.info(f"Data saved to {self.output_file} - {self.commercial_drug_count} commercial drugs")
    
    def _has_items(self) -> bool:
        """Check if the JSON file already has items (array not empty)"""
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # Check if there's more than just [] brackets
                return len(content) > 3
        except Exception:
            return False
    
    def _save_single_drug(self, drug: DrugDetails, need_comma: bool):
        """Append a single drug's data to the JSON file"""
        if not drug.commercial_drugs:
            return
            
        generic_drug_id = drug.id
        generic_english_name = drug.english_name
        generic_persian_name = drug.persian_name
        
        # Process all commercial drugs for this generic drug
        for comm_drug in drug.commercial_drugs:
            flattened_item = {
                "generic_id": generic_drug_id,
                "generic_english_name": generic_english_name,
                "generic_persian_name": generic_persian_name,
                "generic_url": drug.url,
                "persian_name": comm_drug.persian_name,
                "english_name": comm_drug.english_name,
                "manufacturer": comm_drug.manufacturer,
                "url": comm_drug.url,
                "image_url": comm_drug.image_url
            }
            
            # Append to file by manipulating JSON directly
            with open(self.output_file, 'r+', encoding='utf-8') as f:
                # Move to the position before the closing bracket
                f.seek(0, os.SEEK_END)
                position = f.tell()
                # Move back to overwrite the closing bracket
                f.seek(position - 2, os.SEEK_SET)
                
                # Write comma if needed
                if need_comma:
                    f.write(',\n')
                else:
                    f.write('\n')
                    
                # Write the item and close the array
                json_item = json.dumps(flattened_item, ensure_ascii=False, indent=2)
                f.write(json_item + '\n]')
                
            self.commercial_drug_count += 1
            need_comma = True
    
    def load_existing(self) -> List[DrugDetails]:
        """Load existing drug data from JSON file"""
        output_file = os.path.join(self.output_path, "drug_data.json")
        
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert raw dictionaries back to DrugDetails objects
                    return [DrugDetails(**item) for item in data]
            except Exception as e:
                self.logger.error(f"Error loading existing data: {str(e)}")
        
        return []

class CsvHandler(DataHandler):
    def __init__(self, output_path: str):
        super().__init__(output_path)
        self.output_file = os.path.join(self.output_path, "drug_data.csv")
        self.fieldnames = [
            "generic_id", "generic_english_name", "generic_persian_name", "generic_url",
            "persian_name", "english_name", "manufacturer", "url", "image_url"
        ]
        self.file_initialized = False
        self.commercial_drug_count = 0

    def _initialize_file(self):
        """Initialize the CSV file with headers"""
        with open(self.output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
        self.file_initialized = True

    def save(self, drug_data: List[DrugDetails]):
        """Save drug data one by one to avoid holding all in memory"""
        # Initialize file if needed
        if not self.file_initialized:
            self._initialize_file()
        
        # Process each drug and commercial drug one at a time
        for drug in drug_data:
            self._save_single_drug(drug)
            
        self.logger.info(f"Data saved to {self.output_file} - {self.commercial_drug_count} commercial drugs")

    def _save_single_drug(self, drug: DrugDetails):
        """Append a single drug's commercial data to the CSV file"""
        if not drug.commercial_drugs:
            return
            
        generic_drug_id = drug.id
        generic_english_name = drug.english_name
        generic_persian_name = drug.persian_name
        
        # Open file in append mode to add rows
        with open(self.output_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            
            # Process all commercial drugs for this generic drug
            for comm_drug in drug.commercial_drugs:
                # Create flattened item
                flattened_item = {
                    "generic_id": generic_drug_id,
                    "generic_english_name": generic_english_name,
                    "generic_persian_name": generic_persian_name,
                    "generic_url": drug.url,
                    "persian_name": comm_drug.persian_name,
                    "english_name": comm_drug.english_name,
                    "manufacturer": comm_drug.manufacturer,
                    "url": comm_drug.url
                }
                
                # Handle image_url (convert list to string if needed)
                if comm_drug.image_url:
                    if isinstance(comm_drug.image_url, list):
                        flattened_item["image_url"] = '; '.join(comm_drug.image_url)
                    else:
                        flattened_item["image_url"] = comm_drug.image_url
                else:
                    flattened_item["image_url"] = ""
                    
                # Write single row
                writer.writerow(flattened_item)
                self.commercial_drug_count += 1
        
    def load_existing(self) -> List[DrugDetails]:
        """Load existing drug data from CSV files"""
        output_file = os.path.join(self.output_path, "drug_data.csv")
        commercial_file = os.path.join(self.output_path, "commercial_drugs.csv")
        
        if not os.path.exists(output_file):
            return []
            
        try:
            # Read main drug data
            drugs = {}
            with open(output_file, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    drug_id = row.get('id')
                    if drug_id:
                        # Convert combinations back to list
                        if 'combinations' in row and row['combinations']:
                            row['combinations'] = row['combinations'].split('; ')
                        else:
                            row['combinations'] = []
                        
                        # Initialize empty commercial_drugs list
                        row['commercial_drugs'] = []
                        drugs[drug_id] = row
            
            # Read commercial drugs if available
            if os.path.exists(commercial_file):
                with open(commercial_file, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        parent_id = row.get('parent_drug_id')
                        if parent_id and parent_id in drugs:
                            # Create CommercialDrug object and add to parent drug
                            from .core import CommercialDrug
                            comm_drug = CommercialDrug(
                                persian_name=row.get('persian_name', ''),
                                english_name=row.get('english_name', ''),
                                manufacturer=row.get('manufacturer', ''),
                                url=row.get('url', ''),
                                image_url=row.get('image_url')
                            )
                            drugs[parent_id]['commercial_drugs'].append(comm_drug)
            
            # Convert dict items back to DrugDetails objects
            from .core import DrugDetails
            return [DrugDetails(**drug_dict) for drug_dict in drugs.values()]
        except Exception as e:
            self.logger.error(f"Error loading existing CSV data: {str(e)}")
            return []

class DuckDbHandler(DataHandler):
    def __init__(self, output_path: str):
        super().__init__(output_path)
        self.output_file = os.path.join(self.output_path, "drug_data.duckdb")
        self.table_initialized = False
        self.commercial_drug_count = 0

    def _initialize_table(self):
        """Initialize the DuckDB table"""
        try:
            import duckdb
            con = duckdb.connect(self.output_file)
            
            # Drop the existing table if it exists to ensure clean schema
            con.execute("DROP TABLE IF EXISTS commercial_drugs")
            
            # Create new table for flattened commercial drugs data
            con.execute("""
                CREATE TABLE commercial_drugs (
                    generic_id VARCHAR,
                    generic_english_name VARCHAR,
                    generic_persian_name VARCHAR,
                    generic_url VARCHAR,
                    persian_name VARCHAR,
                    english_name VARCHAR,
                    manufacturer VARCHAR,
                    url VARCHAR,
                    image_url VARCHAR
                )
            """)
            
            con.close()
            self.table_initialized = True
        except ImportError:
            self.logger.error("DuckDB is not installed. Will fall back to JSON format")
            return False
        except Exception as e:
            self.logger.error(f"Error initializing DuckDB table: {str(e)}")
            return False
        return True
            
    def save(self, drug_data: List[DrugDetails]):
        try:
            import duckdb
            
            # Initialize table if needed
            if not self.table_initialized and not self._initialize_table():
                # Fall back to JSON if table initialization fails
                JsonHandler(self.output_path).save(drug_data)
                return
                
            # Process each drug one at a time
            con = duckdb.connect(self.output_file)
            
            for drug in drug_data:
                self._save_single_drug(con, drug)
            
            # Close connection
            con.close()
            self.logger.info(f"Data saved to {self.output_file} - {self.commercial_drug_count} commercial drugs")

        except ImportError:
            self.logger.error("DuckDB is not installed. Falling back to JSON format")
            JsonHandler(self.output_path).save(drug_data)

    def _save_single_drug(self, con, drug: DrugDetails):
        """Insert a single drug's commercial data into the DuckDB database"""
        if not drug.commercial_drugs:
            return
            
        generic_drug_id = drug.id
        generic_english_name = drug.english_name
        generic_persian_name = drug.persian_name
        
        # Process each commercial drug separately
        for comm_drug in drug.commercial_drugs:
            # Convert image_url to JSON string if it's a list
            image_url_json = json.dumps(comm_drug.image_url) if comm_drug.image_url else None
            
            # Insert a single row
            con.execute("""
                INSERT INTO commercial_drugs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                generic_drug_id,
                generic_english_name,
                generic_persian_name,
                drug.url,
                comm_drug.persian_name,
                comm_drug.english_name,
                comm_drug.manufacturer,
                comm_drug.url,
                image_url_json
            ])
            
            self.commercial_drug_count += 1
            
    def load_existing(self) -> List[DrugDetails]:
        """Load existing drug data from DuckDB database"""
        try:
            import duckdb
            import pandas as pd
            
            output_file = os.path.join(self.output_path, "drug_data.duckdb")
            
            if not os.path.exists(output_file):
                return []
                
            con = duckdb.connect(output_file)
            
            # Check if tables exist
            tables = con.execute("SHOW TABLES").fetchall()
            table_names = [t[0] for t in tables]
            
            if 'drugs' not in table_names:
                con.close()
                return []
                
            # Load drugs
            drugs_df = con.execute("SELECT * FROM drugs").df()
            
            # Load commercial drugs if available
            commercial_drugs_df = None
            if 'commercial_drugs' in table_names:
                commercial_drugs_df = con.execute("SELECT * FROM commercial_drugs").df()
            
            con.close()
            
            # Convert dataframes to DrugDetails objects
            drugs = []
            for _, row in drugs_df.iterrows():
                # Convert combinations back to list
                combinations = []
                if row['combinations']:
                    combinations = row['combinations'].split('; ')
                
                # Get commercial drugs for this drug
                commercial_drugs = []
                if commercial_drugs_df is not None and not commercial_drugs_df.empty:
                    drug_commercial_df = commercial_drugs_df[commercial_drugs_df['parent_drug_id'] == row['id']]
                    
                    for _, comm_row in drug_commercial_df.iterrows():
                        from .core import CommercialDrug
                        commercial_drugs.append(CommercialDrug(
                            persian_name=comm_row['persian_name'],
                            english_name=comm_row['english_name'],
                            manufacturer=comm_row['manufacturer'],
                            url=comm_row['url'],
                            image_url=None  # Image URLs are not stored in DB
                        ))
                
                # Create DrugDetails object
                from .core import DrugDetails
                drugs.append(DrugDetails(
                    id=row['id'],
                    english_name=row['english_name'],
                    persian_name=row['persian_name'],
                    manufacturer=row['manufacturer'],
                    image_path=row['image_path'],
                    combinations=combinations,
                    url=row['url'],
                    commercial_drugs=commercial_drugs
                ))
            
            return drugs
            
        except Exception as e:
            self.logger.error(f"Error loading existing DuckDB data: {str(e)}")
            return []

def get_data_handler(output_format: str, output_path: str) -> DataHandler:
    handlers = {
        'json': JsonHandler,
        'csv': CsvHandler,
        'duckdb': DuckDbHandler
    }
    
    handler_class = handlers.get(output_format.lower(), JsonHandler)
    return handler_class(output_path)
