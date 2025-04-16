# Configuration for selectors and other constants
GROUP_SELECTORS = [
 '.drug-group a',
 '.card a', 
 '.panel a',
 'a[href*="/drugs/groups/"]',
 'a[href*="/DrugGroups/"]'
]

DRUG_SELECTORS = [
 '.col-md-6 .ahref_Generic'
]

MANUFACTURER_SELECTORS = [
 '.drug-company',
 '.medicine-company',
 '.manufacturer',
 '.company',
 '.producer'
]

IMAGE_SELECTORS = [
 '.drug-image img',
 '.medicine-image img',
 '.product-image img',
 '.image img',
 'img.product',
 'img.drug',
 '.gallery img'
]

INGREDIENTS_SELECTORS = [
 '.drug-ingredients',
 '.medicine-ingredients',
 '.ingredients',
 '.composition',
 '.formula',
 '.active-ingredients'
]

# Request configuration - optimized for async
REQUEST_TIMEOUT = 15  # Reduced from 30 since we're using async
REQUEST_DELAY = 0.5   # Reduced from 2 seconds between requests
PAGE_DELAY = 0.3      # Reduced from 1 second between paginated pages
MAX_RETRIES = 3       # Added retry mechanism

# Output configuration
DEFAULT_OUTPUT_FORMAT = 'json'
DEFAULT_OUTPUT_PATH = 'drug_data'
