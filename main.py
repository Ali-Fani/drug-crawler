import argparse
import asyncio
from crawler.core import DrugDataCrawler

def print_banner():
    banner = """
    🏥 Drug Data Crawler 💊
    =====================
    Fetching pharmaceutical data...
    """
    print(banner)

async def async_main():
    parser = argparse.ArgumentParser(
        description="Drug data crawler for darooyab.ir",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Add a subparser for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Full crawl (original behavior)
    full_parser = subparsers.add_parser('full', help='Run complete crawling process (original behavior)')
    full_parser.add_argument(
        "--max_groups", 
        type=int, 
        help="Maximum number of drug groups to process"
    )
    full_parser.add_argument(
        "--max_pages",
        type=int,
        default=5,
        help="Maximum pages per group to process (default: 5)",
    )
    full_parser.add_argument(
        "--max_drugs", 
        type=int, 
        help="Maximum number of drugs to process"
    )
    
    # Phase 1: Link Discovery
    discover_parser = subparsers.add_parser('discover', 
        help='Phase 1: Only discover and save drug links without processing details')
    
    # Phase 2: Link Processing
    process_parser = subparsers.add_parser('process', 
        help='Phase 2: Process previously discovered drug links')
    process_parser.add_argument(
        "--batch-size", 
        type=int,
        default=50, 
        help="Number of links to process in each batch (default: 50)"
    )
    process_parser.add_argument(
        "--start-index", 
        type=int,
        default=0, 
        help="Index to start processing from (for manual resuming)"
    )
    
    # Resume previously interrupted processing
    resume_parser = subparsers.add_parser('resume', 
        help='Resume processing from the last position')
    
    # Common arguments for all commands
    for subparser in [full_parser, discover_parser, process_parser, resume_parser]:
        subparser.add_argument(
            "--output-format",
            choices=['json', 'csv', 'duckdb'],
            default='json',
            help="Output format (default: json)"
        )

    args = parser.parse_args()
    
    # If no command is provided, show help
    if not args.command:
        parser.print_help()
        return

    print_banner()

    try:
        crawler = DrugDataCrawler(output_format=args.output_format)
        
        if args.command == 'full':
            print("🔧 Configuration:")
            print(f"  • Max Groups: {args.max_groups if args.max_groups else 'All'}")
            print(f"  • Max Pages per Group: {args.max_pages}")
            print(f"  • Max Drugs: {args.max_drugs if args.max_drugs else 'All'}")
            print(f"  • Output Format: {args.output_format}")
            print()
            
            await crawler.run(
                max_groups=args.max_groups, 
                max_pages=args.max_pages, 
                max_drugs=args.max_drugs
            )
            
        elif args.command == 'discover':
            print(f"🔍 Running Phase 1: Drug Link Discovery")
            print(f"  • Output Format: {args.output_format}")
            print()
            
            await crawler.discover_all_drug_links()
            
        elif args.command == 'process':
            print(f"🔄 Running Phase 2: Processing Drug Details")
            print(f"  • Batch Size: {args.batch_size}")
            print(f"  • Starting Index: {args.start_index}")
            print(f"  • Output Format: {args.output_format}")
            print()
            
            await crawler.process_saved_links(
                batch_size=args.batch_size,
                start_index=args.start_index
            )
            
        elif args.command == 'resume':
            print(f"⏯️ Resuming processing from last position")
            print(f"  • Output Format: {args.output_format}")
            print()
            
            await crawler.resume_processing()
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Crawler stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
    finally:
        print("\n👋 Done!")

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
