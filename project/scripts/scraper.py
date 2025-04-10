import os
import json
import argparse

from project.rag.scrappers.wiki_scrapper import WikipediaScraper
from project.rag.scrappers.pubmed_scrapper import PubMedScraper


def main():
    parser = argparse.ArgumentParser(description="Document Scraper")
    parser.add_argument("--field", 
                        choices=["Bio-Medical", "Education", "Finance", "Open-Domain", "Science", "test"],
                        required=True,
                        help="Select the topic file to process")
    parser.add_argument("--source", 
                        choices=["Wiki", "PubMed", "test"],
                        required=True,
                        help="Select the scrapping medium")
    parser.add_argument("--results-dir", 
                        default="project/data/docs",
                        help="Directory containing the results JSON files")
    parser.add_argument("--email",
                         default="leozheng265@gmail.com",
                         help="Email address needed to initialize pubmed api service")
    args = parser.parse_args()

    key_words_file = f"project/data/keyWords/{args.field}.json"


    if not os.path.exists(key_words_file):
        print(f"File not found: {key_words_file}")
        return

    with open(key_words_file, "r") as f:
        key_words = json.load(f)

    if args.source == "Wiki":
        scraper = WikipediaScraper(field=args.field ,docs_dir=args.results_dir)

    if args.source == "PubMed":
        scraper = PubMedScraper(field=args.field, docs_dir=args.results_dir, email=args.email)
    
    elif args.source == "test":
        scraper = None
    
    seen = set()

    for docs in key_words:
        key_words = docs.get("keywords", [])
        for key_word in key_words:
            if key_word in seen or not key_word:
                continue
            seen.add(key_word)
            print(f"Processing {key_word}...")
            scraper.fetch_and_save(key_word)

    
if __name__ == "__main__":
    main()
