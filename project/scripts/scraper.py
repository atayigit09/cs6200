import os
import json
import argparse

from rag.scrappers.wiki_scrapper import WikipediaScraper

def main():
    parser = argparse.ArgumentParser(description="Document Scraper")
    parser.add_argument("--topic", 
                        choices=["Bio-Medical", "Education", "Finance", "Open-Domain", "Science", "test"],
                        required=True,
                        help="Select the topic file to process")
    parser.add_argument("--source", 
                        choices=["Wiki", "test"],
                        required=True,
                        help="Select the scrapping medium")
    parser.add_argument("--results-dir", 
                        default="./data/docs",
                        help="Directory containing the results JSON files")
    args = parser.parse_args()

    key_words_file = f"./data/keyWords/{args.topic}.json"


    if not os.path.exists(key_words_file):
        print(f"File not found: {key_words_file}")
        return

    with open(key_words_file, "r") as f:
        key_words = json.load(f)

    if args.source == "Wiki":
        scraper = WikipediaScraper(field=args.topic ,docs_dir=args.results_dir)
    
    elif args.source == "test":
        scraper = None
    

    seen = set()

    for docs in key_words:
        key_words = docs.get("keywords", [])
        for key_word in key_words:
            if key_word in seen:
                continue
            seen.add(key_word)
            print(f"Processing {key_word}...")
            scraper.fetch_and_save(key_word)

    
    

if __name__ == "__main__":
    main()
