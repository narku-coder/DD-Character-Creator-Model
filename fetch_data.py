import wikipedia
import re
import time

# 1. Define the topics you want your model to learn about. 
# Keep them broad to gather a large amount of text.
topics = [
    "Artificial intelligence", 
    "Space exploration", 
    "History of the Internet",
    "Python (programming language)", 
    "American football", 
    "Magic the Gathering", 
    "Reality Television",
    "List of fantasy novels"
]

filename = "tiny_wiki_dataset.txt"

print(f"Starting data pipeline. Saving to {filename}...\n")

# 2. Open the file in write mode with utf-8 encoding
with open(filename, "w", encoding="utf-8") as f:
    for topic in topics:
        try:
            print(f"Fetching: {topic}")
            
            # Search for the best matching page title
            search_results = wikipedia.search(topic)
            if not search_results:
                print(f"No results found for '{topic}'")
                continue
            
            page_title = search_results[0]
            # Fetch the page object
            page = wikipedia.page(page_title, auto_suggest=False)
            
            # Extract the raw text content
            content = page.content
            
            # 3. Basic Data Cleaning
            # Remove Wikipedia section headers (e.g., "== History ==")
            clean_content = re.sub(r'==+ .*? ==+', '', content)
            # Remove excessive consecutive newlines
            clean_content = re.sub(r'\n{3,}', '\n\n', clean_content)
            
            # 4. Write the cleaned text to our dataset file
            f.write(clean_content + "\n\n")
            
            print(f"Successfully added '{page_title}' ({len(clean_content)} characters)")
            
            # Be polite to Wikipedia's API by pausing briefly between requests
            time.sleep(1)
            
        except wikipedia.exceptions.DisambiguationError as e:
            # Handle cases where the search term is too broad
            print(f"Skipping '{topic}': Disambiguation page. Try being more specific.")
        except wikipedia.exceptions.PageError:
            print(f"Skipping '{topic}': Page not found.")
        except Exception as e:
            print(f"Error fetching '{topic}': {e}")

print(f"\nData collection complete! Open {filename} to view your training data.")