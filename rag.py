from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import requests
import pymupdf
import io 
import re
import nltk
nltk.download('punkt_tab', download_dir='./punkt')  # Ensure that NLTK's sentence tokenizer is available
from nltk.tokenize import sent_tokenize
nltk.data.path.append('./punkt')
import psycopg2
import hashlib
import os
import faiss
import numpy as np
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from openai import OpenAI
from collections import defaultdict

# Config Start ------------------------------------------------------------------------------------------------------------------------------

url = "https://www.who.int/europe/publications/i"

load = "no" # change to yes, if you want to load data 

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)
pages = 1

# Config End -------------------------------------------------------------------------------------------------------------------------------

def fetch_links(start: int, end: int, url: str, pause: int) -> set:
    """
    Fetch at first all links of relevant documents.
    Start is always page 1, this cannot change
    """
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    
    download_links = set()

    for page_number in range(start, end + 1):
        try:
            input_field = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'input.k-textbox'))
            )

            input_field.clear()
            input_field.send_keys(str(page_number))

            next_page_button = driver.find_element(By.CSS_SELECTOR, 'a[aria-label="Go to the next page"]')
            next_page_button.click()

            time.sleep(pause)

            links = driver.find_elements(By.XPATH, "//a[contains(@href, '.pdf')]")
            for link in links:
                download_links.add(link.get_attribute("href"))
                print(f"Loaded page {page_number}... {len(download_links)} in set.")

        except Exception as e:
            print(f"An error occurred on page {page_number}: {e}")
            break

    driver.quit()
    return download_links


def fetch_pdf(link: set) -> str:
    request = requests.get(link)
    filestream = io.BytesIO(request.content)
    with pymupdf.open(stream=filestream, filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text
    

def split_large_paragraph(paragraph: str, max_words: int, overlap: int) -> str:
        sentences = sent_tokenize(paragraph)
        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            words_in_sentence = sentence.split()
            if current_word_count + len(words_in_sentence) > max_words:
                # add the current chunk and create a new one
                chunks.append(' '.join(current_chunk))
                current_chunk = current_chunk[-overlap:]  # Keep overlap from previous chunk
                current_word_count = len(current_chunk)

            current_chunk.extend(words_in_sentence)
            current_word_count += len(words_in_sentence)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks


def split_into_chunks(text: str, max_words=512, overlap=50) -> list:
    # Regular expression patterns for headings and bullet points
    heading_pattern = r'^(?P<heading>[A-Z].+):$'
    bullet_point_pattern = r'^\s*[-•]\s*(?P<bullet>.+)'    

    # Splitting the text into lines
    lines = text.split('\n')

    chunks = []
    current_chunk = []
    current_section = None

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Check if the line is a heading
        heading_match = re.match(heading_pattern, line)
        if heading_match:
            # If there's an ongoing chunk, save it
            if current_chunk:
                paragraph = ' '.join(current_chunk)
                for sub_chunk in split_large_paragraph(paragraph, max_words, overlap):
                    chunks.append({
                        'section': current_section,
                        'type': 'paragraph',
                        'content': sub_chunk
                    })
                current_chunk = []
            
            # New heading detected
            current_section = heading_match.group('heading')
            chunks.append({
                'section': current_section,
                'type': 'heading',
                'content': line
            })
            continue
        
        # Check if the line is a bullet point
        bullet_match = re.match(bullet_point_pattern, line)
        if bullet_match:
            # If there's an ongoing chunk, save it
            if current_chunk:
                paragraph = ' '.join(current_chunk)
                for sub_chunk in split_large_paragraph(paragraph, max_words, overlap):
                    chunks.append({
                        'section': current_section,
                        'type': 'paragraph',
                        'content': sub_chunk
                    })
                current_chunk = []

            # Bullet point detected
            chunks.append({
                'section': current_section,
                'type': 'bullet',
                'content': bullet_match.group('bullet')
            })
            continue

        # Accumulate lines for paragraphs
        current_chunk.append(line)

    # Add any leftover chunk
    if current_chunk:
        paragraph = ' '.join(current_chunk)
        for sub_chunk in split_large_paragraph(paragraph, max_words, overlap):
            chunks.append({
                'section': current_section,
                'type': 'paragraph',
                'content': sub_chunk
            })

    return chunks


def create_vector(chunks: list) -> list:
    """
    Create vectors from text
    """
    response = client.embeddings.create(
        input=chunks,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def hash_text(chunks: list) -> list:
    """
    Hash chunk
    """
    return hashlib.sha256(chunks.encode('utf-8')).hexdigest()


def ingest_to_database(chunks: list, group_id: int) -> None:
    """
    Connect to Postgres and create a table.
    Afterwards ingest data and vectors
    """
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        database="llm",
        user="admin",
        password="admin"
    )
    cur = conn.cursor()

    # Create a table if it doesn't exist
    cur.execute('''
    CREATE TABLE IF NOT EXISTS text_chunks (
        id SERIAL PRIMARY KEY,
        section_content TEXT,
        type TEXT,
        vector VECTOR(1536),  -- Adjust this based on your OpenAI model
        hash TEXT UNIQUE,
        group_id INT,
        timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
    ''')
    conn.commit()

    for chunk in chunks:
        section_content = (chunk['section'] or "") + " " + chunk['content']
        chunk_type = chunk['type']
        vector = create_vector([section_content])
        hash_value = hash_text(section_content)

        cur.execute('''
            INSERT INTO text_chunks (section_content, type, vector, hash, group_id)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (hash) DO NOTHING;
        ''', (section_content, chunk_type, vector, hash_value, group_id))

        conn.commit()
    cur.close()
    conn.close()


if load == "yes":
    links = fetch_links(1,pages,url,5)
    number = 0
    for link in links:
        fetch_file = fetch_pdf(link)
        number += 1
        result = split_into_chunks(fetch_file)
        ingest_to_database(result, number)
        print("Data ingested successfully! Loaded link number " + str(number))


def generate_query_embedding(query):
    """
    Generate embedding for the user's query
    """
    response = client.embeddings.create(
        input=[query],
        model="text-embedding-3-small")
    return response.data[0].embedding

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

def search_similar_chunks_pg(query_vector, top_n=5):
    # Connect to PostgreSQL
    conn = psycopg2.connect(
    host="postgres",
    port=5432,
    database="llm",
    user="admin",
    password="admin"
    )
    cur = conn.cursor()

    # Fetch vectors and their corresponding text chunks from the database
    cur.execute('SELECT id, section_content, vector, group_id FROM text_chunks')
    rows = cur.fetchall()

    query_vector = normalize_vector(query_vector)
    
    # Calculate cosine similarity between query vector and each chunk vector
    similarities = []
    for row in rows:
        chunk_id, section_content, vector_str, group_id = row
        # Convert string representation of vector to numpy array
        chunk_vector = np.array([float(x) for x in vector_str[1:-1].split(',')])
        similarity = 1 - cosine(query_vector, chunk_vector)
        similarities.append((chunk_id, section_content, similarity, group_id))

    # Sort by similarity and get top N results
    similarities = sorted(similarities, key=lambda x: x[2], reverse=True)[:top_n]

    cur.close()
    conn.close()

    return similarities

def search_similar_chunks_faiss(query_vector, top_n=5):
    """ Euclidean distance search using FAISS """
    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        database="llm",
        user="admin",
        password="admin"
    )
    cur = conn.cursor()
    cur.execute('SELECT id, section_content, vector, group_id FROM text_chunks')
    rows = cur.fetchall()

    # Prepare FAISS index
    d = len(query_vector)  # Dimension of the embeddings
    index = faiss.IndexFlatL2(d)  # Using Euclidean (L2) distance

    vectors = []
    ids = []
    contents = []
    group = []

    for row in rows:
        chunk_id, section_content, vector_str, group_id = row
        
        # Assuming vector_str is a string representation of a list of floats, e.g., "[0.1, 0.2, ...]"
        try:
            # Safely convert the string representation of the vector to a NumPy array
            chunk_vector = np.array(eval(vector_str), dtype=np.float32)  # Use eval carefully; safer options exist.
        except Exception as e:
            print(f"Error converting vector for ID {chunk_id}: {e}")
            continue
        
        # Check if chunk_vector is a valid numpy array
        if chunk_vector.ndim != 1 or chunk_vector.shape[0] != d:
            print(f"Invalid vector for ID {chunk_id}: {chunk_vector}")
            continue  # Skip if the vector is not valid

        vectors.append(chunk_vector)
        ids.append(chunk_id)
        contents.append(section_content)
        group.append(group_id)

    # Convert list to numpy array and check if it's valid
    if vectors:  # Check if vectors is not empty
        vectors_np = np.vstack(vectors).astype(np.float32)

        # Check if vectors_np is indeed a NumPy array
        if not isinstance(vectors_np, np.ndarray):
            print(f"Error: vectors_np is not a numpy array. Type: {type(vectors_np)}")
            return []
       
        # Add vectors to FAISS index
        index.add(vectors_np)

        # Convert query_vector to NumPy array
        query_vector_np = np.array(query_vector, dtype=np.float32).reshape(1, -1)

        # Search the query in FAISS
        distances, indices = index.search(query_vector_np, top_n)

        faiss_results = [(ids[i], contents[i], distances[0][n], group[i]) for n, i in enumerate(indices[0])]
    else:
        faiss_results = []

    cur.close()
    conn.close()

    return faiss_results


def search_similar_chunks_faiss_dot(query_vector, top_n=5):
    """ Dot-product similarity search using FAISS """
    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        database="llm",
        user="admin",
        password="admin"
    )
    cur = conn.cursor()
    cur.execute('SELECT id, section_content, vector, group_id FROM text_chunks')
    rows = cur.fetchall()

    # Prepare FAISS index
    d = len(query_vector)  # Dimension of the embeddings
    index = faiss.IndexFlatIP(d)  # Using dot-product similarity

    vectors = []
    ids = []
    contents = []
    group = []
    for row in rows:
        chunk_id, section_content, vector_str, group_id = row
        chunk_vector = np.array([float(x) for x in vector_str[1:-1].split(',')], dtype=np.float32)
        vectors.append(chunk_vector)
        ids.append(chunk_id)
        contents.append(section_content)
        group.append(group_id)

    # Convert list to numpy array and add to FAISS
    vectors_np = np.vstack(vectors).astype(np.float32)
    index.add(vectors_np)

    # Search the query in FAISS
    query_vector_np = np.array(query_vector, dtype=np.float32).reshape(1, -1)
    similarities, indices = index.search(query_vector_np, top_n)

    faiss_dot_results = [(ids[i], contents[i], similarities[0][n], group[i]) for n, i in enumerate(indices[0])]

    cur.close()
    conn.close()

    return faiss_dot_results

def canberra_distance(vec1, vec2):
    return np.sum(np.abs(vec1 - vec2) / (np.abs(vec1) + np.abs(vec2) + 1e-10))  # Avoid division by zero

def search_similar_chunks_canberra(query_vector, top_n=5):
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        database="llm",
        user="admin",
        password="admin"
    )
    cur = conn.cursor()

    # Fetch vectors and their corresponding text chunks from the database
    cur.execute('SELECT id, section_content, vector, group_id FROM text_chunks')
    rows = cur.fetchall()

    # Calculate Canberra distance between query vector and each chunk vector
    similarities = []
    for row in rows:
        chunk_id, section_content, vector_str, group_id = row
        chunk_vector = np.array([float(x) for x in vector_str[1:-1].split(',')])
        distance = canberra_distance(query_vector, chunk_vector)
        similarities.append((chunk_id, section_content, distance, group_id))

    # Sort by distance (lower is better) and get top N results
    similarities = sorted(similarities, key=lambda x: x[2])[:top_n]

    cur.close()
    conn.close()

    return similarities



def display_results(results, method_name):
    print(f"\nTop Results ({method_name}):")
    for i, (chunk_id, section_content, similarity, group_id) in enumerate(results, 1):
        print(f"{i}. ID: {chunk_id} | Similarity: {similarity:.4f} | Group_id: {group_id}\nText: {section_content}\n")


def compare_top_ids_simple(pg_results, faiss_l2_results, faiss_dot_results, canberra_distance_results, top_n=5):
    # Initialize a dictionary to store the total score for each ID
    id_scores = defaultdict(int)
    id_group_mapping = {}
    
    # Function to assign scores based on rank
    def assign_scores(results, method_name):
        for rank, result in enumerate(results[:top_n], start=1):
            chunk_id = result[0]
            group_id = result[3]
            score = top_n - rank + 1  # Higher rank gives higher score (e.g., rank 1 gets top_n points, rank 2 gets top_n-1, etc.)
            id_scores[chunk_id] += score
            id_group_mapping[chunk_id] = group_id
    
    # Assign scores from all four methods
    assign_scores(pg_results, 'pg_cosine')
    assign_scores(faiss_l2_results, 'faiss_l2')
    assign_scores(faiss_dot_results, 'faiss_dot')
    assign_scores(canberra_distance_results, 'canberra')

    # Sort the IDs by their total score, higher score is better
    sorted_ids = sorted(id_scores.items(), key=lambda x: -x[1])
    
    # Extract the top 3 IDs based on the highest scores
    top_ids = sorted_ids[:3]
    final_output = [(chunk_id, id_group_mapping[chunk_id], score) for chunk_id, score in top_ids]
    return final_output


def get_chunk_by_id_from_db(chunk_id):
    """Fetch the chunk text directly from the database using the best chunk ID."""
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        database="llm",
        user="admin",
        password="admin"
    )
    cur = conn.cursor()

    # Query the database for the chunk by its ID
    cur.execute("SELECT section_content FROM text_chunks WHERE id = %s", (chunk_id,))
    result = cur.fetchone()

    cur.close()
    conn.close()

    if result:
        return result[0]  # Return the text content of the chunk
    else:
        return None  # In case the chunk ID is not found

while True:
    query = input("What do you want to know? (type 'exit' to leave): ")

    if query.lower() == 'exit':
        print("Exiting the program.")
        break

    query_vector = generate_query_embedding(query)

    pg_results = search_similar_chunks_pg(query_vector, top_n=5)
    #display_results(pg_results, "Cosine Similarity (Postgres)")
    faiss_l2_results = search_similar_chunks_faiss(query_vector, top_n=5)
    #display_results(faiss_l2_results, "Euclidean Distance (FAISS)")
    faiss_dot_results = search_similar_chunks_faiss_dot(query_vector, top_n=5)
    #display_results(faiss_dot_results, "Dot Product (FAISS)")
    canberra_distance_results = search_similar_chunks_canberra(query_vector, top_n=5)
    #display_results(canberra_distance_results, "Canberra Distance (Canberra)")

    top_ids = compare_top_ids_simple(pg_results, faiss_l2_results, faiss_dot_results, canberra_distance_results)

    for rank, (chunk_id, group_id, score) in enumerate(top_ids, 1):
        print(f"Rank {rank}: Chunk ID {chunk_id}, Group ID {group_id}, Total Score: {score}")

    best_chunk_id = top_ids[0][0]
    best_chunk_content = get_chunk_by_id_from_db(best_chunk_id)

    if best_chunk_content:
        print(f"\nBest Answer (ID: {best_chunk_id})")
    else:
        print("Best chunk not found in the database.")

    MODEL = "gpt-4o"

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that helps me with answering my questions. For that you get additional information from a database. It's always a piece of text. Please consider this text in your answer. Give a detailed answer."},
            {"role": "system", "content": best_chunk_content},
            {"role": "user", "content": query}
        ]
    )
    print("Assistant: " + completion.choices[0].message.content)

