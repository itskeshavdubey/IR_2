### Code Explanation for Vector Space Model (VSM) - Ranked Retrieval


#### **Course Information**:
- **Course**: Information Retrieval (CSD358)
- **Assignment**: Assignment 2 - Vector Space Model (VSM)
- **Group Members**:
  - Keshav Dubey - Roll No: 2110110687
  - Utkarsh Tiwari - Roll No: 2110110xxx
- **Submission Date**: September 24, 2024

---

### **Overview**:
This Python code implements a **Vector Space Model (VSM)** for ranked retrieval using cosine similarity and tf-idf scoring. It follows the **lnc.ltc** ranking scheme to rank documents based on free-text queries.

[Watch the Video](./Output%20Video.mp4)

<video width="600" controls>
  <source src="Output%20Video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


![Output](/Output%20Photo.png)

---

### **Code Breakdown**:

#### 1. **Mounting Google Drive**:
```python
from google.colab import drive
drive.mount('/content/drive')
```
- This mounts your Google Drive to access the corpus files located on Google Drive.

#### 2. **Fetching Corpus Files**:
```python
corpus_folder = '/content/drive/My Drive/corpus_folder/'  # Specify folder path

def get_corpus_files(folder_path):
    corpus_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):  # Only load .txt files
            corpus_files.append(filename)
    return corpus_files
```
- This function dynamically fetches all `.txt` files from the specified folder. Only the text files are considered as part of the corpus.

#### 3. **Loading Corpus and Tokenizing**:
```python
def load_corpus(corpus_files):
    corpus = {}
    for filename in corpus_files:
        file_path = os.path.join(corpus_folder, filename)
        with open(file_path, 'r') as file:
            corpus[filename] = file.read().lower().split()  # Split text into words (tokens)
    return corpus
```
- The `load_corpus()` function reads each file, tokenizes its content by splitting the text into words, and stores the result in a dictionary.

#### 4. **Building Inverted Index**:
```python
def build_index(corpus):
    dictionary = defaultdict(lambda: {'df': 0, 'postings': []})
    document_lengths = {}
    N = len(corpus)  # Total number of documents
```
- Here, a dictionary is created using **defaultdict** to store terms and their corresponding document frequencies and postings lists (docID, term frequency).
- The **document_lengths** dictionary is used to normalize document vectors later.

```python
    for docID, content in corpus.items():
        term_freqs = defaultdict(int)
        for term in content:
            term_freqs[term] += 1
        
        for term, freq in term_freqs.items():
            dictionary[term]['df'] += 1  # Update document frequency
            dictionary[term]['postings'].append((docID, freq))  # Add to postings list
```
- This loop processes each document and counts the frequency of each term. It updates the **df** (document frequency) and adds the posting (docID and term frequency) to the dictionary.

#### 5. **Document Length Calculation**:
```python
        doc_length = 0
        for term, freq in term_freqs.items():
            doc_length += (1 + math.log10(freq)) ** 2
        document_lengths[docID] = math.sqrt(doc_length)
```
- Here, we compute the length of each document for normalization purposes, applying the formula \(1 + \log_{10}(tf)\) for term frequency.

#### 6. **Computing TF-IDF**:
```python
def compute_tfidf(term, freq, df, N, for_query=False):
    tf = 1 + math.log10(freq)  # Logarithmic term frequency
    if for_query:
        idf = math.log10(N / df)  # Inverse document frequency for queries
        return tf * idf
    return tf  # Return only tf for documents (lnc scheme)
```
- This function calculates the **tf-idf** score. For queries, we calculate both tf and idf, but for documents, we only calculate tf (lnc scheme).

#### 7. **Ranking Documents**:
```python
def rank_documents(query, dictionary, document_lengths, N):
    query_terms = query.lower().split()
    query_freqs = defaultdict(int)
    for term in query_terms:
        query_freqs[term] += 1
```
- The query is tokenized and the frequency of each term is counted.

```python
    query_vector = {}
    for term, freq in query_freqs.items():
        if term in dictionary:
            query_vector[term] = compute_tfidf(term, freq, dictionary[term]['df'], N, for_query=True)
```
- For each query term, the **tf-idf** is calculated using the `compute_tfidf` function.

```python
    scores = defaultdict(float)
    for term in query_vector:
        if term in dictionary:
            postings = dictionary[term]['postings']
            for docID, doc_freq in postings:
                doc_tfidf = compute_tfidf(term, doc_freq, dictionary[term]['df'], N, for_query=False)
                scores[docID] += query_vector[term] * doc_tfidf  # Cosine similarity calculation
```
- Each document's score is computed based on cosine similarity between the query vector and document vector. The more similar the document, the higher the score.

```python
    for docID in scores:
        scores[docID] /= document_lengths[docID]  # Normalize by document length
    ranked_docs = sorted(scores.items(), key=lambda x: (-x[1], x[0]))  # Sort by relevance
    return ranked_docs[:10]  # Return top 10 documents
```
- The final scores are normalized by document lengths, and the documents are sorted based on relevance.

#### 8. **Displaying Results**:
```python
def display_results(results):
    print("Top 10 Relevant Documents:")
    for rank, (doc, score) in enumerate(results, start=1):
        print(f"{rank}. {doc} (Relevance Score: {round(score, 5)})")
```
- This function prints the top 10 relevant documents sorted by relevance scores.

#### 9. **Main Function**:
```python
def main():
    corpus_files = get_corpus_files(corpus_folder)  # Fetch corpus files
    corpus = load_corpus(corpus_files)  # Load corpus content
    dictionary, document_lengths, N = build_index(corpus)  # Build inverted index
    
    query = input("Enter your search query: ")  # User inputs search query
    ranked_docs = rank_documents(query, dictionary, document_lengths, N)  # Rank documents
    
    display_results(ranked_docs)  # Show the top 10 results
```
- The **main function** runs the entire process: loading the corpus, building the index, accepting a query from the user, ranking the documents, and displaying the top 10 results.

#### 10. **Running the Script**:
```python
if __name__ == "__main__":
    main()
```
- This ensures the script runs when executed directly.

---

### **Features**:
- **lnc.ltc Scheme**: Implements the logarithmic term frequency, inverse document frequency (idf for queries), and cosine normalization.
- **Top 10 Results**: Returns up to the top 10 most relevant documents for a given free-text query.
- **Dynamic Corpus Handling**: Automatically reads and processes all `.txt` files in the specified directory.

---

### **Instructions to Run**:
1. **Mount Google Drive**: To access the corpus from your drive.
2. **Specify Folder Path**: Adjust the path of the corpus in the script.
3. **Run the Code**: Input queries, and the script will return the top 10 relevant documents sorted by relevance.

---

### **Group Members**:  
- Keshav Dubey  
- Utkarsh Tiwari

