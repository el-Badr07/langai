# langai: LangChain Function Alternatives & Document Intelligence Playground

This repository is a comprehensive playground for experimenting with alternatives to LangChain's document loaders, text splitters, transformers, vector stores, retrieval, agentic workflows, and evaluation utilities. It is designed for developers and researchers building advanced LLM-powered applications, document processing pipelines, and agent-based systems.

---

## 🚀 Features

- **Document Loaders**: Load text, PDF, CSV, JSON, XML, Markdown, HTML, YAML, Excel, images, audio, video, and more.
- **Text Splitters**: Split documents by characters, tokens, or recursively by multiple separators.
- **Transformers**: Convert between HTML, text, JSON, PDF, Markdown, YAML, Excel, and more.
- **Vector Store & Retrieval**: Store and retrieve documents using vector embeddings and semantic search.
- **Agent & Tooling Framework**: Build agentic workflows, chain-of-thought reasoning, prompt templates, and tool integrations.
- **Evaluation & Metrics**: Evaluate retrieval and generation quality with precision, recall, F1, and relevance scoring.
- **Extensible**: 30+ extensible classes for advanced document and agent workflows.

---

## 📝 Example Usage

### 1. Load and Split a Text File

```python
from test import TextLoader, RecursiveCharacterTextSplitter

loader = TextLoader("example.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

print(f"Split {len(documents)} documents into {len(chunks)} chunks")
```

### 2. Transform Documents

```python
from test import HTMLToTextTransformer

transformer = HTMLToTextTransformer()
plain_text_docs = transformer.transform_documents(documents)
```

### 3. Vector Search

```python
from test import VectorStore, Document

def embedding_function(text):
    return [ord(c) for c in text]

vector_store = VectorStore(embedding_model=embedding_function)
vector_store.add_documents(documents)
results = vector_store.similarity_search("AI and machine learning", k=2)
```

---

## 📄 Sample Data

- [`example.txt`](example.txt): 100 technology-related sentences for loader/splitter tests.
- [`test_docs/`](test_docs/): Sample files in TXT, PDF, DOCX, and HTML formats for loader and transformer testing.

---

## 🧪 Testing

- Unit tests are provided in [`unittests1.py`](unittests1.py) and [`exemple_tests.py`](exemple_tests.py).
- Run tests using:
  ```sh
  python -m unittest unittests1.py
  ```

---

## 🛠️ Setup

1. **Clone the repository**
   ```sh
   git clone https://github.com/your-org/giads.git
   cd giads
   ```

2. **Create a virtual environment**
   ```sh
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

4. **Run examples**
   ```sh
   python test.py
   ```

---

## 📚 Documentation

- Each class in [`test.py`](test.py) is documented with comments.
- See the code for detailed usage examples and docstrings.
- For advanced agent and workflow patterns, see the test scripts and sample data.

---

## 🤝 Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, new loaders/transformers, or improved agent workflows.

---

