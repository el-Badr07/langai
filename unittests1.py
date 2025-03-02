import unittest
import os
import tempfile
import json
from unittest.mock import patch, mock_open, MagicMock
from test import Document, TextLoader, PDFLoader, CSVLoader, JSONLoader, DirectoryLoader
from test import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter
from test import HTMLToTextTransformer, TextToHTMLTransformer, TextToJSONTransformer

class TestDocument(unittest.TestCase):
    def test_document_initialization(self):
        doc = Document(page_content="Test content")
        self.assertEqual(doc.page_content, "Test content")
        self.assertEqual(doc.metadata, {})
        
        doc = Document(page_content="Test content", metadata={"source": "test"})
        self.assertEqual(doc.page_content, "Test content")
        self.assertEqual(doc.metadata, {"source": "test"})

class TestTextLoader(unittest.TestCase):
    @patch("builtins.open", mock_open(read_data="Test content"))
    def test_load(self):
        loader = TextLoader("fake_file.txt")
        result = loader.load()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].page_content, "Test content")
        self.assertEqual(result[0].metadata, {"source": "fake_file.txt"})

class TestPDFLoader(unittest.TestCase):
    @patch("PyPDF2.PdfReader")
    @patch("builtins.open", mock_open())
    def test_load(self, mock_pdf_reader):
        # Setup mock PDF pages
        page1 = MagicMock()
        page1.extract_text.return_value = "Page 1 content"
        page2 = MagicMock()
        page2.extract_text.return_value = "Page 2 content"
        
        mock_pdf_reader.return_value.pages = [page1, page2]
        
        loader = PDFLoader("fake.pdf")
        result = loader.load()
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].page_content, "Page 1 content")
        self.assertEqual(result[0].metadata, {"source": "fake.pdf", "page": 1, "total_pages": 2})
        self.assertEqual(result[1].page_content, "Page 2 content")
        self.assertEqual(result[1].metadata, {"source": "fake.pdf", "page": 2, "total_pages": 2})

class TestCSVLoader(unittest.TestCase):
    @patch("builtins.open", mock_open(read_data="col1,col2\nvalue1,value2\nvalue3,value4"))
    def test_load(self):
        loader = CSVLoader("fake.csv")
        result = loader.load()
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].page_content, "col1: value1\ncol2: value2")
        self.assertEqual(result[0].metadata, {"source": "fake.csv", "row": 1})
        self.assertEqual(result[1].page_content, "col1: value3\ncol2: value4")
        self.assertEqual(result[1].metadata, {"source": "fake.csv", "row": 2})

class TestJSONLoader(unittest.TestCase):
    @patch("builtins.open", mock_open(read_data='{"key1": "value1", "key2": "value2"}'))
    def test_load(self):
        loader = JSONLoader("fake.json")
        result = loader.load()
        
        self.assertEqual(len(result), 1)
        self.assertTrue("key1" in result[0].page_content)
        self.assertTrue("value1" in result[0].page_content)
        self.assertEqual(result[0].metadata, {"source": "fake.json"})

class TestDirectoryLoader(unittest.TestCase):
    @patch("os.walk")
    @patch("os.listdir")
    @patch("os.path.isfile")
    @patch("test.TextLoader.load")
    @patch("test.PDFLoader.load")
    def test_load_non_recursive(self, mock_pdf_load, mock_text_load, mock_isfile, mock_listdir, mock_walk):
        mock_listdir.return_value = ["file1.txt", "file2.pdf", "file3.unknown"]
        mock_isfile.return_value = True
        
        # Mock TextLoader.load
        text_doc = Document(page_content="Text content", metadata={"source": "file1.txt"})
        mock_text_load.return_value = [text_doc]
        
        # Mock PDFLoader.load
        pdf_doc = Document(page_content="PDF content", metadata={"source": "file2.pdf"})
        mock_pdf_load.return_value = [pdf_doc]
        
        loader = DirectoryLoader("/fake/dir", recursive=False)
        result = loader.load()
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].page_content, "Text content")
        self.assertEqual(result[1].page_content, "PDF content")
        
        # Verify walk was not called for non-recursive
        mock_walk.assert_not_called()

class TestCharacterTextSplitter(unittest.TestCase):
    def test_split_text(self):
        splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=2)
        text = "This is a test text for splitter."
        chunks = splitter.split_text(text)
        
        self.assertTrue(len(chunks) > 1)
        for chunk in chunks:
            self.assertTrue(len(chunk) <= 10)

    def test_split_documents(self):
        splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=2)
        doc = Document(page_content="This is a test text for splitter.", metadata={"source": "test"})
        chunks = splitter.split_documents([doc])
        
        self.assertTrue(len(chunks) > 1)
        for chunk in chunks:
            self.assertTrue(len(chunk.page_content) <= 10)
            self.assertEqual(chunk.metadata["source"], "test")
            self.assertIn("chunk", chunk.metadata)

class TestRecursiveCharacterTextSplitter(unittest.TestCase):
    def test_split_text(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=2)
        text = "This is a\ntest text for\nsplitter."
        chunks = splitter.split_text(text)
        
        self.assertTrue(len(chunks) > 1)
        for chunk in chunks:
            self.assertTrue(len(chunk) <= 10)

    def test_split_documents(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=2)
        doc = Document(page_content="This is a\ntest text for\nsplitter.", metadata={"source": "test"})
        chunks = splitter.split_documents([doc])
        
        self.assertTrue(len(chunks) > 1)
        for chunk in chunks:
            self.assertTrue(len(chunk.page_content) <= 10)
            self.assertEqual(chunk.metadata["source"], "test")
            self.assertIn("chunk", chunk.metadata)

class TestTokenTextSplitter(unittest.TestCase):
    @patch("tiktoken.encoding_for_model")
    def test_split_text_with_tiktoken(self, mock_tiktoken):
        # Mock tiktoken encode/decode methods
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(20))  # Fake 20 tokens
        mock_encoder.decode.side_effect = lambda x: "Chunk " + str(len(x))
        mock_tiktoken.return_value = mock_encoder
        
        splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=2)
        text = "This is a test text for tokenization."
        chunks = splitter.split_text(text)
        
        self.assertEqual(len(chunks), 2)

    @patch("tiktoken.encoding_for_model", side_effect=ImportError)
    def test_split_text_without_tiktoken(self, mock_tiktoken):
        splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=2)
        text = "This is a test text for tokenization."
        chunks = splitter.split_text(text)
        
        self.assertTrue(len(chunks) >= 1)

class TestHTMLToTextTransformer(unittest.TestCase):
    def test_transform_documents(self):
        transformer = HTMLToTextTransformer()
        html_content = "<html><body><h1>Title</h1><p>Paragraph content</p></body></html>"
        doc = Document(page_content=html_content, metadata={"source": "test.html"})
        
        result = transformer.transform_documents([doc])
        
        self.assertEqual(len(result), 1)
        self.assertNotIn("<html>", result[0].page_content)
        self.assertNotIn("<body>", result[0].page_content)
        self.assertTrue("Title" in result[0].page_content)
        self.assertTrue("Paragraph content" in result[0].page_content)

    @patch("test.BeautifulSoup", side_effect=ImportError)
    def test_transform_documents_fallback(self, mock_bs):
        transformer = HTMLToTextTransformer()
        html_content = "<html><body><h1>Title</h1><p>Paragraph content</p></body></html>"
        doc = Document(page_content=html_content, metadata={"source": "test.html"})
        
        result = transformer.transform_documents([doc])
        
        self.assertEqual(len(result), 1)
        self.assertNotIn("<html>", result[0].page_content)
        self.assertNotIn("<body>", result[0].page_content)

class TestTextToHTMLTransformer(unittest.TestCase):
    def test_transform_documents(self):
        transformer = TextToHTMLTransformer()
        text_content = "Simple text content"
        doc = Document(page_content=text_content, metadata={"source": "test.txt"})
        
        result = transformer.transform_documents([doc])
        
        self.assertEqual(len(result), 1)
        self.assertTrue("<html>" in result[0].page_content)
        self.assertTrue("<body>" in result[0].page_content)
        self.assertTrue("<p>Simple text content</p>" in result[0].page_content)

class TestTextToJSONTransformer(unittest.TestCase):
    def test_transform_documents(self):
        transformer = TextToJSONTransformer()
        text_content = "Simple text content"
        doc = Document(page_content=text_content, metadata={"source": "test.txt"})
        
        result = transformer.transform_documents([doc])
        
        self.assertEqual(len(result), 1)
        # Parse JSON to verify it's valid
        json_data = json.loads(result[0].page_content)
        self.assertEqual(json_data["text"], "Simple text content")
        self.assertEqual(json_data["metadata"], {"source": "test.txt"})

class TestPipeline(unittest.TestCase):
    def test_pipeline(self):
        # Setup pipeline stages
        loader = TextLoader("example.txt")

        splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=2)
        transformer = TextToHTMLTransformer()
        
        # Load and split text
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        
        # Transform each chunk
        transformed_chunks = transformer.transform_documents(chunks)
        
        # self.assertEqual(len(transformed_chunks), 1)
        self.assertTrue("<html>" in transformed_chunks[0].page_content)
        self.assertTrue("<body>" in transformed_chunks[0].page_content)
        self.assertTrue("<p>" in transformed_chunks[0].page_content)

class TestTexttoJSONTransformer(unittest.TestCase):
    def test_transform_documents(self):
        transformer = TextToJSONTransformer()
        text_content = "Simple text content"
        doc = Document(page_content=text_content, metadata={"source": "test.txt"})
        
        result = transformer.transform_documents([doc])
        
        self.assertEqual(len(result), 1)
        # Parse JSON to verify it's valid
        json_data = json.loads(result[0].page_content)
        self.assertEqual(json_data["text"], "Simple text content")
        self.assertEqual(json_data["metadata"], {"source": "test.txt"})
if __name__ == "__main__":
    unittest.main()