from rank_bm25 import BM25Okapi
from typing import List, Dict, Optional, Union, Tuple
from pydantic import BaseModel, Field, HttpUrl
import httpx
from bs4 import BeautifulSoup
import asyncio
from urllib.parse import urljoin, urlparse
import time
from robots import RobotsParser
import mysql.connector
from abc import ABC, abstractmethod
from contextlib import contextmanager
from atlassian import Confluence
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
from datetime import datetime
import re
import nltk
# loaing needed NLTK packages
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.summarization import keywords

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




#data validation via pydantic
class DocumentSource(BaseModel):
    """
    A class representing a source of documents.
    This class inherits from BaseModel and defines attributes for document sources.
    Attributes:
        source_type (str): The type of the document source (e.g., 'file', 'database', 'api').
    Example:
        doc_source = DocumentSource(source_type='file')
    """

    source_type=str

class ConfluenceSource(DocumentSource):
    """A class representing a document source from Confluence.
    This class extends DocumentSource to specifically handle Confluence pages as document sources.
    It contains metadata about a Confluence page including its ID, title, space, version, and other
    attributes.
    Attributes:
        page_id (str): The unique identifier of the Confluence page
        title (str): The title of the Confluence page
        space_key (str): The key of the Confluence space containing the page
        version (int): The version number of the Confluence page
        last_modified (datetime): The timestamp of when the page was last modified
        author (str): The username of the page author
        labels (List[str]): A list of labels/tags associated with the page
    """

    page_id:str
    title:str
    space_key:str
    version:int
    last_modified:datetime
    author:str
    labels:List[str] = Field(default_factory=list)

class LogscaleSource(DocumentSource):
    """LogscaleSource represents a document source from Logscale.

    A class that captures metadata about a document source from Logscale, including its URL,
    title, section, last crawl time, and navigation breadcrumb.

    Attributes:
        url (HttpUrl): The HTTP URL where the document is located
        title (str): The title of the document
        section (str): The section or category the document belongs to
        last_crawled (datetime): Timestamp of when the document was last crawled
        breadcrumb (List[str]): List representing the navigation path to the document

    Inherits from:
        DocumentSource: Base class for document sources
    """

    url:HttpUrl
    title:str
    section:str
    last_crawled:datetime
    breadcrumb: List[str] = Field(default_factory=list)
    
class ConfluenceMetadata(BaseModel):
    """
    A Pydantic model representing metadata for a Confluence page.
    Attributes:
        page_id (str): The unique identifier of the Confluence page
        title (str): The title of the Confluence page
        space_key (str): The key of the space where the page is located
        version (int): The version number of the page
        last_modified (datetime): Timestamp of when the page was last modified
        author (str): The author of the page
        labels (List[str]): List of labels/tags associated with the page, defaults to empty list
    """

    page_id:str
    title:str
    space_key:str
    version:int
    last_modified:datetime
    author:str
    labels: List[str] = Field(default_factory=list)

class DocumentChunk(BaseModel):
    """
    A Pydantic BaseModel class representing a chunk of a document with associated metadata and embeddings.

    Attributes:
        content (str): The text content of the document chunk.
        metadata (Union[ConfluenceMetadata, LogscaleSource]): Metadata associated with the document chunk,
            can be either ConfluenceMetadata or LogscaleSource.
        embedding (Optional[List[float]]): Vector embedding representation of the content.
            Defaults to None.
        relevance_score (Optional[float]): A score indicating relevance of this chunk to a query.
            Defaults to None.
    """

    content:str
    metadata:Union[ConfluenceMetadata, LogscaleSource]
    embedding:Optional[List[float]] = None
    relevance_score:Optional[float] = None

class RankedDocument(BaseModel):
    """
    A document with its associated ranking scores from the RAG pipeline.
    This class represents a document chunk along with its final relevance score and individual stage scores
    from different ranking stages in the retrieval process.
    Attributes:
        chunk (DocumentChunk): The document chunk being ranked
        final_score (float): The final aggregated relevance score for this document
        stage_scores (Dict[str,float]): Dictionary mapping stage names to their individual scores
    """

    chunk:DocumentChunk
    final_score:float
    stage_scores:Dict[str,float]


# connecting to Confluence
class ConfluenceConnector:
    """A class to connect and retrieve content from Confluence.
    This class provides functionality to establish a connection with Confluence
    and retrieve page content along with its metadata.
    Attributes:
        confluence (Confluence): The Confluence API client instance.
    Example:
        ```
        connector = ConfluenceConnector(
            url='https://your-domain.atlassian.net',
            username='your_email@domain.com',
            api_token='your_api_token'
        content, metadata = connector.get_page_content('123456')
        ```
    """

    def __init__(self, url:str, username:str, api_token: str):
        """Initialize a new instance of the class.
        This constructor sets up a connection to Confluence using the Atlassian Python API.
        Args:
            url (str): The base URL of your Confluence instance.
            username (str): The username for authentication.
            api_token (str): The API token for authentication.
        Returns:
            None
        Example:
            >>> instance = ClassName("https://your-domain.atlassian.net", "your_email@example.com", "your_api_token")
        """

        self.confluence = Confluence(
            url=url,
            username=username,
            password=api_token,
            cloud=True
        )
    def get_page_content(self, page_id: str) -> tuple[str, ConfluenceMetadata]:
        """
        Retrieves the content and metadata of a Confluence page by its ID.
        Args:
            page_id (str): The unique identifier of the Confluence page.
        Returns:
            tuple[str, ConfluenceMetadata]: A tuple containing:
                - str: The cleaned HTML content of the page
                - ConfluenceMetadata: A dataclass object containing the page's metadata including:
                    - page_id: The page's unique identifier
                    - title: The page's title
                    - space_key: The key of the space containing the page
                    - version: The page's version number
                    - last_modified: The datetime when the page was last modified
                    - author: The display name of the page's last modifier
                    - labels: List of labels attached to the page
        Raises:
            Exception: If the page cannot be retrieved or if there's an error processing the page content.
        """
        
        page = self.confluence.get_page_by_id(page_id,expand='version,body.storage,metadata.labels')
        metadata = ConfluenceMetadata(
            page_id=page['id'],
            title=page['title'],
            space_key=page['space']['key'],
            version=page['version']['number'],
            last_modified=datetime.fromisoformat(page['version']['when']),
            author=page['version']['by']['displayName'],
            labels=[label['name'] for label in page.get('metadata', {}).get('labels', {}).get('results',[])]
        )

        content = self._clean_html(page['body']['storage']['value'])
        return content, metadata
    
    def _clean_html(self, html_content:str) -> str:
        """
        Clean HTML content by removing HTML tags and normalizing whitespace.
        This method removes all HTML tags from the input text and normalizes whitespace
        by replacing multiple consecutive whitespace characters with a single space.
        Args:
            html_content (str): The HTML content to be cleaned.
        Returns:
            str: The cleaned text with HTML tags removed and normalized whitespace.
        Example:
            >>> _clean_html("<p>Hello  World!</p>")
            'Hello World!'
        """

        text = re.sub('<[^<]+?>', '', html_content)
        text = re.sub(r'\s+',' ',text).strip()
        return text


# processing documents
class DocumentProcessor:
    """A class for processing and chunking documents with metadata.
    This class provides functionality to process documents by breaking them into smaller chunks
    and extracting keywords. It uses TF-IDF vectorization for text processing and NLTK for 
    sentence tokenization.
    Attributes:
        chunk_size (int): Maximum size of each document chunk. Defaults to 512.
        overlap (int): Overlap size between consecutive chunks. Defaults to 50.
        tfidf (TfidfVectorizer): TF-IDF vectorizer for text processing.
    Methods:
        chunk_document(content: str, metadata: ConfluenceMetadata) -> List[DocumentChunk]:
            Splits a document into chunks while preserving sentence boundaries.
        get_document_keywords(text: str, num_keywords: int = 5) -> List[str]:
            Extracts the most important keywords from the given text.
    """
    
    def __init__(self,chunk_size: int = 512,overlap:int =50):
        """
        Initialize the MDR_RAG class.
        A class for handling document retrieval and analysis using TF-IDF vectorization.
        Args:
            chunk_size (int, optional): The size of text chunks for processing. Defaults to 512.
            overlap (int, optional): The overlap size between consecutive chunks. Defaults to 50.
        Attributes:
            chunk_size (int): Size of text chunks for processing
            overlap (int): Overlap between consecutive chunks
            tfidf (TfidfVectorizer): TF-IDF vectorizer instance with English stop words and 1000 max features
        Example:
            >>> rag = MDR_RAG()
            >>> rag = MDR_RAG(chunk_size=256, overlap=25)
        """

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tfidf = TfidfVectorizer(
            max_feature=1000,
            stop_words='english'
        )
    def chunk_document(self,content:str, metadata: ConfluenceMetadata) -> List[DocumentChunk]:
        """
        Chunks a document's content into smaller pieces based on sentence boundaries.
        This method splits the input text into sentences and groups them into chunks while respecting
        sentence boundaries and maintaining a maximum chunk size. Each chunk is created with its
        associated metadata.
        Args:
            content (str): The text content to be chunked.
            metadata (ConfluenceMetadata): Metadata associated with the document.
        Returns:
            List[DocumentChunk]: A list of DocumentChunk objects, each containing a portion of the
            original text and the associated metadata.
        Notes:
            - Uses NLTK's sent_tokenize for sentence splitting
            - Ensures chunks don't exceed the predefined chunk_size
            - Preserves complete sentences within chunks
            - Maintains document metadata across all chunks
        """

        sentences = nltk.sent_tokenize(content)
        chunks=[]
        current_chunks=[]
        current_length=0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > self. chunk_size:
                #creatig new chunk
                if current_chunks:
                    chunk_text = ' '.join(current_chunks)
                    chunks.append(DocumentChunk(content=chunk_text, metadata=metadata))
                
                current_chunks = [sentence]
                current_length = sentence_length
            
            else:
                current_chunks.append(sentence)
                current_lenght += sentence_length

        #add remaining chunks
        if current_chunks:
            chunk_text - ' '.join(current_chunks)
            chunks.append(DocumentChunk(content=chunk_text, metadata=metadata))
        return chunks
    
    def get_document_keywords(self, text: str, num_keywords: int = 5) -> List[str]:
        '''
        Extract keywords from the input text.
        This method processes the input text and returns a list of most relevant keywords.
        Args:
            text (str): The input text from which to extract keywords
            num_keywords (int, optional): Maximum number of keywords to return. Defaults to 5.
        Returns:
            List[str]: A list of extracted keywords, limited to num_keywords elements.
        Example:
            >>> text = "Machine learning is a subset of artificial intelligence"
            >>> get_document_keywords(text, 3)
            ['machine learning', 'artificial intelligence', 'subset']
        '''

        return keywords(text).split('\n')[:num_keywords]


# stage 1 reranking class
class QueryExpansionModule:
    def __init__(self):
        """
        Initialize the POS (Part of Speech) mapping dictionary.
        The POS mapping dictionary maps NLTK POS tags to WordNet POS categories.
        This mapping is essential for lemmatization and other NLP tasks that require
        WordNet POS information.
        Attributes:
            pos_map (dict): Dictionary mapping NLTK POS tags to WordNet categories
                - 'NOUN': wordnet.NOUN
                - 'VERB': wordnet.VERB
                - 'ADJ': wordnet.ADJ
                - 'ADV': wordnet.ADV
        """

        self.pos_map = {
            'NOUN': wordnet.NOUN,
            'VERB':wordnet.VERB,
            'ADJ':wordnet.ADJ,
            'ADV':wordnet.ADV
        }
    def _get_wordnet_pos(self, word: str) ->str:
        """
        Determines the WordNet part of speech (POS) tag for a given word.
        This method takes a word and returns its corresponding WordNet POS tag using NLTK's
        pos_tag function. If the POS tag is not found in pos_map, defaults to NOUN.
        Args:
            word (str): The word to determine the part of speech for.
        Returns:
            str: The WordNet POS tag for the word (e.g., wordnet.NOUN, wordnet.VERB, etc.)
        Example:
            >>> _get_wordnet_pos('running')
            'v'  # Returns verb tag for 'running'
        """
        
        tag = nltk.pos_tag([word])[0][1][0].upper()
        return self.pos_map.get(tag,wordnet.NOUN)
    
    def expand_query(self, query:str) -> List[str]:
        expanded_queries = [query]



# stage 4 reranking class
class BM25Ranking:
    """A class implementing BM25 (Best Matching 25) ranking algorithm for document retrieval and ranking.
    BM25 is a ranking function used to estimate the relevance of documents to a given search query. 
    It builds on the TF-IDF weighting scheme by adding parameters to control term frequency saturation
    and document length normalization.
        bm25 (BM25Okapi | None): Instance of the BM25 ranking model.
        tokenized_corpus (list | None): List of tokenized documents in the corpus.
    Methods:
        _tokenize(text: str) -> List[str]:
            Tokenizes input text by converting to lowercase, removing stopwords and 
            keeping only alphanumeric tokens.
        _prepare_corpus(chunks: List[DocumentChunk]):
            Prepares the document corpus by tokenizing each document chunk and 
            initializing the BM25 model.
        reranking(query: str, chunks: List[DocumentChunk]) -> List[Tuple[DocumentChunk, float]]:
            Ranks document chunks based on their relevance to the query using BM25 scoring.
            Returns sorted list of (chunk, score) tuples."""
    
    def __init__(self):
        """
        Initialize the MDR_RAG class.
        This constructor initializes two instance variables:
        - bm25: Stores the BM25 ranking model, initially set to None
        - tokenized_corpus: Stores the tokenized document corpus, initially set to None
        Attributes:
            bm25 (BM25 | None): The BM25 ranking model instance
            tokenized_corpus (list | None): The tokenized corpus of documents
        """
        
        self.bm25=None
        self.tokenized_corpus=None
        
    def _tokenize(self, text:str) -> List[str]:
        """
        Tokenizes the input text into a list of tokens.
        This method processes text by converting to lowercase, tokenizing, and filtering tokens.
        It removes stopwords (except numbers) and non-alphanumeric characters.
        Args:
            text (str): The input text to be tokenized.
        Returns:
            List[str]: A list of processed tokens that are:
                - Alphanumeric
                - Either numeric or not in English stopwords
        Example:
            >>> _tokenize("The quick brown fox jumps!")
            ['quick', 'brown', 'fox', 'jumps']
        """
        tokens = word_tokenize(text.lower())
        tokens =[
            token for token in tokens
            if token.isalnum() and (token.isnumeric() or token not in stopwords.words('english'))
        ]
        return tokens
   
    def _prepare_corpus(self,chunks:List[DocumentChunk]):
        """
        Prepares the corpus for BM25 retrieval by tokenizing document chunks and initializing BM25Okapi.
        Args:
            chunks (List[DocumentChunk]): A list of DocumentChunk objects containing document content.
        Attributes:
            tokenized_corpus (List[List[str]]): A list of tokenized documents.
            bm25 (BM25Okapi): BM25 scoring object initialized with tokenized corpus.
        Note:
            This method assumes existence of a _tokenize() helper method that processes raw text.
        """
        self.tokenized_corpus = [
            self._tokenize(chunk.content)
            for chunk in chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def reranking(self, query:str, chunks: List[DocumentChunk]) -> List[Tuple[DocumentChunk], float]:
        """
        Reranks document chunks based on BM25 similarity with the query.
        This method uses the BM25 algorithm to score document chunks against a query, 
        normalizes the scores, and returns sorted chunk-score pairs.
        Args:
            query (str): The search query to compare chunks against
            chunks (List[DocumentChunk]): List of document chunks to be ranked
        Returns:
            List[Tuple[DocumentChunk, float]]: A sorted list of tuples containing document chunks 
            and their normalized BM25 scores, ordered by decreasing relevance score
        Note:
            - If BM25 hasn't been initialized or corpus size changed, it will prepare the corpus first
            - Scores are normalized to [0,1] range before sorting
        """
        if not self.bm25 or len(chunks) != len(self.tokenized_corpus):
            self._prepare_corpus(chunks)

        #tokenize query
        tokenized_query = self._tokenize(query)
        # getting bm25 score
        scores = self.bm25.get_scores(tokenized_query)
        
        #Normalize scores
        max_score = max(scores) if scores else 1
        normalized_scores = [score / max_score if max_score >0 else 0 for score in scores ]

        #combine chunks with scores
        chunk_scores = list(zip(chunks,normalized_scores))

        return sorted(chunk_scores,key=lambda x:x[1], reverse=True)




