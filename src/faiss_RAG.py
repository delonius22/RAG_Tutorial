import requests as r 
from pydantic import BaseModel
import logging 
from typing import Optional, Dict, List 
import numpy as np
import faiss
# import mysql.connector
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from atlassian import Confluence
from sentence_transformers import SentenceTransformer
import json
import pickle
import re
from collections import defaultdict
import time
from datetime import datetime
import sqlite3


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentMetadata(BaseModel):
    """A base model class representing metadata information for documents in a RAG system.

    This class stores essential metadata for documents including their content, embeddings,
    source information, and other relevant attributes for document retrieval and management.

    Attributes:
        id (str): Unique identifier for the document
        text (str): The actual text content of the document
        embedding (np.ndarray, optional): Vector representation of the document content
        source_url (str, optional): URL where the document was sourced from
        parent_id (str, optional): ID of the parent document if hierarchical
        metadata (Dict[str, Any], optional): Additional metadata key-value pairs
        created_at (datetime, optional): Timestamp when document was created
        tags (List[str], optional): List of tags/labels associated with document
        version (int, optional): Version number of the document

    Note:
        All optional fields default to None if not specified.
    """
    id: str
    text: str
    embedding: Optional[np.ndarray] = None
    source_url: Optional[str] = None
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    tags: List[str] = None
    version: int = None

class FAISSIndex:
    """A class for managing FAISS (Facebook AI Similarity Search) index operations.
    This class provides functionality for creating, loading, saving and searching vector indices
    using the FAISS library. It supports different index types and is optimized for
    high-dimensional vector similarity searches.
        logger: Logger instance for tracking operations
        vector_size (int): Dimension of vectors to be indexed (default: 768)
        index_type (str): Type of FAISS index being used (default: 'IVFFlat')
        index_params (Dict): Additional parameters for index configuration
        chunk_ids (list): List storing chunk identifiers
        index_path (str): File path for saving/loading the index
    Methods:
        create_index(): Creates a new FAISS index
        load_index(): Loads an existing FAISS index from disk
        save_index(): Saves the current FAISS index to disk
        add_vectors(vectors, chunk_ids): Adds vectors and their IDs to the index
        search_vectors(query_vector, k): Searches for k nearest neighbors of a query vector
    Example:
        ```
        index_manager = FAISSIndex(vector_size=768, index_type='IVFFlat')
        index_manager.add_vectors(vectors, chunk_ids)
        results = index_manager.search_vectors(query_vector, k=10)
        ```
        - The index is automatically created during initialization
        - Vectors are normalized before being added to the index
        - The default index type is 'IVFFlat' which provides a good balance of speed and accuracy"""
 
    def __init__(self, vector_size: int = 768, index_type: str = 'IVFFlat', index_params: Dict = None):
        """Initialize the FAISS Index Manager.
        Constructs a FAISS index manager with specified parameters for vector indexing.
        Args:
            vector_size (int, optional): Dimensionality of vectors to be indexed. Defaults to 768.
            index_type (str, optional): Type of FAISS index to create. Defaults to 'IVFFlat'.
            index_params (Dict, optional): Additional parameters for index creation. Defaults to None.
        Attributes:
            logger: Logger instance for this class
            vector_size (int): Size of vectors to be indexed
            index_type (str): Type of FAISS index being used
            index_params (Dict): Parameters for index configuration
            index: The FAISS index instance
            chunk_ids (list): List to store chunk identifiers
            index_path (str): Path where index will be saved
        Notes:
            The index is automatically created during initialization.
        """
        
        self.logger = logging.getLogger(__name__)
        self.vector_size = vector_size
        self.index_type = index_type
        self.index_params = index_params
        self.index = self.create_index()
        self.chunk_ids = []
        self.index_path = 'faiss_index.index'
        self.logger.info(f'Initialized FAISS index with size {self.vector_size} and type {self.index_type}.')
    def create_index(self):
        """
        Creates a new FAISS index.
        """
        if self.index_type == 'IVFFlat':
            #creating quantinizer for the index
            quantizer = faiss.IndexFlatL2(self.vector_size)

            #number of clusters
            n_clusters = 100

            #Create IVF index
            index = faiss.IndexIVFFlat(quantizer, self.vector_size, n_clusters, faiss.METRIC_INNER_PRODUCT)       
            self.logger.info(f'Created FAISS index of type {self.index_type} with size {self.vector_size}.')
            return index
            
        else:
            #defaulting to flat index
            return faiss.IndexFlatL2(self.vector_size)
  
        

    def load_index(self):
        """
        Loads an existing FAISS index from disk.
        """
        self.index = faiss.read_index(self.index_path)
        self.logger.info(f'Loaded FAISS index from {self.index_path}.')

    def save_index(self):
        """
        Saves the FAISS index to disk.
        """
        faiss.write_index(self.index, self.index_path)
        self.logger.info(f'Saved FAISS index to {self.index_path}.')

    def add_vectors(self, vectors: np.ndarray, chunk_ids: List[str]):
        """
        Adds vectors to the FAISS index.

        Args:
            vectors (np.ndarray): An array of vectors to add to the index.
        """
        try:

            if self.index is None:
                self.logger.error('Index has not been created or loaded.')
                raise ValueError('Index has not been created or loaded.')
            #normalizing for cosine similarity
            faiss.normalize_L2(vectors)


            #train index
            if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
                self.index.train(vectors)

            #adding vectors to the index
            self.index.add(vectors)
            self.chunk_ids.extend(chunk_ids)
            
            self.logger.info(f'Added {vectors.shape[0]} vectors to the FAISS index.')
        except ValueError as e:
            self.logger.error(e)

    def search_vectors(self, query_vector: np.ndarray, k: int = 100):
        """
        Searches the FAISS index for the nearest neighbors of a query vector.

        Args:
            query_vector (np.ndarray): The query vector for which to find the nearest neighbors.
            k (int): The number of nearest neighbors to return.

        Returns:
            List[Tuple[int, float]]: A list of (index, distance) tuples for the nearest neighbors.
        """
        try:
            #normalize the query vector
            # query_vector = query_vector / np.linalg.norm(query_vector)
            faiss.normalize_l2(query_vector.reshape(1, -1)) 

            #perform the search
            self.distances, self.indices = self.index.search(query_vector.reshape(1, -1), k)

            #map indices to chunk ids
            self.result_chunk_ids = [self.chunk_ids[i] for i in self.indices[0]]
        except ValueError as e:
            self.logger.error(f"Error searching vectors: {e}")
            raise

class TextProcessor:
    """TextProcessor class for text processing and embedding generation.
    This class provides functionality for processing text documents using SentenceTransformers,
    handling text chunking, and managing stopwords. It initializes with a specified model
    and chunk size for text processing operations.
    Attributes:
        model (SentenceTransformer): The sentence transformer model for text embedding
        stop_words (set): Set of English stopwords from NLTK
        chunk_size (int): Size of text chunks for processing
        logger (Logger): Logger instance for the class
    Example:
        processor = TextProcessor(model_name='all-MiniLM-L6-v2', chunk_size=1000)
    """
    def __init__(self,model_name:str = 'all-MiniLM-L6-v2',chunk_size:int = 1000):
        """
        Initialize the TextProcessor class.
        This class initializes a text processing pipeline with a sentence transformer model
        and necessary configurations for text processing tasks.
        Args:
            model_name (str, optional): Name of the sentence transformer model to be used.
                Defaults to 'all-MiniLM-L6-v2'.
            chunk_size (int, optional): Size of text chunks for processing.
                Defaults to 1000.
        Attributes:
            model: Initialized SentenceTransformer model
            stop_words: Set of English stop words
            chunk_size: Size of text chunks for processing
            logger: Logger instance for the class
        Example:
            processor = TextProcessor(model_name='all-MiniLM-L6-v2', chunk_size=1000)
        """
        
        self.model = SentenceTransformer(model_name)
        self.stop_words = set(stopwords.words('english'))
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
        self.logger.info(f'Initialized text processor with model {model_name} and chunk size {chunk_size}.')

    def process_text(self, doc: Dict) -> Tuple[List[Dict],np.ndarray]:
        """
        Process a document by cleaning text, chunking, and generating embeddings.
        This method performs text processing operations including text cleaning, chunking,
        and generating embeddings for the chunks using a pre-trained model.
        Args:
            doc (Dict): A dictionary containing document information with the following keys:
                - 'content': The text content to be processed
                - 'metadata': Metadata associated with the document
        Returns:
            Tuple[List[Dict], np.ndarray]: A tuple containing:
                - List[Dict]: List of chunk dictionaries with processed text and metadata
                - np.ndarray: Array of chunk embeddings generated by the model
        Raises:
            Exception: If any error occurs during text processing, cleaning, chunking,
                      or embedding generation.
        """

        try:
            #cleaning and chunking
            cleaned_text = self.clean_text(doc['content'])
            chunks = self.chunk_text(cleaned_text)

            #chunk dict
            self.chunk_dicts = [
                self.create_chunk_dict(chunk, doc['metadata'], idx) for idx, chunk in enumerate(chunks)
            ]

            #embeddings
            self.chunk_embeddings = self.model.encode(chunks,show_progress_bar=True)

            return self.chunk_dicts, self.chunk_embeddings

        except Exception as e:
            self.logger.error(f"Error processing text: {e}")
            raise

    def clean_text(self, text: str) -> str:
        """Clean and preprocess text data by removing HTML tags, special characters, stopwords, and normalizing whitespace.
        Args:
            text (str): Raw text input to be cleaned.
        Returns:
            str: Cleaned and preprocessed text with:
                - HTML tags removed
                - Special characters removed (preserving sentence punctuation)
                - Stopwords removed
                - Normalized whitespace
                - Converted to lowercase
        Example:
            >>> cleaner = TextCleaner()
            >>> cleaner.clean_text("<p>Hello, World! This is a test.</p>")
            'hello world test'"""
       
         # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s.,!?]', '', text)

        #remove stopwords
        cleaned_text = ' '.join([word for word in word_tokenize(cleaned_text) if word.lower() not in self.stop_words])

        #normalize whitespace and convert to lowercase
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip().lower()
        return cleaned_text

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits a text into chunks of approximately equal size based on sentence boundaries.
        This method tokenizes the input text into sentences and then groups them into chunks,
        ensuring that each chunk doesn't exceed the specified maximum chunk size (in words).
        The chunks are created by maintaining sentence integrity - sentences are not split
        across chunks.
        Args:
            text (str): The input text to be chunked.
        Returns:
            List[str]: A list of text chunks, where each chunk is a string containing
                complete sentences and approximately adhering to the specified chunk size.
        Attributes:
            sentences (List[str]): Stores the sentences after tokenization.
            chunks (List[str]): Stores the final chunks of text.
            current_chunk (List[str]): Temporarily stores sentences for the current chunk being built.
        Example:
            >>> chunker = TextChunker(chunk_size=100)
            >>> chunks = chunker.chunk_text("Long text here...")
            >>> print(chunks)
            ['First chunk...', 'Second chunk...', ...]
        """
        
        self.sentences = sent_tokenize(text)
        self.chunks =[]
        self.current_chunk=[]
        current_chunk_len = 0
        #splitting text into chunks

        for sentence in self.sentences:
            sentence_len = len(sentence.split())
            if current_chunk_len + sentence_len > self.chunk_size:
                if self.current_chunk:
                    self.chunks.append(' '.join(self.current_chunk))
                self.current_chunk = [sentence]
                current_chunk_len = sentence_len
            else:
                self.current_chunk.append(sentence)
                current_chunk_len += sentence_len
        if self.current_chunk:
            self.chunks.append(' '.join(self.current_chunk))

        return self.chunks
    

class RAGSystem:
    """A class for managing a Retrieval-Augmented Generation (RAG) system.
    This class provides functionality for indexing documents, searching for relevant
    documents based on a query, and generating responses using a pre-trained model.
    It initializes with a specified text processor, FAISS index, and model for response generation.
    Attributes:
        text_processor (TextProcessor): Text processor instance for document processing
        index_manager (FAISSIndex): FAISS index manager for document indexing
        model (SentenceTransformer): Pre-trained model for response generation
        logger (Logger): Logger instance for the class
    Example"""
    def __init__(self,config:Dict):
        """Initialize the RAGManager.
        This class manages the integration of Text Processing, FAISS indexing, SQLite database,
        and Confluence connectivity for a RAG (Retrieval-Augmented Generation) system.
        Args:
            config (Dict): Configuration dictionary containing the following keys:
                - model_name (str): Name of the model to use for text processing
                - chunk_size (int): Size of text chunks for processing
                - vector_size (int): Dimension of vectors for FAISS index
                - index_type (str): Type of FAISS index to use
                - db_path (str): Path to SQLite database
                - confluence_url (str): URL of the Confluence instance
                - confluence_username (str): Username for Confluence authentication
                - confluence_password (str): Password for Confluence authentication
        Attributes:
            text_processor (TextProcessor): Instance of text processing class
            faiss_index (FAISSIndex): Instance of FAISS indexing class
            db_connection (sqlite3.Connection): SQLite database connection
            confluence_client (Confluence): Confluence API client
            logger (logging.Logger): Logger instance for this class
        """

        self.text_processor = TextProcessor(model_name=config['model_name'],chunk_size=config['chunk_size'])
        self.faiss_index = FAISSIndex(vector_size=config['vector_size'],index_type=config['index_type'])
        self.db_connection = sqlite3.connect(config['db_path'])
        self.confluence_client = Confluence(
            url=config['confluence_url'],
            username=config['confluence_username'],
            password=config['confluence_password']
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info('Initialized RAG system with TextProcessor, FAISSIndex, SQLite, and Confluence client.')

    def process_space(self, space_key: str):
        try:
            #fetching pages from Confluence space
            pages = self.confluence_client.get_all_pages_from_space(space_key)
            self.logger.info(f'Fetched {len(pages)} pages from Confluence space {space_key}.')

            #processing and indexing pages
            for page in pages:
                content = self.confluence_client.get_page_by_id(page['id'])
                
                # Create document dictionary
                doc = {
                    'metadata': DocumentMetadata(
                        doc_id=page['id'],
                        source='confluence',
                        title=page['title'],
                        author=content['by']['displayName'],
                        created_at=content['created'],
                        modified_at=content['modified'],
                        tags=[],
                        version=content['version']['number']
                    ),
                    'content': content['body']['storage']['value']
                }

                #proces document and generate embeddings
                chunk_dicts, chunk_embeddings = self.text_processor.process_text(doc)

                #Adding to FAISS index
                self.faiss_index.add_vectors(chunk_embeddings, [chunk['metadata']['page_id'] for chunk in chunk_dicts])

                #adding to SQLite database
                self.store_chunks(chunk_dicts)

            self.logger.info(f'Processed and indexed {len(pages)} pages from Confluence space {space_key}.')
        except Exception as e:
            self.logger.error(f"Error processing Confluence space: {e}")
            raise

    def search(self,query:str,k:int=10) -> List[Dict]:
        """
        Search through document chunks using semantic similarity.
        This method takes a query string, converts it to embeddings, and searches for similar document chunks
        using FAISS indexing. It returns the k most similar chunks with similarity scores.
        Args:
            query (str): The search query text
            k (int, optional): Number of results to return. Defaults to 10.
        Returns:
            List[Dict]: List of dictionaries containing matched document chunks.
                        Each dict includes the chunk text and similarity score.
        Raises:
            Exception: If there is an error during the search process.
        """

        try:
            #process query
            query_embeddings = self.model.encode([query],show_progress_bar=True)[0]
            #search index
            distance,result_chunk_ids = self.faiss_index.search(query_embeddings,k)
            
            #fetch results from database
            results = self.get_chunks(result_chunk_ids)

            #add sim score
            for result, score in zip(results,distance):
                result['score'] = float(score)

            return results
            
        except Exception as e:
            self.logger.error(f"Error searching documents: {e}")
            raise
    def store_chunks(self,chunks:List[Dict]):
        pass









if __name__=="__main__":
    main()
