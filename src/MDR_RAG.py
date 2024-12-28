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
    def get_page_content(self, page_id: str) -> Tuple[str, ConfluenceMetadata]:
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


class EmbeddingModule:
    """
    A class for generating embeddings from text chunks using Sentence Transformers.
    This module handles the creation of semantic embeddings for document chunks using 
    pre-trained transformer models from the sentence-transformers library.
    Attributes:
        model (SentenceTransformer): The loaded sentence transformer model used for generating embeddings.
    Args:
        model_name (str, optional): Name/path of the sentence transformer model to use.
            Defaults to "sentence-transformers/all-mpnet-base-v2".
    """

    def __init__(self, model_name:str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize a EmbeddingModule instance with a specified sentence transformer model.
        Args:
            model_name (str, optional): The name or path of the sentence transformer model to use. 
                Defaults to "sentence-transformers/all-mpnet-base-v2".
        Attributes:
            model: The loaded SentenceTransformer model instance used for text embeddings.
        """
        
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Generate embeddings for a list of document chunks.
        This method takes document chunks, processes their content through an embedding model,
        and attaches the resulting embeddings back to each chunk.
        Args:
            chunks (List[DocumentChunk]): A list of DocumentChunk objects containing text content
                to be embedded.
        Returns:
            List[DocumentChunk]: The input chunks with embeddings attached to each chunk.
                The embeddings are stored in the 'embedding' attribute as lists.
        Note:
            - The embedding process is done in batches of 32 for efficiency
            - Progress bar is shown during embedding generation
        """

        texts = [chunk.content for chunk in chunks]

        #embedding in batches
        embeddings = self.model.encode(texts, batch_size = 32, show_progress_bar=True)

        #joining embeddings with chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding.tolist()
        
        return chunks
    

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
        """
        Expands the input query by generating variations using synonyms and hypernyms.
        This method takes a query string and generates multiple variations by:
        1. Finding synonyms for each word in the query using WordNet
        2. Replacing words with their synonyms to create new queries
        3. Finding more general terms (hypernyms) and creating additional variations
        4. Normalizing and limiting the results
        Args:
            query (str): The original search query to expand
        Returns:
            List[str]: A list of expanded query variations, limited to 5 queries maximum.
            The list includes the original query and variations created using synonyms
            and hypernyms.
        Example:
            >>> expander.expand_query("happy dog")
            ['happy dog', 'cheerful dog', 'joyful dog', 'canine dog', 'content dog']
        """

        expanded_queries = [query]
        words = word_tokenize(query)

        #getting synonyms for each word in the query
        for word in words:
            synsets = wordnet.synsets(word, pos=self._get_wordnet_pos(word))
            synonyms = set()
            for synset in synsets[:2]: #limit to first two synsets
                expanded_queries.extend([query.replace(word,lemma.name().replace('_',' ')) for lemma in synset.lemmas()[:2]])#limit to top two synonyms

                #add in more general terms
                hypernyms = synset.hypernyms()
                if hypernyms:
                    expanded_queries.extend([query.replace(word,hyper.lemmas()[0].name().replace['_',' ']) for hyper in hypernyms[:1]])#limit to top hypernyms
        #normalizing
        expanded_queries = list(set(expanded_queries))
        return expanded_queries[:5] #limit to 5 expanded queries

# stage 2 reranking class
class FAISSIndex:
    def __init__(self, dimension: int, index_type: str = "IVFFlat",nlist: int = 100):
        self.dimension=dimension
        self.quantizer= faiss.IndexFlatL2(dimension)

        if index_type == "IVFFlat":
            self.index = faiss.IndexIVFFlat(self.quantizer, dimension, nlist)
        elif index_type == "IVFPQ":
            #product quantization
            m=8 #number of subquantizers
            bits = 8 #number of bits per subquantizer
            self.index = faiss.IndexIVFPQ(self.quantizer, dimension, nlist, m, bits)
        else:
            self.index = faiss.IndexFlatL2(dimension)

        self.is_trained = False
        self.chunk_mapping=[]
    def tain_index(self, embeddings: np.array):
        """
        Trains the FAISS index with the given embeddings.
        This method trains the FAISS index with the provided embeddings.
        Args:
            embeddings (np.array): A 2D NumPy array of shape (n_samples, n_features)
                containing the vector embeddings to train the index.
        Returns:
            None
        """

        if not self.is_trained and isinstane(self.index, faiss.IndexIVF):
            self.index.train(embeddings)
            self.is_trained = True
    def add_documents(self, chunks: List[DocumentChunk]):
        embeddings = np.array([chunk.embedding for chunk in chunks])

        #train index if needed
        self.train_index(embeddings)

        # add to index
        self.index.add(embeddings)
        self.chunk_mapping.extend(chunks)

    def search(self, query_embedding: np.ndarray, k:int=10) -> List[Tuple[DocumentChunk, float]]:
        query_embedding = query_embedding.reshape(1,-1)

        #perform search
        if isinstance(self.index, faiss.IndexIVF):
            self.index.nprobe = min(20, self,index.nlist) # number of clusters to search

        distances, indices = self.index.search(query_embedding, k)

        #adding results to chunks
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx != -1:
                chunk = self.chunk_mapping[idx]
                #changing distance to sim score (1/(1+distance))
                similarity = 1.0 / (1.0 + distance)
                results.append((chunk, similarity))
        return results


# stage 4 reranking class          
class ContextualReranker:
    """
    A class that performs contextual reranking of document chunks based on their content similarity and metadata coherence.
    This reranker considers both the semantic similarity between chunks within a context window and the coherence of their
    metadata to improve the relevance ordering of search results.
    Parameters
    ----------
    window_size : int, optional (default=2)
        The number of chunks to consider on either side of a target chunk when calculating context similarity.
    Methods
    -------
    _calculate_context_similarity(chunks: List[DocumentChunk]) -> Dict[DocumentChunk, float]
        Calculates similarity scores between each chunk and its surrounding context using TF-IDF and cosine similarity.
    _calculate_metadata_coherence(chunks: List[DocumentChunk]) -> Dict[DocumentChunk, float]
        Calculates coherence scores based on metadata similarity between chunks.
    rerank(chunks: List[DocumentChunk]) -> List[Tuple[DocumentChunk, float]]
        Reranks the input chunks by combining original relevance scores with context and metadata coherence scores.
    Notes
    -----
    The final relevance score is a weighted combination of:
    - Original relevance score (40%)
    - Context coherence score (30%)
    - Metadata coherence score (30%)
    These weights can be adjusted based on specific use cases and performance requirements.
    """

    def __init__(self,window_size:int = 2):
        """
        Initialize the MDR_RAG class.
        Args:
            window_size (int, optional): Size of the sliding window used for pattern detection. 
                Defaults to 2.
        Attributes:
            window_size (int): Stored window size for pattern detection algorithms.
        """

        self.window_size = window_size

    def _calculate_context_similarity(self, chunks: List[DocumentChunk]) -> Dict[DocumentChunk, float]:
        """Calculate the semantic similarity between each chunk and its surrounding context.
        This method evaluates how well each chunk fits with its neighboring chunks by:
        1. Creating a context window around each chunk
        2. Computing TF-IDF vectors for the chunk and its context
        3. Calculating cosine similarity between the chunk and its context
        4. Taking the mean similarity as the context coherence score
        Parameters
        ----------
        chunks : List[DocumentChunk]
            List of document chunks to analyze for contextual similarity
        Returns
        -------
        Dict[DocumentChunk, float]
            Dictionary mapping each chunk to its context similarity score (0-1),
            where 1 indicates high coherence with surrounding context
        """

        context_scores = {}

        for i, chunk in enumerate(chunks):
            # get context window around chunk
            start = max(0, i - self.window_size)
            end = min(len(chunks), i + self.window_size + 1)
            context_chunks = chunks[start:i] + chunks[i+1:end]

            # calculate contextual coherence score
            if not context_chunks:
                context_scores[chunk] = 1.0
                continue
            
            # get TF_IDF vectors for chunk and its context
            vectorizer = TfidfVectorizer(stop_words='english')
            all_texts = [chunk.content] +[c.content for c in context_chunks]
            tfidf_matrix = vectorizer.fit_transform(all_texts)

            # calculate average sim with context
            chunk_vector = tfidf_matrix[0:1] #getting query
            context_vectors = tfidf_matrix[1:] #getting context(documents)
            similarities = cosine_similarity(chunk_vector, context_vectors).flatten()
            context_scores[chunk] = np.mean(similarities)

        return context_scores
    
    def _calculate_metadata_coherence(self, chunks: List[DocumentChunk]) -> Dict[DocumentChunk, float]:
        """
        Calculate the metadata coherence score for each chunk in the given list of document chunks.
        The metadata coherence score is determined by comparing each chunk's metadata with all other chunks'
        metadata in the list. The score represents how frequently the same metadata appears across chunks,
        normalized by the total number of chunks.
        Args:
            chunks (List[DocumentChunk]): A list of DocumentChunk objects containing text and metadata.
        Returns:
            Dict[DocumentChunk, float]: A dictionary mapping each DocumentChunk to its metadata coherence score.
                                       Scores range from 0.0 to 1.0, where:
                                       - 1.0 indicates the chunk's metadata matches all other chunks
                                       - 0.0 indicates the chunk's metadata is unique in the set
        Example:
            If 3 chunks have identical metadata, each chunk will receive a score of 1.0
            If a chunk's metadata differs from all others, it will receive a score of 1/n 
            (where n is total number of chunks)
        """

        metadata_scores = {}
        metadata = [chunk.metadata for chunk in chunks]

        for i, chunk in enumerate(chunks):
            # calculate metadata coherence score
            metadata_score = 0.0
            for meta in metadata:
                if chunk.metadata == meta:
                    metadata_score += 1.0

            metadata_scores[chunk] = metadata_score / len(metadata)

        return metadata_scores

    def rerank(self, chunks: List[DocumentChunk]) -> List[Tuple[DocumentChunk, float]]:
        """
        Reranks a list of document chunks based on combined scores from relevance, context similarity, and metadata coherence.
        The final ranking is determined by weighted combination of:
        - Original relevance score (40% weight)
        - Context similarity score (30% weight) 
        - Metadata coherence score (30% weight)
        Args:
            chunks (List[DocumentChunk]): List of document chunks to be reranked
        Returns:
            List[Tuple[DocumentChunk, float]]: Sorted list of tuples containing document chunks and their
            combined relevance scores in descending order
        Note:
            - Returns empty list if input chunks is empty
            - Weights can be adjusted based on performance requirements
        """

        if not chunks:
            return chunks

        #calculate context similitary scores
        context_scores = self._calculate_context_similarity(chunks)
        #calculate metadata coherence scores
        metadata_scores = self._calculate_metadata_coherence(chunks)

        #combine scores
        for chunk in chunks:
            #combine existing relevancy scores with context and metadata scores
            chunk.relevance_score = (
                0.4 * chunk.relevance_score + # original relevance score
                0.3 * context_scores[chunk] + # context coherence score
                0.3 * metadata_scores[chunk] # Metadata coherence score
                # weights can be adjusted based on performance
            )
        return sorted(chunks, key=lambda x: x.relevance_score, reverse=True)

    

# stage 3 reranking class
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

class LogscaleScraper:
    """A web scraper specifically designed for crawling and extracting content from LogScale documentation.
    This class implements an asynchronous web scraper that respects robots.txt rules and rate limits
    while crawling LogScale's documentation pages. It extracts content, breadcrumbs, sections, and
    builds structured metadata for each page.
    Attributes:
        base_url (str): The root URL of the LogScale documentation site. Defaults to "https://library.humio.com".
        rate_limit (float): Minimum time (in seconds) between requests. Defaults to 1.0.
        last_request (float): Timestamp of the last made request.
        client (httpx.AsyncClient): Async HTTP client for making requests.
        robots_parser (RobotsParser): Parser for robots.txt rules.
        visited_urls (set): Set of URLs that have been processed.
    Methods:
        initialize(): Sets up the robots.txt parser.
        crawl(max_pages: int = 100): Main crawling method that processes documentation pages.
        _respect_rate_limit(): Ensures proper timing between requests.
        _can_fetch(url: str): Checks if URL can be crawled per robots.txt.
        _extract_breadcrumb(soup: BeautifulSoup): Extracts page breadcrumb navigation.
        _extract_section(soup: BeautifulSoup): Extracts section information from sidebar.
        _process_page(url: str): Processes a single documentation page.
        _find_documentation_links(url: str): Finds all documentation links on a page.
    """

    def __init__(self, base_url:str="https://library.humio.com", rate_limit:float=1.0):
        """Initialize Logscale crawler.
        Args:
            base_url (str): The base URL to start crawling from. Defaults to "https://library.humio.com".
            rate_limit (float): The minimum time (in seconds) between requests. Defaults to 1.0.
        Attributes:
            base_url (str): The base URL for crawling
            rate_limit (float): Rate limiting delay between requests
            last_request (float): Timestamp of the last request made
            client (httpx.AsyncClient): Async HTTP client for making requests
            robots_parser (urllib.robotparser.RobotFileParser): Parser for robots.txt
            visited_urls (set): Set of URLs already visited during crawling
        """
        
        self.base_url = base_url
        self.rate_limit = rate_limit
        self.last_request = 0.0
        self.client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=30.0,
            headers={"User-Agent": "LogScaleDocBot/1.0"}
        )
        self.robots_parser = None
        self.visited_urls = set()
        
    async def initialize(self):
        """
        Initialize the web scraper by fetching and parsing robots.txt.
        This method performs the following:
        1. Constructs the full URL for robots.txt by joining base_url with "/robots.txt"
        2. Makes an HTTP GET request to fetch the robots.txt content
        3. Parses the robots.txt content into a RobotsParser object
        Returns:
            None
        Raises:
            HTTPError: If there is an error fetching robots.txt
        """

        robots_url = urljoin(self.base_url, "/robots.txt")
        response = await self.client.get(robots_url)
        self.robot_parser = RobotsParser.from_string(response.text, robots_url)
    
    async def _respect_rate_limit(self):
        """
        Ensures rate limiting by enforcing minimum time between requests.
        This method implements a basic rate limiting mechanism by tracking the time 
        between requests and introducing delays when necessary to maintain the 
        specified rate limit.
        Returns:
            None
        Example:
            await self._respect_rate_limit()  # Will pause if requests are too frequent
        """

        now = time.time()
        time_since_last = now - self.last_request
        if time_since_last < self.rate_limit:
            await asyncio.sleep(self.rate_limit - time_since_last)
        self.last_request = time.time()

    def _can_fetch(selfm,url:str) -> bool:
        """
        Check if a URL can be fetched according to the site's robots.txt rules.
        Args:
            url (str): The URL to check for permission to fetch.
        Returns:
            bool: True if the URL can be fetched, False otherwise.
        Examples:
            >>> parser = RobotsParser()
            >>> parser._can_fetch("https://example.com/page")
            True
        """

        return self.robots_parser.can_fetch("LogScaleDocBot/1.0", url)
    
    def _extract_breadcrumb(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract breadcrumb navigation from HTML content using BeautifulSoup.
        The method finds the breadcrumb navigation in an HTML page by looking for a <nav> tag
        with class "breadcrumb" and extracts the text from all <a> tags within it.
        Args:
            soup (BeautifulSoup): A BeautifulSoup object containing the parsed HTML content
        Returns:
            List[str]: A list of breadcrumb items as strings. Returns empty list if no breadcrumb is found.
        """

        breadcrumb = []
        breadcrumb_tag = soup.find("nav", class_="breadcrumb")
        if breadcrumb_tag:
            links = breadcrumb_tag.find_all("a")
            breadcrumb = [link.text.strip() for link in links]
        return breadcrumb
    
    def _extract_section(self, soup: BeautifulSoup) -> str:
        """
        Extracts section information from a BeautifulSoup parsed HTML document.
        This method searches for the active section in a sidebar navigation element by looking
        for the currently selected page and its parent section group.
        Args:
            soup (BeautifulSoup): A BeautifulSoup object containing the parsed HTML document.
        Returns:
            str: The extracted section label text. Returns an empty string if no section is found
                or if the section has no aria-label attribute.
        """
        
        section = ""
        sidebar = soup.find("nav", class_="sidebar")
        if sidebar:
            active = sidebar.find("a",{"aria-current":"page"})
            if active:
                section = activie.find_parent("div",{"role":"group"})
                if section:
                    section = section.get("aria-label","")
        return section
    
    async def _process_page(self,url:str) -> Tuple[str, LogscaleSource]:
        """
        Asynchronously processes a webpage to extract content and metadata.
        This method fetches a webpage, respects robots.txt rules, extracts main content,
        and builds associated metadata including title, breadcrumb, and section information.
        Args:
            url (str): The URL of the webpage to process.
        Returns:
            tuple[str, LogscaleSource]: A tuple containing:
                - str: The cleaned main content text of the webpage
                - LogscaleSource: Metadata object containing source information
                Returns (None, None) if the page cannot be fetched or has no main content.
        Raises:
            Any exceptions from HTTP client requests may be raised.
        Note:
            - Respects rate limiting via _respect_rate_limit()
            - Checks robots.txt rules via _can_fetch()
            - Removes script and style elements from content
            - Extracts metadata including title, breadcrumb, and section
        """

        await self._respect_rate_limit()

        if not self._can_fetch(url):
            return None, None
        
        response = await self.client.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        #pulling page data
        main_content = soup.find("main")
        if not main_content:
            return None, None

        #cleaning data
        for element in main_content.find_all(['script','style']):
            element.decompose()

        content = main_content.get_text(separator=" ",strip=True)

        title = soup.find("h1").text.strip() if soup.find("h1") else ""
        breadcrumb = self._extract_breadcrumb(soup)
        section = self._extract_section(soup)

        #building metadata
        metadata = LogscaleSource(
            source_type='logscale',
            url=url,
            title=title,
            section=section,
            last_crawled=datetime.now(),
            breadcrumb=breadcrumb
        )
        return content, metadata

    async def _find_documentation_links(self, url:str) -> List[str]:
        """
        Asynchronously finds and extracts documentation links from a given URL.
        This method crawls a webpage and collects all links that point to documentation pages,
        ensuring they are within the same base URL domain and haven't been visited before.
        Args:
            url (str): The URL to search for documentation links.
        Returns:
            List[str]: A list of full URLs to documentation pages found on the page.
                    Returns an empty list if the URL cannot be fetched or no documentation links are found.
        Notes:
            - Respects rate limiting through _respect_rate_limit()
            - Checks URL fetchability through _can_fetch()
            - Only collects links containing "/docs/" in their path
            - Deduplicates links using a set before returning
        """

        await self._respect_rate_limit()

        if not self._can_fetch(url):
            return []

        response = await self.client.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        links = set()
        for link in soup.find_all("a",href=True):
            href = link.get("href")
            full_url = urljoin(self.base_url,href)

            #only documentation pages
            if (full_url.startswith(self.base_url) and "/docs/" in full_url not in self.visited_urls):
                links.add(full_url)
        return list(links)
        
    async def crawl(self, max_pages: int = 100) -> List[Tuple[str, LogscaleSource]]:
        """Crawls the Logscale documentation website to extract content and metadata.
        This asynchronous method performs a breadth-first crawl of the Logscale documentation,
        starting from the /docs/ page and following internal documentation links.
        Args:
            max_pages (int, optional): Maximum number of pages to crawl. Defaults to 100.
        Returns:
            List[tuple[str, LogscaleSource]]: A list of tuples containing:
                - str: The extracted content from the page
                - LogscaleSource: Metadata about the source document
        The crawler will stop when either:
            - max_pages is reached
            - There are no more links to visit
            - All discovered links have been visited
        Note:
            Requires initialize() to be called first to set up the browser session.
        """

        await self.initialize()

        docs_url = urljoin(self.base_url,"/docs/")
        to_visit = [docs_url]
        results = []

        while to_visit and len(results) < max_pages:
            url = to_visit.pop(0)
            self.visited_urls.add(url)

            content, metadata = await self._process_page(url)
            if content and metadata:
                results.append((content, metadata))

            nlinks = await self._find_documentation_links(url)
            to_visit.extend([l for l in nlinks if l not in self.visited_urls])
        return results

class DatabaseManager:
    """High-level interface for database operations"""
    
    def __init__(self, connection_string: str):
        """Initialize database manager with appropriate backend"""
        if connection_string.endswith('.db'):
            self.db = SQLiteManager(connection_string)
        else:
            # Parse MySQL connection string
            # Format: mysql://user:pass@host:port/dbname
            parsed = urlparse(connection_string)
            self.db = MySQLManager(
                host=parsed.hostname,
                user=parsed.username,
                password=parsed.password,
                database=parsed.path[1:],  # Remove leading '/'
                port=parsed.port or 3306
            )
    
    def store_document(self, chunk: DocumentChunk):
        """Store a document chunk"""
        return self.db.store_document(chunk)
    
    def get_document(self, doc_id: str) -> Optional[DocumentChunk]:
        """Retrieve a document chunk"""
        return self.db.get_document(doc_id)
    
    def update_document(self, doc_id: str, chunk: DocumentChunk):
        """Update an existing document"""
        return self.db.update_document(doc_id, chunk)
    
    def delete_document(self, doc_id: str):
        """Delete a document"""
        return self.db.delete_document(doc_id)
    
    def search_by_metadata(self, metadata_filters: Dict) -> List[DocumentChunk]:
        """Search documents using metadata filters"""
        return self.db.search_by_metadata(metadata_filters)
    
    def initialize_schema(self):
        """Initialize database schema"""
        return self.db.initialize_schema()

class DatabaseInterface(ABC):
    """Abstract base class defining the interface for database operations.
    This interface provides a standard set of methods that must be implemented
    by any concrete database implementation for document storage and retrieval.
    Methods
    -------
    initialize_scheme()
        Initialize the database schema and required tables/collections.
    store_document(document: DocumentChunk)
        Store a new document chunk in the database.
    get_document(doc_id: str) -> Optional[DocumentChunk]
        Retrieve a document chunk by its unique identifier.
    update_document(doc_id: str, chunk: DocumentChunk)
        Update an existing document chunk in the database.
    delete_document(doc_id: str)
        Remove a document chunk from the database.
    search_by_metadata(metadata_filters: Dict) -> List[DocumentChunk]
        Search and retrieve documents based on metadata criteria.
    """

    @abstractmethod
    def initialize_scheme(self):
        """
        Initialize the scheme for the RAG system.
        This method sets up the initial configuration and parameters for the Retrieval-Augmented Generation (RAG) scheme. 
        Currently implemented as a placeholder with no specific functionality.
        Returns:
            None
        """

        pass

    @abstractmethod
    def store_document(self, document:DocumentChunk):
        """Stores a document chunk and metadata in the database.

        Args:
            document (DocumentChunk): _description_
        """
        pass

    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[DocumentChunk]:
        """Retrieve a document chunk by ID"""
        pass
    
    @abstractmethod
    def update_document(self, doc_id: str, chunk: DocumentChunk):
        """Update an existing document chunk"""
        pass
    
    @abstractmethod
    def delete_document(self, doc_id: str):
        """Delete a document chunk"""
        pass
    
    @abstractmethod
    def search_by_metadata(self, metadata_filters: Dict) -> List[DocumentChunk]:
        """Search documents using metadata filters"""
        pass

class SQLiteManager(DatabaseInterface):
    def __init__(self,db_path: str):
        """
        Initialize the MDR_RAG class.
        Parameters
        ----------
        db_path : str
            Path to the SQLite database file where documents will be stored.
        Notes
        -----
        This constructor initializes the database connection and sets up the required schema.
        The database path is stored as an instance variable and the schema is initialized
        through a call to initialize_schema().
        """

        self.db_path = db_path
        self.initialize_schema()

    @contextmanager
    def get_connection(self):
        """
        Creates and manages a SQLite database connection.
        This method is a generator that yields a connection to the SQLite database
        specified by self.db_path. The connection is automatically closed when the 
        generator is exhausted or an exception occurs.
        Yields:
            sqlite3.Connection: A connection object to interact with the SQLite database.
        Example:
            with self.get_connection() as conn:
                # Use the connection
                cursor = conn.cursor()
        """

        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    def initialize_scheme(self):
        """
        Initializes the database schema by creating necessary tables and indexes.
        This method sets up the following database structure:
        - documents table: Stores document content, source type, embeddings, and timestamps
        - document_metadata table: Stores key-value metadata pairs for documents
        - document_labels table: Stores labels/tags for documents
        - Creates indexes on source_type and metadata key-value pairs
        Tables schema:
            documents:
                - id (INTEGER PRIMARY KEY): Unique identifier for each document
                - content (TEXT): The actual document content
                - source_type (TEXT): Type/origin of the document
                - embedding (BLOB): Vector embedding of the document
                - created_at (TIMESTAMP): Document creation timestamp
                - updated_at (TIMESTAMP): Document last update timestamp
            document_metadata:
                - document_id (INTEGER): Foreign key to documents table
                - key (TEXT): Metadata key
                - value (TEXT): Metadata value
            document_labels:
                - document_id (INTEGER): Foreign key to documents table
                - label (TEXT): Document label/tag
        Returns:
            None
        Raises:
            sqlite3.Error: If there's an error executing SQL statements
        """

        with self.get_connection() as conn:
            #create documents table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOUINCREMENT,
                    content TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            #create metadata table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_metadata (
                    document_id INTEGER,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                    PRIMARY KEY (document_id, key)
                )
            """)

            #create labels table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_labels (
                    document_id INTEGER,
                    label TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                    PRIMARY KEY (document_id, label)
                )
            """)

            #create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source_type ON documents(source_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metadata_key_value ON document_metadata (key,value)")

            conn.commit()

    def _store_metadata(self, conn, doc_id: int, metadata: Union[ConfluenceMetadata, LogscaleSource]):
        """Stores metadata and labels for a document in the database.
        This method stores document metadata and labels in their respective database tables.
        For metadata, it handles primitive types (str, int, float, bool) and stores them in
        the document_metadata table. For labels, it stores each label in the document_labels
        table.
        Args:
            conn: Database connection object for executing SQL statements
            doc_id (int): The ID of the document to associate the metadata with
            metadata (Union[ConfluenceMetadata, LogscaleSource]): Named tuple containing 
                metadata and labels to be stored
        Returns:
            None
        Note:
            - Metadata values must be primitive types (str, int, float, bool)
            - Labels must be provided as a list under the 'labels' key
        """

        metadata_dict = metadata._asdict()

        for key, value in metadata_dict.items():
            if isinstance(value, (str, int, float, bool)):
                conn.execute("INSERT INTO document_metadata (document_id, key, value) VALUES (?, ?, ?)", (doc_id, key, str(value)))
            elif isinstance(value, list) and key == "labels":
                #storing labels
                for label in value:
                    conn.execute("INSERT INTO document_labels (document_id, label) VALUES (?, ?)", (doc_id, label))
    
    def store_document(self, chunk:DocumentChunk):
        """
        Stores a document chunk in the database along with its metadata and embedding.
        Args:
            chunk (DocumentChunk): Document chunk object containing content, metadata and embedding information.
                                 The chunk must have the following attributes:
                                 - content: The text content of the chunk
                                 - metadata: Metadata object with source_type attribute
                                 - embedding: Vector embedding of the chunk (optional)
        Returns:
            int: The database ID of the newly inserted document
        Example:
            doc_id = db.store_document(chunk)
        Note:
            - The document content and metadata are stored in separate tables
            - Embeddings are stored as binary arrays
            - Uses context manager to handle database connections
        """

        with self.get_connection() as conn:
            cursor = conn.cursor()

            #storing main docs
            cursor.execute("INSERT INTO documents (content, source_type,embedding) VALUES (?, ?, ?)", (chunk.content, chunk.metadata.source_type,np.array(chunk.embedding).tobytes() if chunk.embedding else None))

            doc_id = cursor.lastrowid
            self._store_metadata(conn, doc_id, chunk.metadata)
            conn.commit()
            return doc_id
    
    def get_document(self, doc_id:str) -> Optional[DocumentChunk]:
        """
        Retrieves a document from the database by its ID.
        This method fetches a document and its associated metadata from the SQLite database,
        constructing a DocumentChunk object with the document's content, metadata, and embedding.
        Args:
            doc_id (str): The unique identifier of the document to retrieve.
        Returns:
            Optional[DocumentChunk]: A DocumentChunk object containing the document's content,
            metadata (either ConfluenceMetadata or LogscaleSource), and embedding if found;
            None if no document matches the given ID.
        Example:
            doc = db.get_document("doc123")
            if doc:
                print(doc.content)
                print(doc.metadata)
        """

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            content, source_type, embedding_bytes = row

            #getting metadata
            cursor.execute("SELECT key, value FROM document_metadata WHERE document_id = ?", (doc_id,))
            metadata_rows = cursor.fetchall()
            metadata_dict= {key: value for key, value in metadata_rows}

            #getting labels
            cursor.execute("SELECT label FROM document_labels WHERE document_id = ?", (doc_id,))
            labels = [row[0] for row in cursor.fetchall()]
            metadata_dict['labels'] = labels

            # create appropriate metadata object
            if source_type == 'confluence':
                metadata = ConfluenceMetadata(**metadata_dict)
            else:
                metadata = LogscaleSource(**metadata_dict)

            # creating document chunk
            return DocumentChunk(content=content, metadata=metadata, embedding=np.frombuffer(embedding_bytes).tolist() if embedding_bytes else None)


    def update_document(self, doc_id: str, chunk: DocumentChunk):
        """
        Updates an existing document in the database with new content and metadata.
        Args:
            doc_id (str): The unique identifier of the document to update.
            chunk (DocumentChunk): A DocumentChunk object containing the new content, metadata and embedding.
        The method performs the following operations:
        1. Updates the main document content, source type and embedding
        2. Deletes existing metadata and labels associated with the document
        3. Stores the new metadata from the provided chunk
        Note:
            - Requires an active database connection
            - Embeddings are stored as binary data if present
            - All changes are committed to the database
        """

        with self.get_connection() as conn:
            cursor = conn.cursor()

            #update main doc
            cursor.execute("UPDATE documents SET content = ?, source_type = ?, embedding = ? WHERE id = ?", (chunk.content, chunk.metadata.source_type, np.array(chunk.embedding).tobytes() if chunk.embedding else None, doc_id))

            #delete existing metadata
            cursor.execute("DELETE FROM document_metadata WHERE document_id = ?", (doc_id,))
            cursor.execute("DELETE FROM document_labels WHERE document_id = ?", (doc_id,))

            #store new metadata
            self._store_metadata(conn, doc_id, chunk.metadata)
            conn.commit()

class MultiStageRAG:
    def __init__(self, confluence_conn:ConfluenceConnector,document_processor:DocumentProcessor,
    embedding_module:EmbeddingModule, db_manager:DatabaseManager, context_reranker:ContextualReranker,faiss_index:FAISSIndex,
    query_expansion: QueryExpansionModule, bm25_ranker:BM25Ranking):
        """
        Initialize the Multi-Stage RAG system.
        Parameters
        ----------
        confluence_conn : ConfluenceConnector
            A connector object to interact with the Confluence API.
        document_processor : DocumentProcessor
            A document processor object to extract content and metadata from Confluence pages.
        embedding_module : EmbeddingModule
            An embedding module object to generate embeddings for document chunks.
        db_manager : DatabaseManager
            A database manager object to handle document storage and retrieval.
        context_reranker : ContextualReranker
            A contextual reranker object to improve relevance ordering of search results.
        faiss_index : FAISSIndex
            A FAISS index object for fast similarity search of document embeddings.
        query_expansion : QueryExpansionModule
            A query expansion module to enhance search queries with additional terms.
        bm25_ranker : BM25Ranking
            A BM25 ranking object for document retrieval and ranking.
        Notes
        -----
        This constructor initializes the Multi-Stage RAG system with the required components and modules.
        The system is designed to perform multi-stage retrieval and ranking of documents from Confluence.
        """

        self.confluence_conn = confluence_conn
        self.document_processor = document_processor
        self.embedding_module = embedding_module
        self.db_manager = db_manager
        self.context_reranker = context_reranker
        self.faiss_index = faiss_index
        self.query_expansion = query_expansion
        self.bm25_ranker = bm25_ranker
        
    def index_page(self, page_id:str):
        # grabbing and processing content
        content, metadata = self.confluence_conn.get_page_content(page_id)
        chunks= self.document_processor.chunk_document(content,metadata)

        #Extractinf keywords from chunks
        for chunk in chunks:
            keywords = self.document_processor.get_document_keywords(chunk.content)
            chunk.metadata.labels.extend(keywords)

        #generating embeddings
        chunks_with_embeddings = self.embedding_module.generate_embeddings(chunks)

        #adding to faiss index
        self.faiss_index.add_documents(chunks_with_embeddings)

        #adding to database
        for chunk in chunks_with_embeddings:
            self.db_manager.store_document(chunk)
    
    def search(self, query: str, k: int=10) -> List[RankedDocument]:
        #stage 1: Query Expansion
        expanded_queries = self.query_expansion.expand_query(query)

        #stage 2: Faiss Index Search
        query_embedding = self.embedding_module.model.encode([query])[0]
        initial_results =[]

        for expanded_query in expanded_queries:
            expanded_embedding = self.embedding_module.model.encode([expanded_query])[0]
            results = self.faiss_index.search(expanded_embedding,k)
            initial_results.extend(results)

        #cleaning dupes
        initial_res = list(set(initial_results))
        #keepign top k
        initial_res = sorted(initial_res, key=lambda x:x[1], reverse=True)[:k]


        chunks = [result[0] for result in initial_res]
        dense_scores = [result[1] for result in initial_res]

        #stage 3: BM25 Reranking (more precise)
        bm25_results = self.bm25_ranker.reranking(query,chunks)
        chunks = [result[0] for result in bm25_results]
        bm25_scores = [result[1] for result in bm25_results]

        #stage 4: Contextual Reranking

        final_reranked_chunks = self.context_reranker.rerank(chunks)

        #combining scores
        final_res=[]
        for chunk, dense_score, bm25_score in zip(final_reranked_chunks, dense_scores, bm25_scores):
            #getting keyword overlap score
            query_keywords = set(word_tokenize(query.lower()))
            chunk_keywords = set(word_tokenize(chunk.content.lower()))
            keyword_overlap = len(query_keywords.intersection(chunk_keywords)) / len(query_keywords)

            #combining scores with weights
            final_score =(
                0.3 * dense_score + #dense score
                0.3 * bm25_score + #bm25 score
                0.25 * chunk.relevance_score + #contextual score
                0.15 * keyword_overlap #simple keyword overlap
            )

            final_res.append(RankedDocument(
                chunk=chunk,
                final_score=final_score,
                stage_score={
                    "dense":dense_score,
                    "bm25":bm25_score,
                    "context":chunk.relevance_score,
                    "keyword_overlap":keyword_overlap
                }))
        return sorted(final_res, key=lambda x:x.final_score, reverse=True)


if __name__ == "__main__":
    # Initialize components
    confluence_connector = ConfluenceConnector(
        url="https://your-domain.atlassian.net",
        username="your-email@domain.com",
        api_token="your-api-token"
    )
    
    document_processor = DocumentProcessor()
    embedding_module = EmbeddingModule()
    faiss_index = FAISSIndex(dimension=768)  # Dimension matches the embedding model
    contextual_reranker = ContextualReranker(window_size=2)
    query_expansion = QueryExpansionModule()
    db_manager = DatabaseManager("rag_store.db")
    
    # Create RAG system
    rag = MultiStageRAG(
        confluence_connector=confluence_connector,
        document_processor=document_processor,
        embedding_module=embedding_module,
        faiss_index=faiss_index,
        contextual_reranker=contextual_reranker,
        query_expansion=query_expansion,
        db_manager=db_manager
    )
    
    # Index some pages
    confluence_pages = ["page-id-1", "page-id-2"]
    for page_id in confluence_pages:
        rag.index_page(page_id)
    
    # Perform search
    results = rag.search("your search query")
    for result in results:
        print(f"Score: {result.final_score}")
        print(f"Content: {result.chunk.content[:200]}...")
        print(f"Metadata: {result.chunk.metadata}")
        print(f"Stage Scores: {result.stage_scores}")
        print("---")