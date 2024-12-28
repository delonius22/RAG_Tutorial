import pytest
from unittest.mock import Mock, patch
from typing import List, Tuple

# Import the classes we want to test
from src.MDR_RAG import (
    ConfluenceConnector,
    DocumentProcessor,
    EmbeddingModule,
    FAISSIndex,
    ContextualReranker,
    QueryExpansionModule,
    DatabaseManager,
    MultiStageRAG,
    DocumentChunk,
    RankedDocument,
    ConfluenceMetadata,
)

@pytest.fixture
def mock_components():
    return {
        'confluence_conn': Mock(spec=ConfluenceConnector),
        'document_processor': Mock(spec=DocumentProcessor),
        'embedding_module': Mock(spec=EmbeddingModule),
        'db_manager': Mock(spec=DatabaseManager),
        'context_reranker': Mock(spec=ContextualReranker),
        'faiss_index': Mock(spec=FAISSIndex),
        'query_expansion': Mock(spec=QueryExpansionModule),
        'bm25_ranker': Mock()  # BM25Ranking doesn't have a clear spec in the provided code
    }

@pytest.fixture
def rag_system(mock_components):
    return MultiStageRAG(**mock_components)

def test_index_page(rag_system, mock_components):
    page_id = "test_page_id"
    mock_content = "Test content"
    mock_metadata = ConfluenceMetadata(
        page_id=page_id,
        title="Test Page",
        space_key="TEST",
        version=1,
        last_modified="2023-01-01T00:00:00Z",
        author="Test Author",
        labels=["test"]
    )
    mock_chunks = [DocumentChunk(content="Chunk 1", metadata=mock_metadata)]
    mock_chunks_with_embeddings = [DocumentChunk(content="Chunk 1", metadata=mock_metadata, embedding=[0.1, 0.2, 0.3])]

    # Set up mock return values
    mock_components['confluence_conn'].get_page_content.return_value = (mock_content, mock_metadata)
    mock_components['document_processor'].chunk_document.return_value = mock_chunks
    mock_components['document_processor'].get_document_keywords.return_value = ["keyword1", "keyword2"]
    mock_components['embedding_module'].generate_embeddings.return_value = mock_chunks_with_embeddings

    # Call the method
    rag_system.index_page(page_id)

    # Assert that all expected methods were called with correct arguments
    mock_components['confluence_conn'].get_page_content.assert_called_once_with(page_id)
    mock_components['document_processor'].chunk_document.assert_called_once_with(mock_content, mock_metadata)
    mock_components['document_processor'].get_document_keywords.assert_called_once()
    mock_components['embedding_module'].generate_embeddings.assert_called_once_with(mock_chunks)
    mock_components['faiss_index'].add_documents.assert_called_once_with(mock_chunks_with_embeddings)
    mock_components['db_manager'].store_document.assert_called_once()

def test_search(rag_system, mock_components):
    query = "test query"
    k = 5
    mock_expanded_queries = ["test query", "expanded query"]
    mock_query_embedding = [0.1, 0.2, 0.3]
    mock_faiss_results = [(DocumentChunk(content="Result 1", metadata=Mock()), 0.9)]
    mock_bm25_results = [(DocumentChunk(content="Result 1", metadata=Mock()), 0.8)]
    mock_final_reranked_chunks = [DocumentChunk(content="Result 1", metadata=Mock(), relevance_score=0.95)]

    # Set up mock return values
    mock_components['query_expansion'].expand_query.return_value = mock_expanded_queries
    mock_components['embedding_module'].model.encode.return_value = [mock_query_embedding]
    mock_components['faiss_index'].search.return_value = mock_faiss_results
    mock_components['bm25_ranker'].reranking.return_value = mock_bm25_results
    mock_components['context_reranker'].rerank.return_value = mock_final_reranked_chunks

    # Call the method
    results = rag_system.search(query, k)

    # Assert that all expected methods were called with correct arguments
    mock_components['query_expansion'].expand_query.assert_called_once_with(query)
    mock_components['embedding_module'].model.encode.assert_called()
    mock_components['faiss_index'].search.assert_called()
    mock_components['bm25_ranker'].reranking.assert_called_once()
    mock_components['context_reranker'].rerank.assert_called_once()

    # Check that the results are of the correct type and number
    assert isinstance(results, List)
    assert len(results) <= k
    for result in results:
        assert isinstance(result, RankedDocument)

def test_error_handling(rag_system, mock_components):
    # Test index_page method with an invalid page_id
    mock_components['confluence_conn'].get_page_content.side_effect = Exception("Page not found")
    with pytest.raises(Exception):
        rag_system.index_page("invalid_page_id")

    # Test search method with an empty query
    with pytest.raises(ValueError):
        rag_system.search("")

if __name__ == '__main__':
    pytest.main()