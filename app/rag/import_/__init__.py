"""
应用主包 / RAG 能力层 / 导入能力子模块的初始化文件，用于声明包边界与导出约定。
"""
from app.rag.import_.embedding_service import embed_chunks
from app.rag.import_.entry_service import resolve_input_file
from app.rag.import_.index_service import index_chunks
from app.rag.import_.markdown_image_service import enrich_markdown_images
from app.rag.import_.pdf_parse_service import parse_pdf_to_markdown
from app.rag.import_.chunking_templates import (
    ChunkingTemplate,
    ChunkingTemplateConfig,
    ChunkingTemplateRegistry,
    chunk_parsed_document,
    chunk_parsed_document_with_profile,
    default_chunking_template_registry,
    list_chunking_templates,
    suggest_template,
)

__all__ = [
    "embed_chunks",
    "enrich_markdown_images",
    "index_chunks",
    "ChunkingTemplate",
    "ChunkingTemplateConfig",
    "ChunkingTemplateRegistry",
    "chunk_parsed_document",
    "chunk_parsed_document_with_profile",
    "default_chunking_template_registry",
    "list_chunking_templates",
    "parse_pdf_to_markdown",
    "resolve_input_file",
    "suggest_template",
]
