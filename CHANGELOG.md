# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2024-01-XX

### Added
- **New Component**: `MixedbreadDocumentParser` for parsing documents from various file formats (PDF, DOCX, PPTX, images)
- Document parsing supports multiple chunking strategies, element type filtering, and rich metadata extraction
- Proper async implementation with concurrent file processing using `asyncio.gather`
- Enhanced error handling with graceful fallback documents
- Comprehensive logging throughout all components

### Changed
- **BREAKING**: Component names simplified by removing "AI" suffix:
  - `MixedbreadAITextEmbedder` → `MixedbreadTextEmbedder`
  - `MixedbreadAIDocumentEmbedder` → `MixedbreadDocumentEmbedder`
  - `MixedbreadAIReranker` → `MixedbreadReranker`
- **BREAKING**: Migrated from `mixedbread_ai` SDK to `mixedbread` SDK
- Improved async/await patterns across all components
- Enhanced error handling with proper logging instead of print statements

### Removed
- **BREAKING**: `LocalMixedbreadRerankV2` component has been removed
  - Use the cloud-based `MixedbreadReranker` instead for all reranking needs

### Fixed
- Async implementations now properly use concurrent processing instead of sequential execution
- Replaced print statements with proper logging using the standard logging module

### Migration Guide
See the [README Migration Guide](./README.md#migration-guide) for detailed upgrade instructions from v2.0.x to v2.1.x.

## [2.0.x] - Previous Versions

Previous versions used the `mixedbread_ai` SDK and included component names with "AI" suffix.