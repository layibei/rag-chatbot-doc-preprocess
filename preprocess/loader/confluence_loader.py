from typing import List, Optional
from langchain_community.document_loaders import ConfluenceLoader as LangChainConfluenceLoader
from langchain_community.document_loaders.confluence import ContentFormat
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from urllib.parse import urlparse, parse_qs
import requests
from base64 import b64encode
import re
import json
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from preprocess.loader.base_loader import DocumentLoader


class ConfluenceLoader(DocumentLoader):
    def load(self, url: str) -> List[Document]:
        """Load and process Confluence page with enhanced content preservation"""
        try:
            self.logger.info(f"Loading Confluence page: {url}")
            loader = self.get_loader(url)
            if not loader:
                raise ValueError(f"Failed to create loader for Confluence URL: {url}")

            # Extract page ID and get content
            page_id = self._extract_page_id(url)
            if not page_id:
                raise ValueError(f"Could not extract page ID from URL: {url}")
            self.logger.info(f"Page ID: {page_id}, url:{url}")

            # Get both VIEW and STORAGE formats for better content processing
            view_docs = self._load_with_format(loader, page_id, ContentFormat.VIEW)
            storage_docs = self._load_with_format(loader, page_id, ContentFormat.STORAGE)

            # Combine and enhance content
            enhanced_docs = self._enhance_content(view_docs, storage_docs)
            self.logger.info(
                f"Enhanced content: {enhanced_docs}, view doc count:{len(view_docs)}, storage doc count:{len(storage_docs)}")

            # Split documents while preserving structure
            splitter = self.get_splitter(enhanced_docs)
            split_docs = splitter.split_documents(enhanced_docs)

            # Post-process split documents to maintain context
            return self._post_process_documents(split_docs, url)

        except Exception as e:
            self.logger.error(f"Failed to load Confluence page: {url}, Error: {str(e)}")
            raise

    def _load_with_format(self, loader: BaseLoader, page_id: str, format: ContentFormat) -> List[Document]:
        """Load content in specific format"""
        loader.content_format = format
        if format == ContentFormat.VIEW:
            loader.keep_markdown_format = True
        else:
            loader.keep_markdown_format = False
        return loader.load(page_ids=[page_id])

    def _enhance_content(self, view_docs: List[Document], storage_docs: List[Document]) -> List[Document]:
        """Enhance document content by combining VIEW and STORAGE formats"""
        enhanced_docs = []

        for idx, (view_doc, storage_doc) in enumerate(zip(view_docs, storage_docs)):
            try:
                # Check if content is already markdown
                if self._is_markdown_content(view_doc.page_content):
                    # Convert markdown to plain text for embedding
                    plain_text = self._extract_plain_text(view_doc.page_content)
                    enhanced_doc = Document(
                        page_content=plain_text,  # Plain text for embedding
                        metadata={
                            **view_doc.metadata,
                            "content_type": "markdown",
                            "markdown_content": view_doc.page_content  # Store original markdown
                        }
                    )
                else:
                    # Process HTML content
                    view_soup = BeautifulSoup(view_doc.page_content, 'html.parser')
                    storage_soup = BeautifulSoup(storage_doc.page_content, 'html.parser')

                    # Extract plain text for embedding
                    plain_text = ' '.join([
                        p.get_text(strip=True)
                        for p in view_soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th'])
                    ])

                    # Process structured content for markdown
                    enhanced_content = []
                    tables = self._process_tables(view_soup)
                    code_blocks = self._process_code_blocks(storage_soup)
                    diagrams = self._process_diagrams(storage_soup)
                    text_content = self._process_text_content(view_soup)

                    enhanced_content.extend(tables + code_blocks + diagrams + text_content)

                    enhanced_doc = Document(
                        page_content=plain_text,  # Plain text for embedding
                        metadata={
                            **view_doc.metadata,
                            "content_type": "plain_text",
                            "markdown_content": "\n\n".join(enhanced_content)  # Store formatted content
                        }
                    )

                enhanced_docs.append(enhanced_doc)
                self.logger.debug(f"Document {idx} processed - Plain text length: {len(enhanced_doc.page_content)}, "
                                  f"Markdown length: {len(enhanced_doc.metadata.get('markdown_content', ''))}")

            except Exception as e:
                self.logger.error(f"Error enhancing document {idx}: {str(e)}")
                enhanced_docs.append(view_doc)

        return enhanced_docs

    def _is_markdown_content(self, content: str) -> bool:
        """Check if content is already in markdown format"""
        markdown_patterns = [
            r'^#{1,6}\s',  # Headers
            r'```',  # Code blocks
            r'\|.*\|.*\|',  # Tables
            r'^\s*[-*+]\s',  # Lists
            r'^\d+\.\s'  # Numbered lists
        ]

        return any(re.search(pattern, content, flags=re.MULTILINE) for pattern in markdown_patterns)

    def _extract_plain_text(self, markdown_content: str) -> str:
        """Extract clean plain text from markdown content, removing all formatting"""
        try:
            # Remove code blocks and their content
            text = re.sub(r'```[\s\S]*?```', '', markdown_content)
            text = re.sub(r'`[^`]+`', '', text)
            
            # Process tables - extract meaningful content
            def process_table_content(table_match):
                lines = table_match.group(0).split('\n')
                content = []
                for line in lines:
                    # Skip separator lines
                    if re.match(r'\|[\s\-:|]+\|', line):
                        continue
                    # Extract cell content without | symbols
                    cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                    if cells:
                        content.append(' '.join(cells))
                return ' '.join(content)
            
            # Replace tables with their processed content
            text = re.sub(r'\|[^\n]+\|[\s\S]*?(?=\n\s*\n|\Z)', process_table_content, text)
            
            # Remove remaining table markers
            text = re.sub(r'\[TABLE\][\s\S]*?\[/TABLE\]', '', text)
            
            # Remove diagrams and charts but keep any text descriptions
            text = re.sub(r'\[DIAGRAM:.*?\][\s\S]*?\[/DIAGRAM\]', '', text)
            text = re.sub(r'```(?:mermaid|plantuml)[\s\S]*?```', '', text)
            
            # Process images and links - keep alt text and link text
            text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)  # Images
            text = re.sub(r'\[IMAGE:\s*([^\]]*)\]\([^)]+\)', r'\1', text)
            text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Links
            
            # Remove formatting while keeping content
            text = re.sub(r'^#{1,6}\s+(.*)$', r'\1', text, flags=re.MULTILINE)  # Headers
            text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)  # Unordered lists
            text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # Ordered lists
            text = re.sub(r'[*_~]{1,2}([^*_~]+)[*_~]{1,2}', r'\1', text)  # Emphasis
            text = re.sub(r'<!--[\s\S]*?-->', '', text)  # Comments
            text = re.sub(r'^\s*>\s+', '', text, flags=re.MULTILINE)  # Blockquotes
            text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)  # Horizontal rules
            
            # Clean up whitespace
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r'\s+', ' ', text)
            
            self.logger.debug(f"Plain text extraction - Original length: {len(markdown_content)}, "
                             f"Extracted length: {len(text)}")
            
            return text.strip()
            
        except Exception as e:
            self.logger.error(f"Error extracting plain text from markdown: {str(e)}")
            return ' '.join(markdown_content.split())

    def _process_tables(self, soup: BeautifulSoup) -> List[str]:
        """Extract and format tables"""
        table_contents = []
        for table in soup.find_all('table'):
            rows = []
            # Process headers
            headers = [th.get_text(strip=True) for th in table.find_all('th')]
            if headers:
                rows.append("| " + " | ".join(headers) + " |")
                rows.append("|" + "|".join(["---"] * len(headers)) + "|")

            # Process rows
            for row in table.find_all('tr'):
                cells = [td.get_text(strip=True) for td in row.find_all('td')]
                if cells:
                    rows.append("| " + " | ".join(cells) + " |")

            if rows:
                table_contents.append("\n".join(rows))

        return [f"[TABLE]\n{content}\n[/TABLE]" for content in table_contents]

    def _process_code_blocks(self, soup: BeautifulSoup) -> List[str]:
        """Extract and format code blocks"""
        code_contents = []
        for code in soup.find_all(['code', 'pre']):
            # Use markdownify for better code block conversion
            markdown_code = md(str(code), code_language="auto")
            if markdown_code.strip():
                code_contents.append(markdown_code.strip())
        return code_contents

    def _process_diagrams(self, soup: BeautifulSoup) -> List[str]:
        """Extract diagram and image information"""
        diagram_contents = []
        for diagram in soup.find_all(['ac:structured-macro', 'img']):
            if diagram.name == 'ac:structured-macro':
                # Handle Confluence diagrams (like Mermaid or Draw.io)
                diagram_type = diagram.get('ac:name', 'unknown')
                diagram_data = diagram.find('ac:plain-text-body')
                if diagram_data:
                    diagram_contents.append(f"[DIAGRAM:{diagram_type}]\n{diagram_data.get_text()}\n[/DIAGRAM]")
            else:
                # Handle regular images
                alt_text = diagram.get('alt', '')
                src = diagram.get('src', '')
                diagram_contents.append(f"[IMAGE: {alt_text}]({src})")
        return diagram_contents

    def _process_text_content(self, soup: BeautifulSoup) -> List[str]:
        """Process regular text content with structure preservation"""
        text_contents = []

        # Convert HTML to Markdown while preserving structure
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol']):
            # Convert the element to markdown
            markdown_text = md(str(element), heading_style="atx")
            if markdown_text.strip():
                text_contents.append(markdown_text.strip())

        return text_contents

    def _post_process_documents(self, documents: List[Document], url: str) -> List[Document]:
        """Post-process split documents to maintain context and structure"""
        processed_docs = []
        for i, doc in enumerate(documents):
            # Add position metadata
            doc.metadata.update({
                "chunk_index": i,
                "total_chunks": len(documents),
                "source_url": url
            })
            processed_docs.append(doc)
        self.logger.info(f"Post-processed {len(processed_docs)} documents")
        return processed_docs

    def get_loader(self, url: str) -> BaseLoader:
        confluence_url = self.base_config.get_embedding_config("confluence.url")
        username = self.base_config.get_embedding_config("confluence.username")
        api_key = self.base_config.get_embedding_config("confluence.api_key")

        return LangChainConfluenceLoader(
            url=confluence_url,
            username=username,
            api_key=api_key,
            # content_format=ContentFormat.EXPORT_VIEW,
            # keep_markdown_format=True,
            keep_newlines=True
        )

    def get_splitter(self, documents: List[Document]) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.get_trunk_size(),
            chunk_overlap=self.get_overlap()
        )

    def is_supported_file_extension(self, file_path: str) -> bool:
        # Confluence pages don't have file extensions to check
        return True

    def _extract_page_id(self, url: str) -> str:
        """Extract page ID from Confluence URL"""
        try:
            parsed = urlparse(url)

            # 1. Check query parameter 'pageId'
            if 'pageId' in parsed.query:
                return parse_qs(parsed.query)['pageId'][0]

            # 2. Check for pages/viewpage.action?pageId format
            if 'viewpage.action' in parsed.path and 'pageId' in parsed.query:
                return parse_qs(parsed.query)['pageId'][0]

            # 3. Check for wiki/spaces format (Cloud)
            if '/wiki/spaces/' in parsed.path:
                path_segments = parsed.path.split('/')
                try:
                    # Find the 'pages' index and get the next segment
                    pages_index = path_segments.index('pages')
                    if len(path_segments) > pages_index + 1:
                        page_id = path_segments[pages_index + 1]
                        if page_id.isdigit():
                            return page_id
                except ValueError:
                    pass

            # 4. Check for display/SPACE/Page+Title format
            if '/display/' in parsed.path:
                # Use the Confluence API to look up page by title
                space_key = parsed.path.split('/display/')[1].split('/')[0]
                page_title = parsed.path.split('/')[-1].replace('+', ' ')

                confluence_url = self.base_config.get_embedding_config("confluence.url")
                username = self.base_config.get_embedding_config("confluence.username")
                api_key = self.base_config.get_embedding_config("confluence.api_key")

                # Use Confluence REST API to get page ID by title
                api_url = f"{confluence_url}/rest/api/content"
                params = {
                    "spaceKey": space_key,
                    "title": page_title,
                    "expand": "version"
                }
                headers = {
                    "Authorization": f"Basic {b64encode(f'{username}:{api_key}'.encode()).decode()}"
                }

                response = requests.get(api_url, params=params, headers=headers)
                if response.status_code == 200:
                    results = response.json()['results']
                    if results:
                        return results[0]['id']

            # 5. Check numeric ID at end of path
            path_parts = parsed.path.split('/')
            if path_parts[-1].isdigit():
                return path_parts[-1]

            # 6. Check for pages/{pageId} format
            if '/pages/' in parsed.path:
                page_segment = parsed.path.split('/pages/')[-1]
                if '/' in page_segment:
                    potential_id = page_segment.split('/')[1]
                    if potential_id.isdigit():
                        return potential_id

            self.logger.error(f"Could not extract page ID from URL using any known format: {url}")
            return None

        except Exception as e:
            self.logger.error(f"Failed to extract page ID from URL: {url}, Error: {str(e)}")
            raise ValueError(f"Invalid Confluence URL format: {url}")
