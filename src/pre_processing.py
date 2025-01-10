import io
import re
import fitz
import torch
import base64
import hashlib
import tempfile
import subprocess
import concurrent.futures
from PIL import Image
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFProcessor:
    def __init__(self, pinecone_api_key, index_name):
        """Initialize the PDFProcessor with a Pinecone API key and index name.

        Args:
            pinecone_api_key (str): API key for Pinecone.
            index_name (str): Name of the Pinecone index to use.
        """
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = index_name
        self.metadata_size_limit = 40000
        self.initialize_index()
        
    def initialize_index(self):
        """Initialize the Pinecone index, creating it if it doesn't already exist."""
        if self.index_name not in [index['name'] for index in self.pc.list_indexes()]:
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        self.index = self.pc.Index(self.index_name)
    
    def generate_file_hash(self, file_content):
        """Generate an MD5 hash for the given file content.

        Args:
            file_content (bytes): Content of the file to hash.

        Returns:
            str: MD5 hash of the file content.
        """
        return hashlib.md5(file_content).hexdigest()
    
    def compress_text(self, text, max_length=5000):
        """Compress text by truncating and removing unnecessary whitespace.

        Args:
            text (str): Input text to compress.
            max_length (int, optional): Maximum length of the compressed text. Defaults to 5000.

        Returns:
            str: Compressed text.
        """
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:max_length]
    
    def get_metadata_size(self, metadata):
        """Calculate the size of metadata in bytes.

        Args:
            metadata (dict): Metadata dictionary.

        Returns:
            int: Size of the metadata in bytes.
        """
        return len(str(metadata).encode('utf-8'))
    
    def optimize_metadata(self, metadata, text_content):
        """Optimize metadata to fit within size limits while preserving essential content.

        Args:
            metadata (dict): Metadata dictionary.
            text_content (str): Text content associated with the metadata.

        Returns:
            dict: Optimized metadata.
        """
        base_metadata = {
            "pdf_name": metadata["pdf_name"],
            "page_number": metadata["page_number"],
            "total_pages": metadata["total_pages"],
            "file_hash": metadata["file_hash"],
            "chunk_index": metadata.get("chunk_index", 0)
        }
        
        optimized = base_metadata.copy()
        current_size = self.get_metadata_size(optimized)
        remaining_space = self.metadata_size_limit - current_size - 100
        
        if remaining_space > 0:
            compressed_text = self.compress_text(text_content, max_length=min(5000, remaining_space // 2))
            optimized["chunk_text"] = compressed_text
            
            remaining_space = self.metadata_size_limit - self.get_metadata_size(optimized) - 100
            if remaining_space > 0:
                compressed_full_text = self.compress_text(metadata.get("full_page_text", ""), 
                                                        max_length=remaining_space)
                optimized["full_page_text"] = compressed_full_text
        
        return optimized
    
    def extract_page_image(self, page):
        """
        Generate a base64-encoded image for a specific page of a PDF in original clarity.

        Args:
            page (fitz.Page): The page object from PyMuPDF.

        Returns:
            str: Base64-encoded image of the specified PDF page.
        """
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        return base64.b64encode(img_byte_arr.read()).decode("utf-8")
    
    def convert_to_pdf(self, uploaded_file):
        """Convert various document formats to PDF using LibreOffice.
        
        Args:
            uploaded_file: The uploaded file object.
            
        Returns:
            bytes: PDF content as bytes.
        """
        
        # If it's already a PDF, return the content directly
        if uploaded_file.name.lower().endswith('.pdf'):
            return uploaded_file.read()
        
        # Create a temporary directory for the conversion
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Save the uploaded file with its original name
                temp_path = Path(temp_dir) / uploaded_file.name
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.read())
                
                # Convert to PDF using LibreOffice
                result = subprocess.run([
                    'libreoffice',
                    '--headless',
                    '--convert-to', 
                    'pdf',
                    '--outdir',
                    temp_dir,
                    str(temp_path)
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise RuntimeError(f"LibreOffice conversion failed: {result.stderr}")
                
                # Get the output PDF path
                pdf_path = temp_path.with_suffix('.pdf')
                
                if not pdf_path.exists():
                    raise FileNotFoundError("PDF file was not created")
                
                # Read and return the PDF content
                with open(pdf_path, 'rb') as f:
                    return f.read()
                    
            except Exception as e:
                raise RuntimeError(f"PDF conversion failed: {str(e)}")

    def process_file(self, uploaded_file):
        """Process a file by converting it to PDF if necessary and then processing the PDF.

        Args:
            uploaded_file: The uploaded file object.

        Returns:
            tuple: Number of vectors upserted, total number of pages, and a list of page images.
        """
        pdf_content = self.convert_to_pdf(uploaded_file)
        return self.process_pdf(io.BytesIO(pdf_content))

    def process_pdf(self, uploaded_file):
        """Process a PDF file by extracting text, images, and metadata, and storing it in Pinecone.

        Args:
            uploaded_file: The uploaded PDF file object.

        Returns:
            tuple: Number of vectors upserted, total number of pages, and a list of page images.
        """
        # Clear the existing index before processing the new PDF
        self.clear_index()

        file_content = uploaded_file.read()
        file_hash = self.generate_file_hash(file_content)
        doc = fitz.open(stream=file_content, filetype="pdf")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        vectors_to_upsert = []
        page_images = []
        
        def process_page(page_num):
            page = doc[page_num]
            page_text = page.get_text()
            page_image = self.extract_page_image(page)
            
            base_metadata = {
                "pdf_name": uploaded_file.name if hasattr(uploaded_file, 'name') else 'uploaded_file',
                "page_number": page_num + 1,
                "total_pages": len(doc),
                "file_hash": file_hash
            }
            
            # Store text metadata
            text_metadata = base_metadata.copy()
            text_metadata.update({
                "content_type": "text",
                "full_page_text": page_text
            })
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            text_chunks = text_splitter.split_text(page_text)
            
            # Create embeddings for chunks
            page_vectors = []
            for chunk_idx, chunk in enumerate(text_chunks):
                embedding = model.encode(chunk, convert_to_tensor=True).cpu().numpy()
                
                chunk_metadata = text_metadata.copy()
                chunk_metadata.update({
                    "chunk_index": chunk_idx,
                    "chunk_text": chunk
                })
                
                vector_id = f"{file_hash}_p{page_num + 1}_c{chunk_idx}"
                page_vectors.append((vector_id, embedding, chunk_metadata))
            
            return page_image, page_vectors
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_page, range(len(doc))))
        
        for page_image, page_vectors in results:
            page_images.append(page_image)
            vectors_to_upsert.extend(page_vectors)
        
        # Batch upsert vectors
        self.batch_upsert(vectors_to_upsert, batch_size=20)
        
        return len(vectors_to_upsert), len(doc), page_images
    
    def clear_index(self):
        """Clear the existing index."""
        try:
            self.index.delete(delete_all=True)
        except Exception as e:
            print(f"Error clearing index: {str(e)}")
    
    def batch_upsert(self, vectors, batch_size=50):
        """Upsert vectors in small batches.

        Args:
            vectors (list): List of vectors to upsert.
            batch_size (int, optional): Number of vectors in each batch. Defaults to 50.
        """
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            try:
                self.index.upsert(vectors=batch)
            except Exception as e:
                print(f"Error upserting batch {i//batch_size}: {str(e)}")
                if batch_size > 10:
                    self._batch_upsert(batch, batch_size=batch_size//2)
                else:
                    raise e
    
    def search(self, query_text, top_k=5):
        """Search for text in the Pinecone index.

        Args:
            query_text (str): Query string to search for.
            top_k (int, optional): Number of top results to return. Defaults to 5.

        Returns:
            list: Processed search results with metadata and highlighted text.
        """
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode(query_text).tolist()
        
        search_results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter={"content_type": "text"}
        )
        
        processed_results = []
        seen_pages = set()
        query_terms = set(query_text.lower().split())
        
        for match in search_results.matches:
            if not match.metadata:
                continue
                
            page_number = match.metadata.get('page_number')
            if page_number in seen_pages:
                continue
            
            seen_pages.add(page_number)
            
            full_text = match.metadata.get('full_page_text', '')
            
            result = {
                'pdf_name': match.metadata.get('pdf_name'),
                'page_number': page_number,
                'total_pages': match.metadata.get('total_pages'),
                'full_text': full_text,
                'score': match.score,
                'query_term_locations': self.find_query_term_locations(full_text, query_terms)
            }
            processed_results.append(result)
            
        processed_results.sort(key=lambda x: (-x['score'], x['page_number']))
        return processed_results
    
    def find_query_term_locations(self, text, query_terms):
        """Find locations of query terms in text.

        Args:
            text (str): Text to search for query terms.
            query_terms (set): Set of query terms to find.

        Returns:
            list: Locations of query terms in the text with context.
        """
        locations = []
        text_lower = text.lower()
        for term in query_terms:
            term_lower = term.lower()
            start = 0
            while True:
                pos = text_lower.find(term_lower, start)
                if pos == -1:
                    break
                locations.append({
                    'term': term,
                    'position': pos,
                    'context': text[max(0, pos-50):min(len(text), pos+len(term)+50)]
                })
                start = pos + 1
        return sorted(locations, key=lambda x: x['position'])