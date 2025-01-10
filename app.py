import re
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from src.pre_processing import PDFProcessor
from langchain.prompts import ChatPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()

# Must be the first Streamlit command
st.set_page_config(
    page_title="Summary Generator and Question - Answering",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .citation-box {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .highlight {
        background-color: #FFD1DC;
        padding: 2px 5px;
        border-radius: 3px;
    }
    .st-emotion-cache-1v0mbdj.e115fcil1 {
        width: 100%;
    }
    .document-name {
        font-weight: bold;
    }
    .page-number {
        color: gray;
    }
    .source-content {
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize constants
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "quickstart"

# Cache the PDFProcessor instance
@st.cache_resource
def get_pdf_processor():
    """
    Returns a cached instance of PDFProcessor for handling PDF uploads and processing.
    """
    return PDFProcessor(PINECONE_API_KEY, INDEX_NAME)

# Initialize processor using cached resource
pdf_processor = get_pdf_processor()

# Define Prompt
prompt = ChatPromptTemplate.from_template("""
You are an expert researcher tasked with analyzing the provided documents. Summarize the documents into a detailed and well-cited report that:

- Provides any information that could reasonably address the question, must related to the content of the provided documents.
- Cites key points directly from the provided sources, attributing them accurately.
- Synthesizes information across sources to provide a unified and insightful answer.
- Ensure the response is strictly related to the content of the provided documents and avoids adding unnecessary or unrelated details.

Important:
- Do not include a "References" section.
- Do not include pdf, Page: in response.
- Do not include "Source:".

Question: {question}
Documents: {documents}

Deliver a focused, structured, and well-supported response based on the documents.
""")

@st.cache_resource
def get_llm_chain():
    """
    Returns a cached instance of LLMChain configured with the specified ChatPromptTemplate and LLM.
    """
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"),temperature=0)
    return LLMChain(prompt=prompt, llm=llm)

def highlight_citation_content(summary, citation_content):
    """
    Highlights matching phrases from the summary in the citation content.

    Args:
    summary (str): The generated summary text.
    citation_content (str): Text content from the citation.

    Returns:
    str: Citation content with highlighted matching phrases.
    """
    summary_phrases = []
    summary_words = summary.split()
    
    for length in [5, 4, 3]:  
        for i in range(len(summary_words) - length + 1):
            phrase = ' '.join(summary_words[i:i + length])
            if len(phrase) > 10:  
                summary_phrases.append(phrase)
    
    summary_phrases.sort(key=len, reverse=True)
    highlighted_content = citation_content
    
    for phrase in summary_phrases:
        if phrase.lower() in citation_content.lower():
            escaped_phrase = re.escape(phrase)
            highlighted_content = re.sub(
                rf"\b{escaped_phrase}\b",
                r'<span style="background-color: #FFD1DC; padding: 2px 5px; border-radius: 3px; font-weight: bold; font-style: italic;">\g<0></span>',
                highlighted_content,
                flags=re.IGNORECASE
            )
    
    return highlighted_content

def highlight_relevant_content(query, content, relevant_words, snippet_length=100):
    """
    Highlights relevant words from the query in the document content.

    Args:
    query (str): The user query.
    content (str): The full text of the document.
    relevant_words (set): Set of relevant words to highlight.
    snippet_length (int, optional): Number of characters to show around the matched word. Default is 100.

    Returns:
    str: Highlighted content.
    """
    query_terms = [re.sub(r'[^\w\s]', '', term.lower()) for term in query.split()]
    seen_ranges = set()
    highlighted_content = ""
    
    if relevant_words:
        for term in relevant_words:
            term = term.lower()
            if term in query_terms:
                matches = re.finditer(
                    rf"(\b{re.escape(term)}\b)",
                    content,
                    flags=re.IGNORECASE
                )

                for match in matches:
                    start = max(match.start() - snippet_length, 0)
                    end = min(match.end() + snippet_length, len(content))

                    if any(start <= existing_end and end >= existing_start 
                        for existing_start, existing_end in seen_ranges):
                        continue

                    seen_ranges.add((start, end))
                    snippet = content[start:end]
                    snippet = re.sub(
                        rf"(\b{re.escape(term)}\b)",
                        r"<span style='background-color: #FFD1DC; font-weight: bold; font-style: italic;'>\1</span>",
                        snippet,
                        flags=re.IGNORECASE
                    )

                    highlighted_content += f"... {snippet} ...\n"

        return highlighted_content if highlighted_content else content
    else:
        return ""

def generate_summary_with_citations(query, retrieved_docs):
    """
    Generates a summary of the documents with citations.

    Args:
    query (str): User's question or query.
    retrieved_docs (list): List of document data retrieved from the search.

    Returns:
    tuple: A tuple containing the summary, citation buttons, and source documents.
    """
    processed_docs = []
    seen_pages = set()
    
    for doc in retrieved_docs:
        page_number = int(doc['page_number'])
        if page_number not in seen_pages and doc.get('full_text'):
            seen_pages.add(page_number)
            relevant_words = set(word.lower() for word in doc["full_text"].split() if len(word) > 3 and word.lower() in query.lower())
            highlighted_content = highlight_relevant_content(query, doc['full_text'], relevant_words)
            if highlighted_content.strip():
                processed_docs.append({
                    "text": doc['full_text'],
                    "citation": f"Page {page_number}",
                    "page_number": page_number,
                    "pdf_name": doc['pdf_name'],
                    "image_base64": doc.get('image_base64', ''),
                    "highlighted_content": highlighted_content
                })
    
    if not processed_docs:
        return "Sorry, no meaningful summary could be generated. Please try again.", {}, []

    formatted_docs = "\n".join([
        f"Context from {doc['citation']}:\n{doc['text']}"
        for doc in processed_docs
    ])
    
    try:
        llm_chain = get_llm_chain()
        summary = llm_chain.run({
            "question": query,
            "documents": formatted_docs[:3000]
        })

        if not summary.strip() or "not explicitly mentioned" in summary.lower() or "not directly addressed" in summary.lower() or "do not contain information" in summary.lower():
            return "Sorry, no relevant content found for your query.", {}, []
        
        formatted_answer = ""
        citation_buttons = {}
        vectorizer = TfidfVectorizer()
        doc_texts = [doc['text'] for doc in processed_docs]
        
        if doc_texts:
            paragraphs = [p.strip() for p in summary.split('\n') if p.strip()]
            all_texts = doc_texts + paragraphs
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            for i, paragraph in enumerate(paragraphs):
                if not paragraph:
                    continue
                
                paragraph_cleaned = re.sub(r'\(.*?\)', '', paragraph).strip()
                paragraph_idx = len(doc_texts) + i
                
                similarities = cosine_similarity(
                    tfidf_matrix[paragraph_idx:paragraph_idx+1], 
                    tfidf_matrix[:len(doc_texts)]
                )[0]
                
                matching_docs = [
                    (idx, processed_docs[idx], similarities[idx])
                    for idx in range(len(doc_texts))
                    if similarities[idx] > 0.3
                ]
                
                matching_docs.sort(key=lambda x: x[2], reverse=True)
                
                # Store citations with their corresponding content
                for _, match_doc, _ in matching_docs:
                    citation = match_doc['citation']
                    if citation not in citation_buttons:
                        citation_buttons[citation] = {
                            'text': match_doc['text'],
                            'highlighted_content': match_doc['highlighted_content'],
                            'pdf_name': match_doc['pdf_name'],
                            'page_number': match_doc['page_number']
                        }

                formatted_answer += f"{paragraph_cleaned}"
                if matching_docs:
                    citations_text = "; ".join(set(match_doc['citation'] for _, match_doc, _ in matching_docs))
                    formatted_answer += f" <b>({citations_text})</b>"
                formatted_answer += "\n\n"

        # Apply highlighting to citation content
        highlighted_citations = {}
        for citation, content in citation_buttons.items():
            # Highlight the full text based on the summary
            highlighted_text = highlight_citation_content(summary, content['text'])
            # Create a new dictionary with highlighted content
            highlighted_citations[citation] = {
                'text': highlighted_text,
                'highlighted_content': content['highlighted_content'],  # Keep the query-based highlighting
                'pdf_name': content['pdf_name'],
                'page_number': content['page_number']
            }
        
        # Replace the original citation_buttons with highlighted version
        citation_buttons = highlighted_citations
        
        source_documents = [
            {
                "page_number": doc["page_number"],
                "pdf_name": doc["pdf_name"],
                "highlighted_content": doc["highlighted_content"],
                "full_text": doc["text"],
                "image_base64": doc["image_base64"]
            }
            for doc in processed_docs
        ]

        return formatted_answer.strip(), citation_buttons, source_documents
        
    except Exception as e:
        print(f"Error in generate_summary_with_citations: {e}")
        return "An error occurred while generating the summary. Please try again.", {}, []

@st.cache_data
def process_pdf_data(uploaded_file):
    """Cache the processed PDF data to avoid reprocessing on each query"""
    pdf_processor = get_pdf_processor()    
    num_chunks, total_pages, page_images = pdf_processor.process_file(uploaded_file)
    return {
        'num_chunks': num_chunks,
        'total_pages': total_pages,
        'page_images': page_images,
        'filename': uploaded_file.name
    }

# Main UI Layout
def main():
    st.markdown("""
        <h1 style='text-align: center; color: #1E88E5; margin-bottom: 2rem;'>
            📚 Summary Generator and Question - Answering
        </h1>
        """, unsafe_allow_html=True)
    
    # Store processed data in session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    # Upload Document Section
    st.markdown("### 📤 Upload Document")
    uploaded_file = st.file_uploader("Upload your document", type=["pdf", "docx", "odt", "txt", "doc"], label_visibility="collapsed")
    if uploaded_file:
        # Only process if it's a new file or no file has been processed yet
        current_file = st.session_state.processed_data.get('filename') if st.session_state.processed_data else None
        
        if current_file != uploaded_file.name:
            st.success(f"📄 Uploaded: {uploaded_file.name}")
            with st.spinner("Processing PDF..."):
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    try:
                        # Store processed data in session state
                        st.session_state.processed_data = process_pdf_data(uploaded_file)
                        progress_bar.progress(100)
                        st.info(f"✅ Processed {st.session_state.processed_data['total_pages']} pages")
                    except Exception as e:
                        st.error(f"❌ Error processing PDF: {str(e)}")
                        st.session_state.processed_data = None
    else:
        st.session_state.processed_data = None  # Reset processed data when no file is uploaded
    
    left_column, right_column = st.columns([7, 3])
    citations = {}

    # Content on the left side
    with left_column:
        st.markdown("### 🔍 Ask Questions")
        query = st.text_input("Enter your question about the document:", 
                            placeholder="Enter your question",
                            help="Type your question and press Enter to search")
        
        if query.strip() and st.session_state.processed_data:
            try:
                with st.spinner("🧠 Analyzing document..."):
                    pdf_processor = get_pdf_processor()
                    search_results = pdf_processor.search(query, top_k=3)
                    
                    if search_results:
                        page_images = st.session_state.processed_data['page_images']
                        summary, citations, source_documents = generate_summary_with_citations(query, search_results)
                        
                        if summary == "Sorry, no relevant content found for your query.":
                            st.warning("⚠️ No relevant content found for your query.")
                        else:
                            # Analysis Results Section
                            st.markdown("### 📝 Analysis Results")
                            st.markdown(summary, unsafe_allow_html=True)
                            
                            # Source Documents Section
                            st.subheader("Source Documents:")
                            for doc in source_documents:
                                col1, col2 = st.columns([1, 3])
                                with col1:
                                    page_number = int(doc['page_number']) - 1
                                    if page_number < len(page_images):
                                        page_image = page_images[page_number]
                                        st.image(f"data:image/png;base64,{page_image}", 
                                                caption=f"Page {doc['page_number']}", 
                                                use_container_width=True)
                                    else:
                                        st.warning(f"⚠️ Page image for page {doc['page_number']} not found.")
                                with col2:
                                    st.markdown(f'<span class="document-name">{uploaded_file.name}</span>', 
                                                unsafe_allow_html=True)
                                    st.markdown(f'<span class="page-number">Page {doc["page_number"]}</span>', 
                                                unsafe_allow_html=True)
                                    st.markdown(f'<div class="source-content">{doc["highlighted_content"]}</div>', 
                                                unsafe_allow_html=True)

                                    # Expander for detailed content
                                    with st.expander(f"Content of {doc['pdf_name']} (Page {doc['page_number']})"):
                                        st.write(doc["full_text"])
                                
                                st.markdown("---")  # Separator between documents
                    else:
                        st.warning("⚠️ No relevant content found for your query.")
                        return  # Exit the function if no relevant content is found
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # Content on the right side
    with right_column:
        st.markdown('<div class="source-content" style="background-color: white;">', unsafe_allow_html=True)
        st.markdown("### 📑 Citations")
        
        # Add citation buttons to the response
        for citation, content in citations.items():
            with st.expander(f"Citation Details: {citation}"):
                st.markdown(content["text"], unsafe_allow_html=True)

if __name__ == "__main__":
    main()