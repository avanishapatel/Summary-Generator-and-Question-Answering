# Summary Generator and Question Answering

## Overview

Summary Generator and Question Answering is an advanced tool designed to extract, summarize, and analyze text from various document formats, including PDF, DOCX, ODT, TXT, and DOC. It enables efficient data extraction, search, and analysis, helping users quickly generate summaries and obtain relevant answers from the contents of their files. This tool enhances productivity by providing automated insights and facilitating data-driven decision-making within documents.

## Features

- Text Summarization: Extracts the most important information from a large text and generates a concise summary.
- Question Answering: Given a text and a question, the model answers the question using relevant information from the input text.

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/avanishapatel/Summary-Generator-and-Question-Answering.git
   cd Summary-Generator-and-Question-Answering
2. **Set Up a Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
3. **Configure Environment Variables**
   ```bash
   PINECONE_API_KEY="your_pinecone_api_key"
   GROQ_API_KEY="your_groq_api_key"
4. **Run the Application**
   ```bash
   streamlit run app.py

## Project Structure
```bash
Summary-Generator-and-Question-Answering/
├── data/                       # Directory for storing PDF files
├── venv/                       # Virtual environment folder
├── src/                        # Source code
│   ├── pre_processing.py       # Module for parsing and processing PDFs
├── app.py                      # Main application logic
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables
└── README.md                   # Project documentation
