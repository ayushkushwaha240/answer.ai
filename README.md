# AI-Powered Q&A Chrome Extension

A powerful question-answering system that combines Wikipedia and ArXiv knowledge sources with advanced language models to deliver accurate answers directly in your browser via a Chrome extension.

## Overview

This project consists of two main components:
1. **Backend API**: A FastAPI server that processes questions using LangChain tools and Hugging Face models
2. **Chrome Extension**: A browser extension that provides a user interface for asking questions

The system works by:
- Collecting relevant information from Wikipedia and ArXiv based on your question
- Ranking the collected information using the sentence-transformer model
- Extracting a precise answer using BERT's question-answering capabilities

## Features

- **Multi-source Knowledge**: Searches both Wikipedia and ArXiv for comprehensive answers
- **Semantic Ranking**: Uses the sentence-transformer model to identify the most relevant information
- **Precise Answers**: Extracts specific answers rather than just returning passages
- **Browser Integration**: Convenient access through a Chrome extension
- **REST API**: Simple HTTP interface for integration with other applications

## Tech Stack

### Backend
- **FastAPI**: Modern, high-performance web framework
- **LangChain**: Framework for building applications with language models
- **Hugging Face Models**:
  - `sentence-transformers/all-MiniLM-L6-v2` for ranking responses
  - `google-bert/bert-large-uncased-whole-word-masking-finetuned-squad` for question answering
- **Docker**: Containerization for easy deployment

### Frontend
- **Chrome Extension**: Browser-based interface

## Installation

### Backend Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a `.env` file with your Hugging Face API key:
   ```
   hf_key=your_huggingface_api_key
   ```

3. Build and run with Docker:
   ```bash
   docker build -t qa-backend .
   docker run -p 7860:7860 qa-backend
   ```

   Alternatively, run locally:
   ```bash
   pip install -r requirements.txt
   uvicorn app:app --host 0.0.0.0 --port 7860
   ```

### Chrome Extension Installation

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" (toggle in the top-right corner)
3. Click "Load unpacked" and select the `chrome-extension` directory
4. The extension icon should appear in your browser toolbar

## Usage

1. Click on the extension icon in your Chrome toolbar
2. Type your question in the input field
3. Press Enter or click the submit button
4. View the answer in the extension popup

## API Documentation

The backend exposes the following endpoints:

- `GET /`: Health check endpoint
- `POST /ask/`: Question answering endpoint
  - Request body: `{"question": "Your question here?"}`
  - Response: `{"answer": "The answer to your question"}`
