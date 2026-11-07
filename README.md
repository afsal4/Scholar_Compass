<div align="center">

# 🧭 Scholar Compass

**An AI-powered assistant to discover relevant research papers and understand them through PDF chat.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Django](https://img.shields.io/badge/Django-5.1-092E20?style=flat-square&logo=django&logoColor=white)
![Gemini](https://img.shields.io/badge/Google_Gemini-8E75B2?style=flat-square&logo=googlegemini&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector_Search-00A98F?style=flat-square)

</div>

## 🔍 About

Scholar Compass helps researchers find academic papers that match their interests. Describe a research topic in a conversation, refine the requirements with the AI assistant, and retrieve relevant papers from a research-paper collection. Once you choose a paper, you can read its PDF and ask questions about its content.

## ✨ Features

| | Feature | Description |
|---|---|---|
| 💬 | **Research assistant** | Uses Google Gemini to understand your research interests and turn them into a focused search query. |
| 🔎 | **Multiple search methods** | Supports context-based vector search, hybrid search, and TF-IDF search. |
| 📚 | **Paper recommendations** | Retrieves relevant paper titles and links to their arXiv PDFs. |
| 📄 | **PDF viewer** | Opens the selected paper in an integrated PDF reader. |
| 🤖 | **PDF chat** | Ask questions about a paper using either FAISS-based RAG or the ChatPDF API. |
| 🕘 | **Chat history** | Stores conversation messages by session in the application database. |

## ⚙️ How It Works

1. **Describe:** Explain your research topic and requirements to the Gemini assistant.
2. **Refine:** The assistant asks follow-up questions and creates a focused search query.
3. **Retrieve:** Choose a search method to find relevant papers from Pinecone.
4. **Explore:** Open a recommended paper, read its PDF, and ask questions using RAG or ChatPDF.

### Search methods

- **Context Search:** Semantic similarity search using Google embeddings and a Pinecone vector index.
- **Hybrid Search:** Combines dense embeddings with BM25 sparse retrieval.
- **TF-IDF Search:** Uses a TF-IDF vectorizer to retrieve matching papers from a Pinecone index.

## 🛠️ Tech Stack

| Purpose | Tools |
|---|---|
| Web application | Django, Bootstrap |
| AI assistant | Google Gemini, LangChain |
| Vector search | Pinecone, FAISS |
| Text retrieval | BM25, scikit-learn TF-IDF |
| PDF processing | PyPDF, PDF.js |
| Database | PostgreSQL |

## 🚀 Getting Started

### Requirements

- Python 3.10+
- PostgreSQL
- Google Gemini API key
- Pinecone API key and preconfigured indexes
- ChatPDF API key (the current code reads this key at startup, even when using RAG mode)

### Installation

```bash
git clone https://github.com/afsal4/Scholar_Compass.git
cd Scholar_Compass

python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Linux / macOS
# source venv/bin/activate

pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
CHATPDF_API_KEY=your_chatpdf_api_key

DB_NAME=scholar_compass
DB_USER=your_postgres_user
DB_PASS=your_postgres_password
DB_HOST=localhost
```

Create the PostgreSQL database and make sure the database server is running. The current Django settings use PostgreSQL on port `5432`. The Django secret key and `DEBUG` setting are currently hard-coded in `settings.py`; move the secret key to an environment variable, disable debug mode, and configure allowed hosts before deploying publicly.

The application expects the following Pinecone indexes to be available and populated with research-paper data:

| Search method | Pinecone index |
|---|---|
| Context Search | `recomentation` |
| Hybrid Search | `recommendation-hybrid-test` |
| TF-IDF Search | `recommendation-tfidf-bigram` |

The `data/` directory must also contain the retrieval assets used by the application:

- `tf_idf_vectorizer.pkl`
- `bm25_data.json`

### Run the application

```bash
python manage.py migrate
python manage.py runserver
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser.

## 📁 Project Structure

```text
Scholar_Compass/
├── data/                 # TF-IDF vectorizer and BM25 data
├── home/                 # Django app, views, prompts, and models
├── scholar_compass/      # Django project settings and URL configuration
├── templates/            # Research chat and PDF viewer templates
├── tmp_pdf_vectorstore/  # Local FAISS store created for PDF RAG
├── manage.py
└── requirements.txt
```

<div align="center">

**Built to make finding and understanding research papers easier.**

</div>
