from django.shortcuts import render, redirect, HttpResponse
from django.apps import apps
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_pinecone import PineconeVectorStore
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone
from .prompts import template_t1, manual_prompt, react_prompt
import os
import requests
import io
import re 
import joblib
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
load_dotenv()

INDEX_NAME = 'recomentation'
LINK = 'https://arxiv.org/pdf/'
SIMILARITY_THRESHOLD = 0.67
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 40

pinecone_api_key = os.getenv('PINECONE_API_KEY')

if 'GOOGLE_API_KEY' not in os.environ:
    os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
if 'PINECONE_API_KEY' not in os.environ:
    os.environ['PINECONE_API_KEY'] = pinecone_api_key

# vectorizer loading tfidf 
vectorize = joblib.load('data/tf_idf_vectorizer.pkl')

print('finished vector loading')
headers = {
  'x-api-key': os.environ['CHATPDF_API_KEY'],
  'Content-Type': 'application/json'
}
chatpdf_messages = []

chat_messages = []
chat_model = ChatGoogleGenerativeAI(model='gemini-pro', convert_system_message_to_human=True, stop=['A:'])

pdf_chat_messages = []
pdf_chat_model = ChatGoogleGenerativeAI(model='gemini-pro', convert_system_message_to_human=True)


# custom_filters.py
import markdown
from django import template
from django.utils.safestring import mark_safe

register = template.Library()

def get_markdown(text):
    markdown_text = text
    html_content = markdown.markdown(markdown_text, extensions=['extra', 'codehilite', 'toc'])
    return mark_safe(html_content) 

def clean_text(text):
    text = text.replace('Answer:', '') if 'Answer:' in text else text 
    text = text.replace('*', '') if '*' in text else text 
    cleaned_string = re.sub(r'\(.*?\)', '', text)
    cleaned_string = ' '.join(cleaned_string.split())
    return cleaned_string


def get_chat_message(text):
    print(text)
    parser = StrOutputParser()
    chain = template_t1 | chat_model | parser
    with_message_history = RunnableWithMessageHistory(chain, get_session)
    config = {'configurable': {'session_id': 'sgd'}}
    chat_messages.append(HumanMessage(text))
    ai_message = with_message_history.invoke(chat_messages, config=config)
    ai_message = clean_text(ai_message)
    chat_messages.append(AIMessage(ai_message))

def get_session(session_id):
    return ChatMessageHistory()

def get_correct_ids(similar_docs):
    ''''Return dictionary of id and score after correction of id'''
    res_id = [{i[0].metadata['id']: [i[1], i[0].metadata['title']]} for i in similar_docs]
    corrected_ids = {}
    for documents in res_id:
        doc_id = list(documents.keys())[0]
        tmp_id = doc_id
        tmp_id_splits = str(tmp_id).split('.')
        while len(tmp_id_splits[0]) < 4:
            tmp_id_splits[0] = '0' + tmp_id_splits[0]

        while len(tmp_id_splits[1]) < 4:
            tmp_id_splits[1] = tmp_id_splits[1] + '0'
        tmp_id = '.'.join(tmp_id_splits)
        corrected_ids[tmp_id] = documents[doc_id]
    return corrected_ids

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vectorstore = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
    vectorstore.save_local('tmp_pdf_vectorstore')
    return vectorstore

def get_tf_idf_res(query):
    print('TFIDF Searching....')

    index_name = 'recommendation-tfidf-bigram'
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

    query_vec = vectorize.transform([query]).toarray()
    query_vec = query_vec.squeeze()
    val = [float(i) for i in query_vec]

    recommendations = index.query(vector=val, top_k=4, include_metadata=True)

    pdf_links = []
    titles = []
    for results in recommendations['matches']:
        id = results['metadata']['id']
        title = results['metadata']['title']
        pdf_links.append(LINK+str(id))
        titles.append(title)
    return pdf_links, titles


def get_hybrid_res(query):
    print('Hybrid Searching...')
    index_name = 'recommendation-hybrid-test'
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

    bm25_path = 'data/bm25_data.json'

    bm25encoder = BM25Encoder().load(bm25_path)
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    pc_t = PineconeHybridSearchRetriever(index=index, embeddings=embeddings, sparse_encoder=bm25encoder, top_k=4)
    recommendations = pc_t.invoke(query)
    pdf_links = []
    titles = []
    for results in recommendations:
        id = results.metadata['id']
        title = results.metadata['title']
        pdf_links.append(LINK+str(id))
        titles.append(title)
    return pdf_links, titles


def get_pdf_doc_as_chunks(path, chunk_size=600, chunk_overlap=40):
    pdf_loader = PyPDFLoader(file_path=path)
    pdf_documents = pdf_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(pdf_documents)
    return chunks


def load_and_get_file(query):
    "Loads the vector data base and returns the similarity search result"
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    db = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    return db.similarity_search_with_score(query)

def create_link(ids, min_thresh):
    '''Returns a list of links in the order of most similar to less similar if it pass threshold'''

    sorted_list = sorted(ids, key=lambda x: ids[x][0], reverse=True)
    pdf_links = []
    titles = []

    for id in sorted_list:
        score, title = ids[id]
        if score > min_thresh:
            pdf_links.append(LINK+str(id))
            titles.append(title)
    return pdf_links, titles

def get_links(query, method='Context Search'): 
    'Given a query sentence, search for similar docs and returns the download links of paper'
    pdf_ordered_links, ordered_titles = None, None
    print('Search Method', method)
    match method:
        case 'Context Search':
            similar_docs = load_and_get_file(query)
            corrected_ids = get_correct_ids(similar_docs)
            pdf_ordered_links, ordered_titles = create_link(corrected_ids, SIMILARITY_THRESHOLD)
        case 'Hybrid Search':
            pdf_ordered_links, ordered_titles = get_hybrid_res(query)
        case 'TFIDF Search':
            pdf_ordered_links, ordered_titles = get_tf_idf_res(query)

    return pdf_ordered_links, ordered_titles

def home(request):
    messages = []
    tmp = {}
    papers = {}
    method = 'Context Search'

    if request.method == 'POST':
        text = request.POST['text']
        method = request.POST['search_type']
        request.session['chat_method'] = request.POST['state']
        print('\n\nn\n post ', text, method, chat_messages)
        if text:
            get_chat_message(text)

    for message in chat_messages:
                
        if type(message) == HumanMessage:
            tmp['human'] = message.content
        elif type(message) == AIMessage:
            tmp['ai'] = message.content
            
        if len(tmp) % 2 == 0:
            messages.append(tmp)
            tmp = {} 
    if chat_messages and 'Query' in chat_messages[-1].content:
        result_query = chat_messages[-1].content.strip()
        pdf_ordered_links, ordered_titles = get_links(result_query, method)
        papers = [{'name': i[0], 'link': i[1]} for i in zip(ordered_titles, pdf_ordered_links)]
            
    context = {'messages': messages, 'papers': papers, 'search': method}
    return render(request, 'home.html', context)

def chat_post(request):
    global pdf_chat_messages, chatpdf_messages
    # resetting chat history
    pdf_chat_messages = []
    chatpdf_messages = []

    link = request.POST['link']
    name = request.POST['name']
    path = get_pdf(link)

    request.session['path'] = path
    request.session['visited'] = False
    request.session['link'] = link
    
    if 'srcid' in request.session:
        delete_chatpdf(headers, request.session['srcid'])
        del request.session['srcid']
    request.session['state'] = request.POST['state']

    return redirect('view_pdf_page')

    
def get_pdf(link):
    """Download the PDF from the provided link and show it in Streamlit."""
    # Define the path for the PDF file
    pdf_path = os.path.join(apps.get_app_config('home').path, 'static', 'pdf.pdf')

    try:
        response = requests.get(link)
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.exceptions.HTTPError as e:
        return HttpResponse('Http error')
    except requests.exceptions.RequestException as e:
        return HttpResponse('Download error')

    # Load the content into a BytesIO object
    pdf_bytes = io.BytesIO(response.content)

    # Write the PDF to a file
    with open(pdf_path, 'wb') as file:
        file.write(pdf_bytes.read())

    return pdf_path

def pdf_rag(request):
    path = request.session['path']

    if request.session['visited'] == True:
        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        db = FAISS.load_local('tmp_pdf_vectorstore', embeddings, allow_dangerous_deserialization=True)
    else:
        pdf_chunks = get_pdf_doc_as_chunks(path)
        db = get_vectorstore(pdf_chunks)
        request.session['visited'] = True
    text = request.POST['text']
    ai_res = manual_prompt(text, pdf_chat_messages, db, pdf_chat_model)
    pdf_chat_messages.append(HumanMessage(text))
    pdf_chat_messages.append(AIMessage(ai_res))
        

def create_chatpdf(headers, link):
    data = {'url': link}
    response = requests.post(
        'https://api.chatpdf.com/v1/sources/add-url', headers=headers, json=data)
    
    if response.status_code == 200:
        print('Source ID:', response.json()['sourceId'])
        return response.json()['sourceId']
    else:
        print('Status:', response.status_code)
        print('Error:', response.text)


def get_chatpdf_response(headers, srcid, messages):
    data = {
        'sourceId': srcid,
        'messages': messages
    }
    response = requests.post(
    'https://api.chatpdf.com/v1/chats/message', headers=headers, json=data)

    if response.status_code == 200:
        print('Result:', response.json()['content'])
        return response.json()['content']
    else:
        print('Status:', response.status_code)
        print('Error:', response.text)


def delete_chatpdf(headers, srcid):
    data = {
    'sources': [srcid],
    }

    try:
        response = requests.post(
            'https://api.chatpdf.com/v1/sources/delete', json=data, headers=headers)
        response.raise_for_status()
        print('Success')
    except requests.exceptions.RequestException as error:
        print('Error:', error)
        print('Response:', error.response.text)




def chatpdf(request):
    if request.session['visited'] == False:
        link = request.session['link'] 
        request.session['srcid'] = create_chatpdf(headers, link)
     

    text = request.POST['text']
    if text: 
        chatpdf_messages.append({'role': 'user', 'content': text})
        response = get_chatpdf_response(headers, request.session['srcid'], chatpdf_messages[-6:])
        chatpdf_messages.append({'role': 'assistant', 'content': response})



def view_pdf_page(request):
    global chat_messages
    chat_messages = []

    state = request.session['state']

    if request.method == 'POST':

        func = pdf_rag if state == 'rag' else chatpdf
        func(request)
    messages = [get_markdown(i.content) for i in pdf_chat_messages] if state == 'rag' else [get_markdown(i['content']) for i in  chatpdf_messages]
    context = {'link': request.session['link'], 'messages': messages}
    return render(request, 'view_pdf.html', context)  
