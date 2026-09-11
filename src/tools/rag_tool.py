import os
#tool
from langchain.tools import tool
#파일로드
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
#Split
from langchain_text_splitters import RecursiveCharacterTextSplitter
# RAG 데이터 Embed
from langchain_openai import OpenAIEmbeddings
# RAG 데이터 Store
from langchain_core.vectorstores import InMemoryVectorStore


#파일 로드
def pdf_loader(filename:str):
  loader = PyPDFLoader(f"C:/Users/nodecrew/Desktop/{filename}")
  pages = loader.load()
  return pages


#Split
def split_pdf(docs:Document):
  splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)#쪼개진 chunk의 시작 index

  splits = splitter.split_documents(docs)
  return splits


#임베딩 셋팅
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
google_pdf = pdf_loader('GOOG-10-K-2025.pdf')
google_splits = split_pdf(google_pdf)
# 벡터스토어 세팅(임베딩)
google_vectorstore = InMemoryVectorStore(embedding=embeddings)
# 벡터스토어 저장 (문서조각)
google_vectorstore.add_documents(documents=google_splits)


#툴
@tool(parse_docstring=True)
def rag_search_google_document(query:str):
    """Alphabet Inc.(Google) FY2026 10-K 연례 보고서 문서에서 관련 정보를 검색합니다.

    Google의 재무 실적(매출, 영업이익), 사업 부문(Google Services, Google Cloud, Other Bets),
    주요 제품 및 서비스(Search, YouTube, Gemini, Android, Waymo 등), 리스크 요인(반독점 규제, AI 경쟁 등),
    공급망 및 설비 투자(CapEx) 현황에 대한 질문이 들어올 때 이 툴을 사용합니다.

    Args:
        query: Alphabet/Google 10-K 보고서에서 찾고자 하는 내용에 대한 자연어 검색어.
    """
    retrieved_docs = google_vectorstore.similarity_search(query, k=4)

    result = '\n---\n'.join(map(lambda doc: doc.page_content, retrieved_docs))

    return result
