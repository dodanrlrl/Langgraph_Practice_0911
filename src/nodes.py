from src.state import RagState
from src.tools.rag_tool import rag_search_google_document
from langchain.agents import create_agent


prompt = """너는 기업 분석 및 금융/기술 정보 전문 AI 에이전트이다.
사용자의 질문에 답하기 위해 제공된 툴을 적극적으로 활용하여 정확하고 신뢰할 수 있는 정보를 제공하라.

[툴 사용 규칙]
1. 기업 보고서 우선 조회 (RAG Tools):
   - Alphabet(Google) 관련 재무, 실적, 사업, 제품(Gemini 등) 질문 -> `rag_search_google_document` 사용
   - NVIDIA 관련 재무, 실적, 사업, 제품(Blackwell 등) 질문 -> `rag_search_nvidia_document` 사용
   - 두 기업 비교 질문 -> 두 RAG 툴을 모두 호출하여 데이터 비교

2. 외부 검색 활용 (Web Search Tool):
   - 10-K 보고서에 없는 최신 뉴스, 실시간 시장 반응, 주가, 기타 최신 정보 -> `search_tavily_tool` 사용
   - RAG 툴 검색 결과가 불충분하거나 정보가 없을 경우 보완 목적으로 사용

3. 근거 기반 답변:
   - 반드시 툴을 통해 조회된 사실(Fact)에 기반하여 답변하라.
   - 추측이나 근거 없는 정보 제공은 엄격히 금지한다.
   - 정보 출처(예: Alphabet FY2026 10-K, 최신 웹 검색 등)를 답변에 명시하라.
"""

agent = create_agent(
  model='openai:gpt-4.1-mini',
  tools=[rag_search_google_document],
  system_prompt=prompt,
)

def use_rag_data(state : RagState):

  answer = agent.invoke(
  # 1번인자 : 메세지
  {
      'messages': state['messages'][-1]
  }
    
  )

  return{'messages': answer}

  