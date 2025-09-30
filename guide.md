# Customer Support Agent 환경 설정 가이드

## 1. uv 설치

PowerShell에서 다음 명령어를 실행:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

설치 확인:
```cmd
uv --version
```

## 2. 프로젝트 폴더 다운로드

프로젝트 폴더 다운로드 및 이동
```cmd
cd "프로젝트 폴더 경로"
```

## 3. uv 프로젝트 초기화

uv로 프로젝트 초기화:
```cmd
uv init
```

## 4. 가상환경 생성 및 활성화

가상환경 생성:
```cmd
uv venv --python 3.11
```

가상환경 활성화:
```cmd
.venv\Scripts\activate
```

- 권한설정 오류 발생시
    1. 파워셸 실행 정책 확인: Restricted
    ```cmd
    Get-ExecutionPolicy
    ```
    - Restricted 출력 확인
    2. 실행 정책 변경
    ```cmd
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    ```
    3. 가상환경 활성화
    ```cmd
    .venv\Scripts\activate
    ```

## 5. 의존성 설치

필요한 패키지들을 설치:
```cmd
uv add langgraph langchain-core openai==1.52.2 faiss-cpu numpy tavily-python python-dotenv "httpx<0.28.0" jupyter ipykernel langchain-upstage langchain langgraph-cli[inmem]
```

## 6. Jupyter 노트북 실행

가상환경에서 Jupyter 노트북을 실행:
```cmd
uv run jupyter notebook
```

브라우저에서 자동으로 Jupyter 노트북이 열리고, 모든 패키지가 정상적으로 인식됩니다.

**Jupyter 노트북 종료:**
```cmd
Ctrl + C (터미널에서)
```

## 7. 환경변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 다음 내용을 추가:

```
UPSTAGE_API_KEY=your_upstage_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

## 8. LangGraph 설정

프로젝트 루트에 `langgraph.json` 파일을 생성하고 다음 내용을 추가:

```json
{
    "dependencies": [
        "."
    ],
    "graphs": {
        "customer_support": "graph_build:graph"
    },
    "env": ".env"
}
```

### API 키 발급 방법

**Upstage API:**
1. [Upstage Console](https://console.upstage.ai/) 가입
2. API 키 생성

**Tavily API:**
1. [Tavily](https://tavily.com/) 가입
2. API 키 생성


## 문제 해결

**Python 버전 문제:**
```cmd
uv python install 3.11
```

**가상환경 확인:**
```cmd
where python
```

**의존성 재설치:**
```cmd
uv cache clean
uv sync
```

**Jupyter 노트북 실행 문제:**
```cmd
uv run jupyter --version
uv run jupyter notebook --port=8889
```
