# 🚀 한국주식 분석 웹앱 배포 가이드

## 📁 다운로드할 파일 3개
1. **app.py** - 메인 웹앱 코드
2. **requirements.txt** - 필요한 라이브러리 목록
3. **runtime.txt** - Python 버전 지정

---

## 🏁 1단계: 내 컴퓨터에서 먼저 테스트 (5분)

### 1-1. 파일 준비
세 개의 파일을 **같은 폴더**에 저장하세요. 예:
```
C:\Users\wj201\OneDrive\Desktop\우진\코딩\stockapp\
├── app.py
├── requirements.txt
└── runtime.txt
```

### 1-2. Streamlit 설치
VS Code 터미널에서:
```
py -3.12 -m pip install -r requirements.txt
```

### 1-3. 실행
```
py -3.12 -m streamlit run app.py
```

→ 자동으로 브라우저가 열리고 `http://localhost:8501`에서 웹앱이 뜹니다!

---

## 🌐 2단계: GitHub에 업로드 (10분)

### 2-1. GitHub 계정 만들기
👉 https://github.com/signup (무료)

### 2-2. 새 Repository 만들기
1. GitHub 로그인 후 우측 상단 **+ 버튼** → **New repository**
2. **Repository name**: `korean-stock-app` (원하는 이름)
3. **Public** 선택 (무료 배포를 위해 공개 필수)
4. **Add a README file** 체크
5. **Create repository** 클릭

### 2-3. 파일 업로드
1. 만든 repository 화면에서 **Add file** → **Upload files** 클릭
2. `app.py`, `requirements.txt`, `runtime.txt` 세 파일을 드래그 앤 드롭
3. 아래쪽에 **Commit changes** 클릭

---

## ☁️ 3단계: Streamlit Cloud에 배포 (5분)

### 3-1. Streamlit Cloud 접속
👉 https://share.streamlit.io/

### 3-2. GitHub 계정으로 로그인
**Continue with GitHub** 클릭 → 로그인 → 권한 승인

### 3-3. 앱 배포
1. 우측 상단 **Create app** 클릭
2. **Deploy a public app from GitHub** 선택
3. 다음 정보 입력:
   - **Repository**: `내아이디/korean-stock-app`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: 원하는 URL (예: `korean-stock-wj201`)
4. **Deploy!** 클릭

### 3-4. 대기
3~5분간 빌드 진행. 완료되면 URL이 생성됩니다:
```
https://korean-stock-wj201.streamlit.app
```

---

## 🎉 4단계: 어디서든 사용!

### 💻 PC에서
- 크롬/엣지/사파리 등 어떤 브라우저든 URL 접속

### 📱 스마트폰에서
- 같은 URL 접속 → 모바일 최적화되어 표시됨
- 홈 화면에 추가하면 앱처럼 사용 가능!
  - **iOS**: 사파리 → 공유 → 홈 화면에 추가
  - **안드로이드**: 크롬 → 메뉴 → 홈 화면에 추가

---

## 🔧 문제 해결

### 로컬에서 `streamlit: command not found`
```
py -3.12 -m streamlit run app.py
```

### Streamlit Cloud 빌드 실패
- **requirements.txt** 파일 확인
- **runtime.txt**에 `python-3.12` 있는지 확인
- 로그 확인: Streamlit Cloud 대시보드에서 **Manage app** → **Logs**

### 데이터 안 나옴
- pykrx는 장 마감 후(18시 이후) 당일 데이터 제공
- 주말/공휴일은 데이터 없음

---

## 🔄 코드 수정하고 싶을 때

1. GitHub repository에서 `app.py` 클릭
2. 우측 연필 아이콘 🖊️ 클릭해서 수정
3. **Commit changes** 클릭
4. **Streamlit Cloud가 자동으로 재배포합니다!** (2~3분)

---

## 💡 추가 팁

### URL 짧게 만들기
bitly.com 등으로 단축 URL 만들면 공유 편리

### 비공개로 쓰고 싶다면
Streamlit Cloud 유료 플랜 필요, 또는 비밀번호 추가 가능:
```python
# app.py 상단에 추가
password = st.text_input("비밀번호", type="password")
if password != "내비밀번호":
    st.stop()
```

### 업데이트 자동화
GitHub에 push할 때마다 Streamlit Cloud가 자동 재배포!
