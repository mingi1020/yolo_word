import html
import io
import hashlib
import re

import numpy as np
import streamlit as st
from gtts import gTTS
from gtts.tts import gTTSError
from PIL import Image, ImageDraw, UnidentifiedImageError
from arena import render_battle_arena
from ultralytics import YOLO
from local_store import (
    accept_battle,
    accept_friend_request,
    authenticate,
    create_account,
    create_battle,
    friends,
    incoming_friend_requests,
    incoming_battles,
    initialize_database,
    list_photos,
    recent_battles,
    save_photo,
    send_friend_request,
    search_users,
)


KOREAN_WORDS = {
    "person": "사람",
    "bicycle": "자전거",
    "car": "자동차",
    "motorcycle": "오토바이",
    "airplane": "비행기",
    "bus": "버스",
    "train": "기차",
    "truck": "트럭",
    "boat": "배",
    "traffic light": "신호등",
    "fire hydrant": "소화전",
    "stop sign": "정지 표지판",
    "parking meter": "주차 요금기",
    "bench": "벤치",
    "bird": "새",
    "cat": "고양이",
    "dog": "강아지",
    "horse": "말",
    "sheep": "양",
    "cow": "소",
    "elephant": "코끼리",
    "bear": "곰",
    "zebra": "얼룩말",
    "giraffe": "기린",
    "backpack": "배낭",
    "umbrella": "우산",
    "handbag": "손가방",
    "tie": "넥타이",
    "suitcase": "여행 가방",
    "frisbee": "원반",
    "skis": "스키",
    "snowboard": "스노보드",
    "sports ball": "공",
    "kite": "연",
    "baseball bat": "야구 방망이",
    "baseball glove": "야구 글러브",
    "skateboard": "스케이트보드",
    "surfboard": "서핑보드",
    "tennis racket": "테니스 라켓",
    "bottle": "병",
    "wine glass": "와인 잔",
    "cup": "컵",
    "fork": "포크",
    "knife": "칼",
    "spoon": "숟가락",
    "bowl": "그릇",
    "banana": "바나나",
    "apple": "사과",
    "sandwich": "샌드위치",
    "orange": "오렌지",
    "broccoli": "브로콜리",
    "carrot": "당근",
    "hot dog": "핫도그",
    "pizza": "피자",
    "donut": "도넛",
    "cake": "케이크",
    "chair": "의자",
    "couch": "소파",
    "potted plant": "화분",
    "bed": "침대",
    "dining table": "식탁",
    "toilet": "변기",
    "tv": "텔레비전",
    "laptop": "노트북",
    "mouse": "마우스",
    "remote": "리모컨",
    "keyboard": "키보드",
    "cell phone": "스마트폰",
    "microwave": "전자레인지",
    "oven": "오븐",
    "toaster": "토스터",
    "sink": "싱크대",
    "refrigerator": "냉장고",
    "book": "책",
    "clock": "시계",
    "vase": "꽃병",
    "scissors": "가위",
    "teddy bear": "곰 인형",
    "hair drier": "드라이어",
    "toothbrush": "칫솔"
}

TEXT = {
    "ko": {
        "page_title": "단어 학습",
        "subtitle": "사진을 찍고, 새로운 단어를 한국어와 영어로 배워봐요!",
        "language": "화면 언어 / Display language",
        "camera": "아래 버튼을 눌러 사물을 찍어보세요!",
        "no_camera": "카메라 화면이 보이지 않나요? 브라우저의 카메라 권한을 확인해 주세요.",
        "recognized": "찾은 사물",
        "again": "사물을 조금 더 가까이, 밝은 곳에서 다시 찍어볼까요? 📸",
        "image_error": "사진을 읽지 못했어요. 다시 찍어 주세요.",
        "model_error": "사물 인식 모델을 불러오지 못했어요. 잠시 후 다시 시도해 주세요.",
        "inference_error": "사진을 분석하는 중 문제가 생겼어요. 다시 시도해 주세요.",
        "speech_error": "음성을 만들지 못했어요. 인터넷 연결을 확인해 주세요.",
        "tip": "팁: 사물 하나를 화면 가운데에 놓으면 더 잘 찾아요!",
        "ko_label": "한국어",
        "en_label": "영어",
    },
    "en": {
        "page_title": "Snap! Word Explorer",
        "subtitle": "Take a picture and learn a new word in Korean and English!",
        "language": "Display language / 화면 언어",
        "camera": "Tap the button below to take a picture!",
        "no_camera": "Can't see the camera? Please check your browser's camera permission.",
        "recognized": "We found",
        "again": "Move a little closer and try again in a bright place! 📸",
        "image_error": "We couldn't read that photo. Please take another one.",
        "model_error": "We couldn't load the object detector. Please try again shortly.",
        "inference_error": "Something went wrong while looking at the photo. Please try again.",
        "speech_error": "We couldn't make the audio. Please check your internet connection.",
        "tip": "Tip: Put one object in the middle of the picture for best results!",
        "ko_label": "Korean",
        "en_label": "English",
    },
}


@st.cache_resource
def load_model():
    """Load YOLO once and reuse it across Streamlit reruns."""
    return YOLO("yolov8n.pt")


@st.cache_data(show_spinner=False, max_entries=32)
def detect_photo(photo_bytes):
    """Detect an object once and cache the result for identical photos."""
    image = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
    image.thumbnail((960, 960), Image.Resampling.LANCZOS)
    model = load_model()
    prediction = model.predict(
        source=np.asarray(image),
        verbose=False,
    )[0]
    detections = detected_objects(prediction)
    return (
        tuple(
            (
                english_word,
                tuple(float(value) for value in coordinates),
                confidence,
                max(
                    0.0,
                    min(
                        1.0,
                        float(
                            (coordinates[2] - coordinates[0])
                            * (coordinates[3] - coordinates[1])
                            / (image.width * image.height)
                        ),
                    ),
                ),
            )
            for english_word, coordinates, confidence in detections
        ),
        image.width,
        image.height,
    )


@st.cache_data(show_spinner=False, max_entries=64)
def create_audio(korean_word, english_word, language):
    """Create spoken audio for the selected language."""
    audio = io.BytesIO()
    if language == "ko":
        gTTS(text=f"{korean_word}!", lang="ko").write_to_fp(audio)
    else:
        gTTS(text=f"{english_word}!", lang="en").write_to_fp(audio)
    return audio.getvalue()


@st.cache_resource
def initialize_local_database():
    initialize_database()
    return True


def detected_objects(result):
    """Return each detected class, ordered by the size of its largest box."""
    if result.boxes is None or len(result.boxes) == 0:
        return []

    coordinates = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    areas = (coordinates[:, 2] - coordinates[:, 0]) * (
        coordinates[:, 3] - coordinates[:, 1]
    )
    detections = []
    seen_words = set()
    for index in np.argsort(areas)[::-1]:
        english_word = result.names[class_ids[index]]
        if english_word not in seen_words:
            detections.append(
                (english_word, coordinates[index], float(confidences[index]))
            )
            seen_words.add(english_word)
    return detections


def show_word_card(korean_word, english_word, labels, language):
    """Display the word in the selected language in a child-friendly card."""
    korean_word = html.escape(korean_word)
    english_word = html.escape(english_word)
    if language == "ko":
        word_line = f'<div class="korean-word">{korean_word}</div>'
    else:
        word_line = f'<div class="english-word">{english_word}</div>'
    st.markdown(
        f"""
        <div class="word-card">
          <div class="card-caption">{html.escape(labels["recognized"])}</div>
          {word_line}
        </div>
        """,
        unsafe_allow_html=True,
    )


def image_to_jpeg(image):
    """Convert a PIL image to JPEG bytes for history storage."""
    image = image.copy()
    image.thumbnail((960, 960), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=82, optimize=True)
    return buffer.getvalue()


st.set_page_config(page_title="Snap! Word Explorer", page_icon="📸", layout="wide")
st.markdown(
    """
    <style>
    :root {
      --ink: #28234f;
      --paper: #ffffff;
      --lemon: #f5f3ff;
      --orange: #7658e8;
      --cyan: #e8f4ff;
      --pink: #ffb8ce;
      --line: 1px solid #ebe8f5;
      --hard-shadow: 0 14px 36px rgba(54, 43, 107, 0.11);
    }
    .stApp {
      background: linear-gradient(155deg, #f1f5ff 0%, #f8f3ff 58%, #fff8f2 100%);
      color: var(--ink);
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
    }
    [data-testid="stAppViewContainer"] .main .block-container {
      max-width: 980px;
      padding: 1.5rem 1.5rem 6rem;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1rem; }
    .brand-bar {
      display: flex; align-items: center; justify-content: space-between;
      gap: 1rem; background: rgba(255,255,255,.92); border: var(--line);
      box-shadow: 0 8px 25px rgba(54, 43, 107, .08); padding: 0.75rem 1rem;
      margin: 0.25rem 0 1.5rem; font-size: 0.95rem; font-weight: 900;
      letter-spacing: 0.01em; border-radius: 20px;
    }
    .brand-name { display: flex; align-items: center; gap: 0.6rem; }
    .brand-square {
      width: 38px; height: 38px; background: #eeeaff; border-radius: 14px;
      display: inline-grid; place-items: center;
    }
    .brand-square::after { content: "📸"; font-size: 1.2rem; }
    .brand-status {
      color: #7056dc; background: #f0edff; border-radius: 999px;
      padding: 0.35rem 0.7rem; font-size: 0.68rem; letter-spacing: .04em;
    }
    .welcome-hero {
      display: grid; grid-template-columns: minmax(0, 1fr) 150px;
      align-items: center; gap: 1rem; margin: 0 auto 1.25rem; padding: 1.5rem;
      max-width: 760px; background: linear-gradient(120deg, #fff 5%, #f0edff 100%);
      border: var(--line); border-radius: 26px; box-shadow: var(--hard-shadow);
    }
    .hero-chip {
      display: inline-block; margin-bottom: .65rem; padding: .3rem .7rem;
      color: #654bce; background: #e9e4ff; border-radius: 999px;
      font-size: .72rem; font-weight: 900; letter-spacing: .05em;
    }
    .hero-title {
      margin: 0; color: var(--ink); font-size: clamp(1.7rem, 4vw, 2.4rem);
      line-height: 1.13; font-weight: 900; letter-spacing: -.04em;
    }
    .hero-copy { margin: .55rem 0 0; color: #68677f; font-size: .95rem; }
    .hero-art {
      width: 132px; height: 132px; display: grid; place-items: center;
      justify-self: center; border-radius: 42% 58% 61% 39% / 43% 43% 57% 57%;
      background: linear-gradient(145deg, #ffd8e6, #ffe9b7);
      font-size: 4.1rem; transform: rotate(4deg);
    }
    .main-title {
      color: var(--ink); text-align: left;
      font-size: clamp(1.9rem, 5vw, 3rem); line-height: 1.08;
      font-weight: 900; letter-spacing: -0.05em;
      margin: 0.5rem 0 0.35rem;
    }
    .subtitle {
      color: #68677f; text-align: left; font-size: 1rem;
      font-weight: 600; margin: 0 0 1.4rem;
    }
    div[data-testid="stTabs"] { max-width: 760px; margin: 0 auto; }
    .word-card {
      background: var(--paper); border: var(--line); box-shadow: var(--hard-shadow);
      border-radius: 22px; padding: 1.35rem; margin: 1rem 0 1.5rem;
      text-align: left;
    }
    .card-caption {
      color: #6048c3; display: inline-block; background: #eae6ff;
      border-radius: 999px; padding: 0.3rem 0.7rem;
      font-size: 0.78rem; font-weight: 900; text-transform: uppercase;
      margin-bottom: 0.7rem;
    }
    .korean-word, .english-word {
      color: var(--ink); font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      font-size: clamp(2rem, 5vw, 3.4rem); line-height: 1.12;
      font-weight: 900; overflow-wrap: anywhere;
    }
    .friendly-note {
      background: #fff2ce; border: 0; box-shadow: none;
      border-radius: 18px; padding: 1rem; color: var(--ink); text-align: left;
      font-size: 1rem; font-weight: 800;
    }
    .history-label {
      color: var(--ink); font-size: 1.35rem; font-weight: 900;
      text-align: left; margin: 1rem 0;
    }
    .history-time { color: #454545; font-size: 0.8rem; font-weight: 700; }
    div[data-testid="stRadio"] > label { display: none; }
    div[data-testid="stRadio"] [role="radiogroup"] {
      display: flex; flex-wrap: wrap; gap: 0.4rem; width: fit-content;
      background: rgba(255,255,255,.82); border: var(--line);
      box-shadow: 0 6px 20px rgba(54, 43, 107, .07);
      border-radius: 999px; padding: 0.35rem; margin: 0.75rem 0 1.5rem;
    }
    div[data-testid="stRadio"] [role="radio"] {
      border: 0; border-radius: 999px;
      background: transparent; color: #625e7d; padding: 0.55rem 0.9rem;
      font-weight: 800;
    }
    div[data-testid="stRadio"] [role="radio"][aria-checked="true"] {
      background: var(--orange); color: #fff;
      box-shadow: 0 5px 12px rgba(118, 88, 232, .24);
    }
    .st-key-section {
      position: fixed !important; z-index: 999;
      left: 0; right: 0; bottom: 0; width: 100%;
      box-sizing: border-box; margin: 0 !important;
      padding: .35rem max(16px, calc((100vw - 600px) / 2))
        calc(.35rem + env(safe-area-inset-bottom, 0px));
      background: #fff; border: 0; border-top: 1px solid #dbdbdb;
      border-radius: 0; box-shadow: 0 -3px 10px rgba(0, 0, 0, .035);
    }
    .st-key-section div[data-testid="stRadio"] { width: 100%; margin: 0; }
    .st-key-section div[data-testid="stRadio"] [role="radiogroup"] {
      display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: .2rem; width: 100%; margin: 0; padding: 0;
      background: transparent; border: 0; border-radius: 0; box-shadow: none;
    }
    .st-key-section [data-testid="stRadioOption"] {
      justify-content: center; min-width: 0; padding: .25rem 0;
      color: #737373; background: transparent; border: 0; border-radius: 0;
      box-shadow: none; cursor: pointer;
    }
    .st-key-section [data-testid="stRadioOption"] > div {
      display: flex; flex-direction: column; align-items: center; gap: 2px;
      width: 100%;
    }
    .st-key-section [data-testid="stRadioOption"] > div > div:first-child {
      display: none;
    }
    .st-key-section [data-testid="stRadioOption"] > div::before {
      color: inherit; font: 1.25rem/1.2 "Segoe UI Symbol", sans-serif;
    }
    .st-key-section [role="radiogroup"] > div:nth-child(1) [data-testid="stRadioOption"] > div::before {
      content: "⌂";
    }
    .st-key-section [role="radiogroup"] > div:nth-child(2) [data-testid="stRadioOption"] > div::before {
      content: "📝";
    }
    .st-key-section [role="radiogroup"] > div:nth-child(3) [data-testid="stRadioOption"] > div::before {
      content: "🫱🏻‍🫲🏻";
    }
    .st-key-section [role="radiogroup"] > div:nth-child(4) [data-testid="stRadioOption"] > div::before {
      content: "🗡️";
    }
    .st-key-section [data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] p {
      margin: 0; color: inherit; font-size: .65rem; line-height: 1.2;
      white-space: nowrap;
    }
    .st-key-section [data-testid="stRadioOption"][data-selected="true"] {
      color: #111; background: transparent;
    }
    .st-key-section [data-testid="stRadioOption"][data-selected="true"] p {
      font-weight: 700;
    }
    div[data-testid="stTabs"] [role="tablist"] {
      gap: 0.45rem; border-bottom: 1px solid #ebe8f5;
    }
    div[data-testid="stTabs"] [role="tab"] {
      background: #f6f4fc; border: 1px solid #ebe8f5; border-bottom: 0;
      border-radius: 14px 14px 0 0; color: #68677f; font-weight: 800;
      padding: 0.65rem 1rem;
    }
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
      background: #fff; color: var(--orange);
    }
    div[data-testid="stForm"] {
      background: var(--paper); border: var(--line); border-radius: 0 0 22px 22px;
      box-shadow: var(--hard-shadow); padding: 1.35rem;
    }
    div[data-testid="stTextInputRootElement"] {
      background: white; border: 1px solid #e6e2f1 !important;
      border-radius: 14px !important; box-shadow: none !important;
    }
    div[data-testid="stTextInput"] input {
      background: white; color: var(--ink); border: 0 !important;
      border-radius: 14px !important; font-family: inherit; font-weight: 600;
    }
    div[data-testid="stTextInputRootElement"]:focus-within {
      border-color: #9c87f0 !important; box-shadow: 0 0 0 3px #eeeaff !important;
    }
    div[data-testid="stTextInputRootElement"] button {
      border-radius: 0 14px 14px 0 !important; border-left: 1px solid #e6e2f1 !important;
    }
    input[type="radio"] { accent-color: var(--orange); }
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
      background: var(--paper); border: 1px solid #e6e2f1; border-radius: 14px;
      box-shadow: 0 4px 12px rgba(54, 43, 107, .06);
    }
    div[data-testid="stButton"] button,
    div[data-testid="stFormSubmitButton"] button {
      color: #fff; background: var(--orange); border: 0;
      border-radius: 999px; box-shadow: 0 7px 16px rgba(118, 88, 232, .2);
      font-family: inherit; font-weight: 800;
      transition: transform 120ms ease, box-shadow 120ms ease, background 120ms ease;
    }
    div[data-testid="stButton"] button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
      color: #fff; border-color: transparent; background: #6245d1;
      transform: translateY(-1px); box-shadow: 0 9px 20px rgba(98, 69, 209, .26);
    }
    div[data-testid="stButton"] button:disabled {
      color: #9291a2 !important; background: #e9e9ef !important;
      border: 1px solid #e0dfe8 !important; box-shadow: none !important;
      opacity: 1 !important; transform: none !important;
    }
    .st-key-wordbook div[data-testid="stButton"] button:not(:disabled) {
      color: var(--ink) !important; background: #fff !important;
      border: 1px solid #ebe8f5 !important;
      box-shadow: 0 5px 15px rgba(54, 43, 107, .08) !important;
    }
    .st-key-wordbook div[data-testid="stButton"] button:not(:disabled):hover {
      color: var(--ink) !important; background: #f0edff !important;
    }
    div[data-testid="stAlert"] {
      background: var(--paper); border: var(--line); border-radius: 18px;
      box-shadow: 0 7px 20px rgba(54, 43, 107, .07); color: var(--ink);
    }
    [data-testid="stImage"] img {
      border: 0; border-radius: 20px; box-shadow: 0 8px 22px rgba(54, 43, 107, .12);
    }
    div[data-testid="stCameraInput"] {
      overflow: hidden; background: #fff; border: var(--line); border-radius: 22px;
      box-shadow: var(--hard-shadow);
    }
    div[data-testid="stProgress"] > div > div > div {
      background: var(--orange);
    }
    div[data-testid="stExpander"] {
      background: #fff; border: var(--line); border-radius: 18px;
      box-shadow: 0 6px 18px rgba(54, 43, 107, .07);
    }
    h1, h2, h3, p, label { color: var(--ink); }
    @media (max-width: 600px) {
      [data-testid="stAppViewContainer"] .main .block-container {
        padding: 0.8rem 0.85rem 6rem;
      }
      .brand-bar { gap: .5rem; font-size: .8rem; }
      .brand-status { font-size: .58rem; }
      .welcome-hero {
        grid-template-columns: minmax(0, 1fr) 82px; gap: .5rem;
        padding: 1.1rem; border-radius: 22px;
      }
      .hero-art { width: 78px; height: 78px; font-size: 2.7rem; }
      .hero-copy { font-size: .82rem; }
      .word-card { padding: 1rem; }
      div[data-testid="stRadio"] [role="radiogroup"] { width: 100%; justify-content: center; }
      div[data-testid="stRadio"] [role="radio"] { padding: .5rem .65rem; font-size: .82rem; }
      .st-key-section [data-testid="stRadioOption"] {
        padding: .2rem 0;
      }
      .st-key-section [data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] p {
        font-size: .62rem;
      }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="brand-bar"><div class="brand-name"><span class="brand-square"></span>'
    'Snap! Word Explorer</div><span class="brand-status">LITTLE WORDS · BIG DISCOVERIES</span></div>',
    unsafe_allow_html=True,
)

language_choice = st.radio(
    "화면 언어 / Display language",
    options=("한국어", "English"),
    horizontal=True,
    label_visibility="visible",
)
language = "ko" if language_choice == "한국어" else "en"
labels = TEXT[language]

initialize_local_database()
auth_user = st.session_state.get("auth_user")

if auth_user is None:
    welcome_title = (
        "사진으로 시작하는<br>즐거운 단어 탐험"
        if language == "ko"
        else "Every photo starts<br>a little adventure"
    )
    welcome_copy = (
        "사물 이름을 배우고 나만의 단어장을 채워 보세요."
        if language == "ko"
        else "Discover new words, build your collection, and learn as you play."
    )
    st.markdown(
        f'<div class="welcome-hero"><div><span class="hero-chip">SNAP · LEARN · PLAY</span>'
        f'<h2 class="hero-title">{welcome_title}</h2><p class="hero-copy">{welcome_copy}</p>'
        f'</div><div class="hero-art" aria-hidden="true">📸</div></div>',
        unsafe_allow_html=True,
    )
    login_tab, signup_tab = st.tabs(
        ("로그인", "회원가입") if language == "ko" else ("Sign in", "Sign up")
    )
    with login_tab:
        with st.form("login_form"):
            username = st.text_input("사용자 이름" if language == "ko" else "Username")
            password = st.text_input(
                "비밀번호" if language == "ko" else "Password",
                type="password",
            )
            login_submitted = st.form_submit_button(
                "로그인" if language == "ko" else "Sign in",
                use_container_width=True,
            )
        if login_submitted:
            auth_user = authenticate(username, password)
            if auth_user:
                st.session_state.auth_user = auth_user
                st.rerun()
            st.error(
                "사용자 이름이나 비밀번호를 확인해 주세요."
                if language == "ko"
                else "Check your username and password."
            )

    with signup_tab:
        with st.form("signup_form"):
            username = st.text_input(
                "사용자 이름 (영문, 숫자, 밑줄 3~24자)"
                if language == "ko"
                else "Username (3–24 letters, numbers, underscores)"
            )
            password = st.text_input(
                "비밀번호 (8자 이상)" if language == "ko" else "Password (8+ characters)",
                type="password",
            )
            signup_submitted = st.form_submit_button(
                "회원가입" if language == "ko" else "Create account",
                use_container_width=True,
            )
        if signup_submitted:
            try:
                st.session_state.auth_user = create_account(username, password)
                st.rerun()
            except ValueError as error:
                st.error(str(error) if language == "ko" else "Check your username and password requirements.")
    st.stop()

user_id = auth_user["id"]
st.caption(f'👤 {auth_user["username"]}')
if st.button("로그아웃" if language == "ko" else "Sign out", key="sign_out"):
    st.session_state.pop("auth_user", None)
    st.rerun()

st.markdown(f'<div class="main-title">{labels["page_title"]} ✨</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subtitle">{labels["subtitle"]}</div>', unsafe_allow_html=True)

if "selected_history_word" not in st.session_state:
    st.session_state.selected_history_word = None
if "section" not in st.session_state:
    st.session_state.section = "explore"

explore_label = "사진 탐험" if language == "ko" else "Photo Explorer"
history_label = "단어장" if language == "ko" else "Word Book"
section = st.radio(
    "메뉴",
    options=("explore", "history", "friends", "battles"),
    format_func=lambda value: {
        "explore": explore_label,
        "history": history_label,
        "friends": "친구" if language == "ko" else "Friends",
        "battles": "대전" if language == "ko" else "Battles",
    }[value],
    horizontal=True,
    label_visibility="collapsed",
    key="section",
)

if section == "explore":
    st.info(labels["camera"])
    photo = st.camera_input(labels["camera"], key="object_camera")

    if photo is None:
        st.caption(labels["no_camera"])
    else:
        try:
            original_image = Image.open(io.BytesIO(photo.getvalue())).convert("RGB")
        except (UnidentifiedImageError, OSError):
            st.error(labels["image_error"])
        else:
            try:
                with st.spinner(
                    "사진을 살펴보고 있어요..."
                    if language == "ko"
                    else "Looking at your photo..."
                ):
                    detection = detect_photo(photo.getvalue())
            except Exception as error:
                st.error(f'{labels["model_error"]} ({error})')
            else:
                detections, inference_width, inference_height = detection
                if not detections:
                    st.markdown(
                        f'<div class="friendly-note">{labels["again"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    scale_x = original_image.width / inference_width
                    scale_y = original_image.height / inference_height
                    annotated_image = original_image.copy()
                    draw = ImageDraw.Draw(annotated_image)
                    line_width = max(6, original_image.width // 120)
                    for english_word, coordinates, _, _ in detections:
                        scaled_coordinates = (
                            coordinates[0] * scale_x,
                            coordinates[1] * scale_y,
                            coordinates[2] * scale_x,
                            coordinates[3] * scale_y,
                        )
                        draw.rectangle(
                            scaled_coordinates,
                            outline="#20c45a",
                            width=line_width,
                        )
                    cloud_image = image_to_jpeg(annotated_image)
                    for english_word, _, confidence, area_ratio in detections:
                        korean_word = KOREAN_WORDS.get(english_word, "사물")
                        save_photo(
                            user_id,
                            hashlib.sha1(
                                photo.getvalue()
                                + english_word.encode("utf-8")
                            ).hexdigest(),
                            english_word,
                            korean_word,
                            cloud_image,
                            confidence,
                            area_ratio,
                        )
                    st.image(annotated_image, use_container_width=True)
                    english_word = detections[0][0]
                    korean_word = KOREAN_WORDS.get(english_word, "사물")
                    show_word_card(korean_word, english_word, labels, language)
                    try:
                        audio_bytes = create_audio(korean_word, english_word, language)
                        audio_buffer = io.BytesIO(audio_bytes)
                        st.audio(audio_buffer, format="audio/mp3", autoplay=True)
                    except gTTSError as error:
                        st.warning(f'{labels["speech_error"]} ({error})')

    st.markdown(
        f'<div class="friendly-note">{labels["tip"]}</div>',
        unsafe_allow_html=True,
    )

elif section == "history":
    history = list_photos(user_id)
    unlocked_words = {item["class_name"] for item in history}
    if st.session_state.selected_history_word:
        selected_word = st.session_state.selected_history_word
        matching_items = [
            item for item in history if item["class_name"] == selected_word
        ]
        if not matching_items:
            st.session_state.selected_history_word = None
            st.rerun()

        back_label = "← 단어장으로" if language == "ko" else "← Back to Word Book"
        if st.button(back_label, key="back_to_word_book"):
            st.session_state.selected_history_word = None
            st.rerun()

        display_word = (
            KOREAN_WORDS.get(selected_word, "사물")
            if language == "ko"
            else selected_word
        )
        st.markdown(
            f'<div class="history-label">{html.escape(display_word)}</div>',
            unsafe_allow_html=True,
        )
        for index, item in enumerate(matching_items):
            st.image(item["image"], use_container_width=True)
            st.markdown(
                f'<div class="history-time">{html.escape(item["created_at"][:16])}</div>',
                unsafe_allow_html=True,
            )
    else:
        category_title = "찍을수록 열리는 단어장" if language == "ko" else "Unlock words as you discover them"
        st.markdown(
            f'<div class="history-label">{category_title}</div>',
            unsafe_allow_html=True,
        )
        word_keys = sorted(
            KOREAN_WORDS,
            key=lambda word: KOREAN_WORDS[word] if language == "ko" else word,
        )
        columns = st.container(key="wordbook").columns(3)
        for index, english_word in enumerate(word_keys):
            unlocked = english_word in unlocked_words
            word = (
                KOREAN_WORDS[english_word]
                if language == "ko"
                else english_word
            )
            card_label = f"🔓 {word}" if unlocked else f"🔒 {word}"
            if columns[index % 3].button(
                card_label,
                key=f"word_card_{english_word}",
                disabled=not unlocked,
                use_container_width=True,
            ):
                st.session_state.selected_history_word = english_word
                st.rerun()

elif section == "friends":
    st.markdown(
        '<div class="history-label">친구와 연결하기</div>'
        if language == "ko"
        else '<div class="history-label">Connect with friends</div>',
        unsafe_allow_html=True,
    )
    with st.form("friend_search_form"):
        username_query = st.text_input(
            "사용자 이름 검색" if language == "ko" else "Search usernames"
        )
        search_submitted = st.form_submit_button(
            "검색" if language == "ko" else "Search"
        )
    if search_submitted:
        username_query = username_query.strip()
        if not re.fullmatch(r"[A-Za-z0-9_]{3,24}", username_query):
            st.session_state.friend_search_results = []
            st.warning(
                "사용자 이름을 3~24자의 영문, 숫자, 밑줄로 입력해 주세요."
                if language == "ko"
                else "Enter 3–24 letters, numbers, or underscores."
            )
        else:
            st.session_state.friend_search_results = search_users(
                username_query,
                user_id,
            )

    for candidate in st.session_state.get("friend_search_results", []):
        left, right = st.columns([3, 1])
        left.write(f'@{candidate["username"]}')
        if right.button(
            "친구 요청" if language == "ko" else "Add friend",
            key=f'friend_request_{candidate["id"]}',
        ):
            try:
                send_friend_request(user_id, candidate["id"])
            except ValueError as error:
                st.warning(str(error))
            else:
                st.success(
                    "친구 요청을 보냈어요."
                    if language == "ko"
                    else "Friend request sent."
                )

    st.subheader("받은 친구 요청" if language == "ko" else "Incoming requests")
    pending_requests = incoming_friend_requests(user_id)
    if not pending_requests:
        st.caption("받은 요청이 없어요." if language == "ko" else "No incoming requests.")
    for request in pending_requests:
        left, right = st.columns([3, 1])
        left.write(f'@{request["username"]}')
        if right.button(
            "수락" if language == "ko" else "Accept",
            key=f'accept_friend_{request["id"]}',
        ):
            accept_friend_request(request["id"], user_id)
            st.rerun()

    st.subheader("내 친구" if language == "ko" else "My friends")
    friend_list = friends(user_id)
    if not friend_list:
        st.caption("아직 친구가 없어요." if language == "ko" else "No friends yet.")
    else:
        for friend in friend_list:
            st.write(f'👋 @{friend["username"]}')

elif section == "battles":
    st.markdown(
        '<div class="history-label">사진 자동 대전</div>'
        if language == "ko"
        else '<div class="history-label">Photo Auto Battle</div>',
        unsafe_allow_html=True,
    )
    my_photos = list_photos(user_id)
    friend_list = friends(user_id)
    if not my_photos:
        st.info(
            "먼저 사물을 찍어 사진을 모아 주세요."
            if language == "ko"
            else "Take some pictures first to build your photo collection."
        )
    elif not friend_list:
        st.info(
            "친구를 맺으면 사진 대전을 시작할 수 있어요."
            if language == "ko"
            else "Add a friend before starting a photo battle."
        )
    else:
        selected_scene = st.radio(
            "전투 장소" if language == "ko" else "Battle arena",
            options=("beach", "forest"),
            format_func=lambda value: (
                "🏖️ 해변" if language == "ko" and value == "beach"
                else "🌲 숲" if language == "ko"
                else "🏖️ BEACH" if value == "beach"
                else "🌲 FOREST"
            ),
            horizontal=True,
        )
        friend_options = {
            f'@{friend["username"]}': friend["id"] for friend in friend_list
        }
        selected_friend_name = st.selectbox(
            "대전할 친구" if language == "ko" else "Choose a friend",
            list(friend_options),
        )
        if st.button(
            "대전 시작" if language == "ko" else "Start battle",
            key="start_battle",
            use_container_width=True,
        ):
            create_battle(
                friend_options[selected_friend_name],
                user_id,
                my_photos[0]["id"],
                selected_scene,
            )
            st.success(
                f"{selected_friend_name}에게 대전을 신청했어요!"
                if language == "ko"
                else f"Battle invitation sent to {selected_friend_name}!"
            )

    waiting_battles = incoming_battles(user_id)
    if waiting_battles:
        st.subheader("받은 대전 신청" if language == "ko" else "Battle invitations")
        if my_photos:
            for battle in waiting_battles:
                left, right = st.columns([3, 1])
                scene_name = (
                    "해변" if language == "ko" and battle["scene"] == "beach"
                    else "숲" if language == "ko"
                    else "BEACH" if battle["scene"] == "beach"
                    else "FOREST"
                )
                left.write(
                    f'@{battle["challenger"]} · {scene_name} '
                    + ("도전!" if language == "ko" else "challenged you!")
                )
                if right.button(
                    "수락하고 대전" if language == "ko" else "Accept & fight",
                    key=f'accept_battle_{battle["id"]}',
                ):
                    accept_battle(battle["id"], user_id, my_photos[0]["id"])
                    st.rerun()
        else:
            st.info(
                "대전을 수락하려면 먼저 사진을 찍어야 해요."
                if language == "ko"
                else "Take a picture before accepting a battle."
            )

    battle_history = recent_battles(user_id)
    st.subheader("최근 대전" if language == "ko" else "Recent battles")
    if not battle_history:
        st.caption("아직 대전 기록이 없어요." if language == "ko" else "No battles yet.")
    for index, battle in enumerate(battle_history):
        duel_title = f'@{battle["challenger"]} VS @{battle["opponent"]}'
        with st.expander(duel_title, expanded=index == 0):
            st.markdown(
                render_battle_arena(battle, language),
                unsafe_allow_html=True,
            )
