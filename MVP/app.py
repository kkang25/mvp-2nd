import datetime
import pandas as pd
from collections import deque
import streamlit as st
import os # 환경 변수를 읽기 위해 추가
from openai import AzureOpenAI # Azure OpenAI와 연동하기 위해 추가
import holidays # 공휴일 처리를 위해 추가
from dotenv import load_dotenv

load_dotenv()

# --- Azure OpenAI 설정 (환경 변수에서 불러오기) ---
# 실제 배포 시에는 환경 변수나 Azure Key Vault 등을 사용하는 것이 안전해요!
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION =  os.getenv("AZURE_OPENAI_API_VERSION") # 사용하려는 API 버전에 맞춰주세요.
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini-deployment") # Azure에 배포한 gpt-4o-mini 모델의 배포 이름

# Azure OpenAI 클라이언트 초기화 함수
def get_azure_openai_client():
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        st.warning("Azure OpenAI 설정 환경 변수(AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY)가 설정되지 않았습니다. OpenAI 연동 기능은 작동하지 않습니다.")
        return None
    try:
        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION
        )
        return client
    except Exception as e:
        st.error(f"Azure OpenAI 클라이언트 초기화 중 오류 발생: {e}")
        return None

# --- 핵심 회의 일정 생성 로직 함수 ---
def schedule_meetings_logic(start_date_obj, tasks_df, holiday_list_str):
    """
    회의 일정을 생성하는 핵심 로직.
    - 시작일로부터 최대 4일 근무일 동안 회의를 배정합니다.
    - 점심시간(11:30~13:00)과 주말, 공휴일을 제외합니다.
    - 동일 과제 요청자의 과제는 연이어 배치합니다.
    - 당일 남은 시간이 부족하면 다음 날로 회의를 미룹니다.
    """
    # 공휴일 리스트 파싱
    parsed_holidays = []
    if holiday_list_str:
        for date_str in holiday_list_str.split(','):
            try:
                parsed_holidays.append(datetime.datetime.strptime(date_str.strip(), '%Y-%m-%d').date())
            except ValueError:
                st.warning(f"공휴일 날짜 형식 오류: '{date_str.strip()}'는 유효한 'YYYY-MM-DD' 형식이 아닙니다. 무시됩니다.")

    # 회의 시작 시간, 종료 시간, 점심시간 설정
    DAILY_MEETING_START = datetime.time(10, 0)  # 오전 10시
    LUNCH_START = datetime.time(11, 30)        # 오전 11시 30분
    LUNCH_END = datetime.time(13, 0)           # 오후 1시 (13시)
    DAILY_MEETING_END = datetime.time(17, 0)    # 오후 5시 (17시)

    # 이 리스트에 최종적으로 배정된 과제들을 엑셀 형식 + '회의일시'로 담을 거예요.
    final_scheduled_tasks_for_display = []
    
    log_messages = [] # 로그 메시지를 저장할 리스트

    # 과제 리스트를 deque로 변환 (원본 데이터프레임 보호를 위해 복사본 사용)
    tasks_to_schedule = deque(tasks_df.to_dict('records'))

    # 요청자별 과제 덱을 관리하여 연이어 배치할 수 있도록 준비
    tasks_by_requester = {}
    for task in tasks_to_schedule:
        requester = task.get('과제요청자', '미정')
        if requester not in tasks_by_requester:
            tasks_by_requester[requester] = deque()
        tasks_by_requester[requester].append(task)
    
    current_date = start_date_obj
    working_days_scheduled = 0 # 회의를 배정한 근무일 수 카운트
    max_working_days = 4 # 최대 회의 가능 근무일
    
    last_scheduled_requester = None # 마지막으로 회의를 배정한 요청자
    
    log_messages.append(f"\n✨ {start_date_obj.strftime('%Y-%m-%d')}부터 최대 {max_working_days} 근무일 동안의 회의 일정을 생성합니다! ✨\n")

    # 최대 근무일 또는 모든 과제가 배정될 때까지 반복
    while working_days_scheduled < max_working_days and tasks_to_schedule:
        # 주말(토요일=5, 일요일=6) 또는 공휴일은 회의에서 제외
        if current_date.weekday() >= 5 or current_date in parsed_holidays:
            log_messages.append(f"🗓️ {current_date.strftime('%Y-%m-%d')}은 주말/공휴일이라 회의가 없어요! 쉬세요~ 😌")
            current_date += datetime.timedelta(days=1)
            continue

        # 근무일 카운트 증가
        working_days_scheduled += 1

        current_meeting_time = datetime.datetime.combine(current_date, DAILY_MEETING_START)
        end_of_day_meeting_limit = datetime.datetime.combine(current_date, DAILY_MEETING_END)

        log_messages.append(f"--- 📅 {current_date.strftime('%Y년 %m월 %d일')} 일정 (근무일 {working_days_scheduled}/{max_working_days}) ---")

        # 그 날의 회의 가능 시간 동안 과제를 배정
        while current_meeting_time < end_of_day_meeting_limit and tasks_to_schedule:
            
            # 현재 시간이 점심시간 시작 전이고, 다음 회의 시작 시간이 점심시간과 겹치면 점심시간 이후로 이동
            if current_meeting_time.time() >= LUNCH_START and current_meeting_time.time() < LUNCH_END :
                log_messages.append(f"  ⏰ {current_meeting_time.strftime('%H:%M')}: 점심시간이에요! 밥 먹고 만나요~ 😋")
                current_meeting_time = datetime.datetime.combine(current_date, LUNCH_END)
                # 점심시간 이후로 이동했는데, 이미 하루 종료 시간을 넘겼다면 해당 날짜 회의 종료
                if current_meeting_time >= end_of_day_meeting_limit:
                    break
            
            # --- 과제 선택 로직 (동일 요청자 연이어 배치) ---
            next_task = None
            if last_scheduled_requester and last_scheduled_requester in tasks_by_requester and tasks_by_requester[last_scheduled_requester]:
                # 마지막 요청자의 다음 과제가 있다면 그것을 우선 선택
                next_task = tasks_by_requester[last_scheduled_requester][0]
            
            if next_task is None: # 마지막 요청자의 과제가 없거나, 처음이라면 전체 덱에서 선택
                next_task = tasks_to_schedule[0]
            
            task_duration_minutes = next_task['회의지속시간']

            # 실제 회의가 끝나는 시간 계산 (점심시간 고려)
            potential_task_end_time = current_meeting_time + datetime.timedelta(minutes=task_duration_minutes)
            actual_task_end_time = potential_task_end_time # 초기화

            # 회의가 점심시간을 걸치는지 확인
            if current_meeting_time.time() < LUNCH_END and potential_task_end_time.time() > LUNCH_START:
                # 점심시간 전까지의 회의 시간
                duration_before_lunch = datetime.timedelta(0)
                if current_meeting_time.time() < LUNCH_START: # 현재 회의 시작 시간이 점심시간 전이라면
                    duration_before_lunch = datetime.datetime.combine(current_date, LUNCH_START) - current_meeting_time
                
                # 점심시간 이후에 남은 회의 시간
                remaining_duration_after_lunch_needed = datetime.timedelta(minutes=task_duration_minutes) - duration_before_lunch
                
                # 점심시간 이후 회의 시작 시간
                start_time_after_lunch = datetime.datetime.combine(current_date, LUNCH_END)
                
                actual_task_end_time = start_time_after_lunch + remaining_duration_after_lunch_needed

            # 계산된 회의 종료 시간이 하루 회의 종료 시간을 넘는지 확인
            # 또는, 당일 마지막 회의가 4시 30분인데 회의지속시간이 50분이면 다음날로 넘김
            if actual_task_end_time > end_of_day_meeting_limit:
                log_messages.append(f"  ⏰ {current_meeting_time.strftime('%H:%M')}: '{next_task['과제명']}' 과제가 오늘 다 들어가지 않아서 다음 날로 미뤄져요! ➡️")
                break # 다음 날로 넘어가기 위해 현재 날짜의 일정 배정 중단
            
            # 과제 배정 및 덱에서 제거
            if next_task == tasks_to_schedule[0]: # 전체 덱에서 가져온 경우
                task_to_add = tasks_to_schedule.popleft()
                # 요청자별 덱에서도 제거 (동일 객체 참조이므로 찾아서 제거)
                requester_of_task = task_to_add.get('과제요청자', '미정')
                if requester_of_task in tasks_by_requester and task_to_add in tasks_by_requester[requester_of_task]:
                    tasks_by_requester[requester_of_task].remove(task_to_add) # remove는 O(N)이지만 덱 크기가 작으므로 괜찮음
            else: # 요청자별 덱에서 가져온 경우
                task_to_add = tasks_by_requester[last_scheduled_requester].popleft()
                # 전체 덱에서도 제거 (동일 객체 참조이므로 찾아서 제거)
                if task_to_add in tasks_to_schedule:
                    tasks_to_schedule.remove(task_to_add)

            # '회의일시' 문자열 포맷팅
            meeting_datetime_str = (
                f"{current_date.strftime('%Y-%m-%d')} "
                f"{current_meeting_time.strftime('%H:%M')} - "
                f"{actual_task_end_time.strftime('%H:%M')}"
            )

            # 배정된 과제를 위한 딕셔너리 생성 (원본 엑셀 컬럼 + 회의일시)
            scheduled_task_entry = {
                '과제번호': task_to_add.get('과제번호', '-'), # .get()을 사용하여 컬럼이 없을 경우 기본값 설정
                '과제명': task_to_add['과제명'],
                '과제요청자': task_to_add.get('과제요청자', '-'),
                '소속팀': task_to_add.get('소속팀', '-'),
                '회의지속시간': task_to_add['회의지속시간'],
                '회의일시': meeting_datetime_str # 새로 추가된 컬럼!
            }
            final_scheduled_tasks_for_display.append(scheduled_task_entry)

            log_messages.append(f"  ✅ {current_meeting_time.strftime('%H:%M')} - {actual_task_end_time.strftime('%H:%M')}: '{task_to_add['과제명']}' ({task_to_add['회의지속시간']}분)")
            current_meeting_time = actual_task_end_time # 다음 회의 시작 시간 업데이트
            last_scheduled_requester = task_to_add.get('과제요청자', '미정') # 마지막 요청자 업데이트
        
        current_date += datetime.timedelta(days=1)
        
        if not tasks_to_schedule and working_days_scheduled <= max_working_days:
            log_messages.append(f"\n🎉 모든 과제가 {working_days_scheduled} 근무일 내에 배정 완료되었어요! 🥳")
            break # 모든 과제가 배정되었으면 루프 종료

    if tasks_to_schedule:
        log_messages.append("\n⚠️ 아쉽게도 모든 과제를 배정하지 못했어요! 다음 회의 기간에 이어서 배정해야 할 것 같아요. 😥")
        # 미배정 과제는 함수 반환 시 함께 전달될 거예요.
    else:
        log_messages.append("\n🎉 모든 과제 배정이 성공적으로 완료되었어요! 수고했어요! 🎉")

    # 미배정된 과제들을 리스트로 변환하여 반환
    remaining_tasks_list = list(tasks_to_schedule)

    return final_scheduled_tasks_for_display, log_messages, remaining_tasks_list


if "scheduled" not in st.session_state:
    st.session_state.scheduled = {
        "scheduled_tasks_list": [],
        "log_messages": [],
        "remaining_tasks": []
    }

# --- Streamlit 웹 애플리케이션 ---
st.set_page_config(layout="wide") # 페이지 레이아웃을 넓게 설정
st.title("🗓️ 과제확정회의 일정 자동 배정기 🗓️")
st.markdown("엑셀 파일에서 과제 리스트를 읽어와, 지정된 기간 동안 점심시간, 주말, 공휴일을 제외하고 회의 일정을 자동으로 생성해 드려요! **동일 요청자 과제는 연이어 배치**되며, **최대 4일 근무일** 내에 일정이 배정됩니다.")

# 엑셀 파일 업로드
st.header("1. 과제 리스트 엑셀 파일 업로드")
uploaded_file = st.file_uploader("여기에 'tasks.xlsx' 파일을 업로드해주세요.", type=['xlsx'])

tasks_df = pd.DataFrame()
if uploaded_file:
    try:
        tasks_df = pd.read_excel(uploaded_file)
        st.success("✅ 엑셀 파일이 성공적으로 업로드되었습니다!")
        st.write("---") # 구분선 추가
        st.subheader("업로드된 과제 미리보기")
        st.dataframe(tasks_df.head(), use_container_width=True) # 미리보기
        st.write("---") # 구분선 추가
        
        # 필요한 컬럼 확인
        required_columns = ['과제번호', '과제명', '과제요청자', '소속팀', '회의지속시간']
        if not all(col in tasks_df.columns for col in required_columns):
            st.error(f"⚠️ 엑셀 파일에 필요한 컬럼이 모두 없어요! 다음 컬럼들이 필요해요: {required_columns}")
            tasks_df = pd.DataFrame() # 유효하지 않으면 데이터프레임 초기화
        else:
            st.info(f"총 {len(tasks_df)}개의 과제가 로드되었습니다.")

    except Exception as e:
        st.error(f"엑셀 파일을 읽는 중 오류가 발생했어요: {e}")
        tasks_df = pd.DataFrame()

# 회의 기간 및 공휴일 입력
st.header("2. 회의 기간 및 공휴일 설정")
col1, col2 = st.columns(2)
with col1:
    start_date_input = st.date_input("회의 시작일", datetime.date.today())
with col2:
    st.markdown("최대 4일 근무일 동안 회의가 진행됩니다.")
    holiday_list_str = st.text_area(
        "공휴일 입력 (콤마로 구분, YYYY-MM-DD 형식)",
        value=f"{datetime.date.today().year}-01-01, {datetime.date.today().year}-03-01, {datetime.date.today().year}-05-05" # 예시 공휴일
    )

# 일정 생성 버튼
if st.button("✨ 회의 일정 생성하기 ✨", type="primary") and not tasks_df.empty:
    st.subheader("3. 생성된 회의 일정")

    # 일정 생성 로직 호출
    scheduled_tasks_list, log_messages, remaining_tasks = schedule_meetings_logic(
        start_date_input, tasks_df.copy(), holiday_list_str # tasks_df.copy()로 원본 보호
    )
    
    st.session_state.scheduled = {
        "scheduled_tasks_list": scheduled_tasks_list,
        "log_messages": log_messages,
        "remaining_tasks": remaining_tasks
    }


if st.session_state.scheduled["scheduled_tasks_list"]:
    scheduled_tasks_list = st.session_state.scheduled["scheduled_tasks_list"]
    log_messages = st.session_state.scheduled["log_messages"]
    remaining_tasks = st.session_state.scheduled["remaining_tasks"]

    # 로그 메시지 출력
    with st.expander("생성 과정 로그 보기"):
        for msg in log_messages:
            st.write(msg)

    # 결과 출력: 엑셀 형식에 '회의일시'가 추가된 표 형태로!
    

    if scheduled_tasks_list:
        st.success("🎉 회의 일정이 성공적으로 생성되었어요! 아래 표에서 확인해주세요. 🎉")
        scheduled_df = pd.DataFrame(scheduled_tasks_list)
        # 컬럼 순서 재조정 (회의일시를 앞으로)
        cols = ['회의일시'] + [col for col in scheduled_df.columns if col != '회의일시']
        scheduled_df = scheduled_df[cols]
        st.dataframe(scheduled_df, use_container_width=True) # 표 형태로 출력
        
    else:
        st.warning("일정을 생성하지 못했어요. 엑셀 파일과 날짜를 다시 확인해주세요. 😥")

    # 미배정 과제 목록 출력
    if remaining_tasks:
        st.error("⚠️ 아쉽게도 모든 과제를 배정하지 못했어요! 다음 회의 기간에 이어서 배정해야 할 것 같아요.")
        st.write("미배정된 과제 목록:")
        remaining_tasks_df = pd.DataFrame(remaining_tasks)
        st.dataframe(remaining_tasks_df, use_container_width=True)

    # --- Azure OpenAI를 이용한 요약 ---
    st.subheader("4. Azure OpenAI를 이용한 일정 요약 (선택 사항)")
    openai_client = get_azure_openai_client()
    if openai_client:
        if st.button("AI 요약 생성하기 🧠"):
            with st.spinner("AI가 일정을 요약 중이에요..."):
                try:
                    # 요약할 내용 준비 (생성된 표 데이터를 활용)
                    summary_text = "생성된 회의 일정:\n"

                    if scheduled_tasks_list:
                        for task in scheduled_tasks_list:
                            summary_text += f"- {task['회의일시']}: {task['과제명']} ({task['회의지속시간']}분) / 요청자: {task['과제요청자']}\n"
                    else:
                        summary_text += "(배정된 회의 없음)\n"
                    
                    if remaining_tasks:
                        summary_text += "\n미배정 과제:\n"
                        for task in remaining_tasks:
                            summary_text += f"- {task['과제명']} ({task['회의지속시간']}분) / 요청자: {task['과제요청자']}\n"

                    # GPT-4.1 호출
                    response = openai_client.chat.completions.create(
                        model=AZURE_OPENAI_DEPLOYMENT_NAME,
                        messages=[
                            {"role": "system", "content": "당신은 회의 일정을 분석하고 친절하게 요약해주는 유능한 비서입니다."},
                            {"role": "user", "content": f"다음 회의 일정을 친절하고 간략하게 요약해주세요:\n\n{summary_text}"}
                        ],
                        temperature=0.7,
                        max_tokens=500
                    )
                    st.success("AI 요약이 완료되었습니다!")
                    st.info(response.choices[0].message.content)

                except Exception as e:
                    st.error(f"AI 요약 생성 중 오류가 발생했어요. Azure OpenAI 설정과 배포 이름을 확인해주세요: {e}")
    else:
        st.info("Azure OpenAI 설정을 완료하면 AI 요약 기능을 사용할 수 있어요! (환경 변수를 설정해주세요)")

elif st.button("✨ 회의 일정 생성하기 ✨") and tasks_df.empty:
    st.warning("엑셀 파일을 먼저 업로드해주세요! ⬆️")

st.markdown("---")