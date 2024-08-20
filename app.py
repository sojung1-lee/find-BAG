import streamlit as st
import pandas as pd
import re
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Streamlit 애플리케이션 제목
st.title("기존 BAG 품번 탐색 시스템")


# 추가적인 데이터 처리 및 분석
def replace_number_x_number(text):
    return re.sub(r'\((\d+\s?[-*xX]\s?\d+)\)', r'PLACEHOLDER_\1_PLACEHOLDER', text)

def remove_parentheses_except_number_x_number(text):
    text = replace_number_x_number(text)
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'PLACEHOLDER_(\d+X\d+)_PLACEHOLDER', r'(\1)', text)
    return text

def to_tuple(spec):
    if pd.isna(spec):
        return None
    parts = spec.replace('*', 'X').replace('x', 'X').replace('-', 'X').split('X')
    return (int(parts[0]), int(parts[1]))

def sort_tuple(t):
    return tuple(sorted(t))

        
# 예측값과 label 비교
def find_closest_label(prediction, labels):
    # 각 요소의 절댓값 차이의 합을 계산
    abs_diff_sum = labels['Technical Specification'].apply(lambda x: np.sum(np.abs(prediction - np.array(x))))
    closest_index = abs_diff_sum.idxmin()
    closest_size = labels.iloc[closest_index]['Technical Specification']
    return labels[labels['Technical Specification'] == closest_size]


# 예제 데이터
L = pd.read_excel('Ldata.xlsx', engine='openpyxl')
L = L[['Product','Size']]

# 물건_size와 포장물_size를 각각 분리하여 새로운 컬럼으로 추가
L[['a', 'b', 'c']] = L['Product'].str.strip('()').str.split(',', expand=True)
L[['d', 'e']] = L['Size'].str.strip('()').str.split(',', expand=True)

# 불필요한 컬럼 제거
L = L.drop(columns=['Product', 'Size'])

# 입력 데이터 (X)와 출력 데이터 (y) 분리
X = L[['a', 'b', 'c']]
y = L[['d', 'e']]

## 선형 이진 분류 수행
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8, shuffle=True)

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(X, y)

# 엑셀 데이터 복사 붙여넣기 위젯
excel_text = st.text_area("NPDM 시스템에서 MAF* 품번 List를 엑셀파일로 추출한 후 데이터를 복사하여 붙여넣으시오.")

if excel_text:
    # 텍스트 데이터를 판다스 데이터프레임으로 변환
    from io import StringIO
    excel_data = StringIO(excel_text)
    df = pd.read_csv(excel_data, delimiter='\t')  # 탭으로 구분된 데이터로 읽기

    # 원하는 열만 남기기
    df = df[['Part No.', 'Technical Specification']]

    # (숫자X숫자) 형식은 제외하고 () 지우기
    df['Technical Specification'] = df['Technical Specification'].apply(lambda x: remove_parentheses_except_number_x_number(x) if isinstance(x, str) else x)

    # 특정 문자 지우기
    df['Technical Specification'] = df['Technical Specification'].str.replace(r'\(|\)|\[|\]|,|mm|MM|H|h|W|w|:', '', regex=True)

    # 소숫점 지우기
    df['Technical Specification'] = df['Technical Specification'].str.replace('.0', '')

    # -,*,x,X 앞뒤 숫자 덩어리 형식만 남기기
    df['Technical Specification'] = df['Technical Specification'].str.extract(r'(\d+\s?[-*xX]\s?\d+)')

    # NaN 값을 포함하는 행 지우기
    label = df.dropna(subset=['Technical Specification'])
    label.reset_index(inplace=True, drop=True)

    # 추출한 문자를 tuple 형식으로 바꾸기
    label['Technical Specification'] = label['Technical Specification'].apply(to_tuple)

    # 각 튜플을 오름차순으로 정렬
    label['Technical Specification'] = label['Technical Specification'].apply(sort_tuple)

    # 데이터프레임을 화면에 출력
    st.write("기존 BAG 품번 LIST:")
    st.dataframe(label)

    # 사용자 입력 받기
    st.subheader('BAG 품번 탐색')
    st.markdown('개발하고자하는 부품 Size를 입력하시오. (단, x<y)')
    st.markdown('부품 Size가 아닌 Bag Size로 검색하고 싶다면, z값은 0을 입력하시오')
    a = st.number_input('x 값을 입력하세요', value=0)
    b = st.number_input('y 값을 입력하세요', value=0)
    c = st.number_input('z 값을 입력하세요', value=0)

        
    # 예측 버튼
    if st.button('예측'):
        user_input = np.array([[a, b, c]])
        user_input_df = pd.DataFrame(user_input, columns=['a', 'b', 'c'])
        if c==0:
            prediction = user_input_df[['a','b']].values.flatten()
        else:
            prediction = model.predict(user_input_df).flatten()


        # 예측값과 가장 비슷한 label 찾기
        closest_labels = find_closest_label(prediction, label)
        st.write("가장 유사한 기존 BAG 추천:")
        for _, row in closest_labels.iterrows():
            st.write(f"Size {row['Technical Specification']}, Part No. {row['Part No.']}")

else:
    st.write("엑셀 파일: ctrl+A, ctrl+C  →  입력: ctrl+V, ctrl+enter")

