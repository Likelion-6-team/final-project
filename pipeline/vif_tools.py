import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import os

# VIF 계산 함수
def calculate_vif(df):
    df_const = add_constant(df)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df_const.values, i + 1) for i in range(df.shape[1])]
    return vif_data

# 계단식 제거 및 VIF 테이블 및 제거 로그 저장
def stepwise_vif_table(df, min_cols, output_path="vif_table_wide.csv", log_path="vif_removed_log.csv"):
    """
    VIF 계단식 제거 및 테이블+로그 저장

    Parameters:
    - df: DataFrame (수치형 컬럼만)
    - min_cols: 최소 컬럼 수까지 제거 (기본값 5)
    - output_path: VIF 테이블 저장 경로
    - log_path: 제거 로그 저장 경로
    """
    current_df = df.copy()
    vif_dict = {}
    vif_log = []

    while current_df.shape[1] > min_cols:
        col_count = current_df.shape[1]

        vif_result = calculate_vif(current_df)
        vif_series = vif_result.set_index("Variable")["VIF"]
        vif_dict[f"{col_count}개"] = vif_series

        # 제거할 컬럼
        max_vif_variable = vif_series.sort_values(ascending=False).index[0]
        max_vif_value = vif_series.loc[max_vif_variable]

        vif_log.append({
            "제거된_컬럼": max_vif_variable,
            "해당_VIF": max_vif_value,
            "제거_시점_컬럼수": col_count,
            "남은_컬럼수": col_count - 1
        })

        current_df = current_df.drop(columns=[max_vif_variable])

    # 마지막 단계도 저장
    vif_result = calculate_vif(current_df)
    vif_series = vif_result.set_index("Variable")["VIF"]
    vif_dict[f"{min_cols}개"] = vif_series

    # 계단식 wide VIF 테이블
    vif_df = pd.DataFrame(vif_dict)
    vif_df.to_csv(output_path, encoding='utf-8-sig')

    # 제거 로그 저장
    vif_log_df = pd.DataFrame(vif_log)
    vif_log_df.to_csv(log_path, index=False, encoding='utf-8-sig')

    print(f"VIF 테이블 저장 완료: {output_path}")
    print(f"제거 로그 저장 완료: {log_path}")
    return vif_df, vif_log_df


"""
# 예시
from pipeline.vif_tools import stepwise_vif_table
import pandas as pd

# 데이터 불러오기
df = pd.read_csv('./data/my_data.csv')

# 수치형 컬럼만 선택 (VIF는 수치형만 계산 가능)
numeric_df = df.select_dtypes(include='number')

# VIF 계단식 테이블 생성 및 저장
vif_df, vif_log_df = stepwise_vif_table(
    df=numeric_df,
    min_cols=5,
    output_path="./result/vif_table_wide.csv",
    log_path="./result/vif_removed_log.csv"
)
# 결과 확인
print(vif_df.head())
"""