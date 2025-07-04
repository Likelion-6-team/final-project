
import pandas as pd
from functools import reduce

def merge_segment_df(selected_columns, paths):
    """
    선택된 컬럼들 + Segment 병합 후 ID, 기준년월 제거한 DataFrame 반환

    Parameters:
    - selected_columns: list, 사용할 컬럼명 리스트
    - paths: list, 각 데이터셋의 경로
    Returns:
    - final_df: 병합된 DataFrame (Segment 포함, 기준년월 제외)
    """
    df_list = []

    # Segment 정보 불러오기
    seg_path = [p for p in paths if '1_회원정보' in p][0]
    segment_df = pd.read_parquet(seg_path)[['ID', '기준년월', 'Segment']]

    for path in paths:
        df = pd.read_parquet(path)
        base_cols = ['ID', '기준년월']
        common_cols = list(set(df.columns) & set(selected_columns + base_cols))
        if common_cols:
            df_list.append(df[common_cols])

    merged_df = reduce(lambda left, right: pd.merge(left, right, on=['ID', '기준년월'], how='outer'), df_list)
    merged_df = pd.merge(merged_df, segment_df, on=['ID', '기준년월'], how='left')
    final_df = merged_df.drop(columns=['기준년월'])

    return final_df

"""
# 예시
selected_columns = ['A', 'B', 'C']
paths = [
    './data/train/1_회원정보_train.parquet',
    './data/train/2_신용정보_train.parquet',
    ...
]

df = merge_segment_df(selected_columns, paths)
"""
