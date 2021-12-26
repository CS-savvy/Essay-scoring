from pathlib import Path
import pandas as pd
import shortuuid

data_path = Path("../Dataset/mohler/mohler_formatted.csv", dtype=str)
save_path = Path("../Dataset/mohler/mohler_processed.csv")


def generate_id(prefix: str = None, length: int = 10) -> str:
    """
    Generates unique id

    :argument
        prefix: insert the text in beginning
        length: length of unique id excluding prefix

    :returns
        unique id string.
    """
    if prefix:
        new_id = prefix + "_" + shortuuid.ShortUUID().random(length=length)
        return new_id
    return shortuuid.ShortUUID().random(length=length)


column_to_keep = ['id', 'question', 'desired_answer', 'student_answer', 'score_me', 'score_other', 'score_avg']
df = pd.read_csv(data_path, delimiter=",", encoding='utf8', usecols=column_to_keep)
df['uid'] = df.apply(lambda x: generate_id(str(x.id).replace('.', '_')), axis=1)

df.to_csv(save_path, index=False)
