from pathlib import Path
import pandas as pd
import shortuuid

data_path = Path("../Dataset/scientsbank/scientsbank_core.xlsx", dtype=str)
save_path = Path("../Dataset/scientsbank/scientsbank_processed.xlsx")


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


column_to_keep = ['answer_id', 'student_answer', 'accuracy', 'score']
df = pd.read_excel(data_path, usecols=column_to_keep)
df['uid'] = df.apply(lambda x: generate_id(str(x.answer_id).replace('.', '_')), axis=1)

df.to_excel(save_path, index=False)
