"""Load MOOCCube, clean it, and save the tables used by all methods."""
import pandas as pd

from config import DATA_RAW, DATA_PROCESSED, MIN_COURSES_PER_USER


def load_raw():
    uc = pd.read_csv(DATA_RAW / "relations" / "user-course.json",
                     sep="\t", header=None, names=["user", "course"])
    cc = pd.read_csv(DATA_RAW / "relations" / "course-concept.json",
                     sep="\t", header=None, names=["course", "concept"])
    return uc, cc


def clean(uc, cc, min_courses=MIN_COURSES_PER_USER):
    """Keep courses that have concepts, then users with enough courses."""
    df = uc[uc["course"].isin(set(cc["course"]))].drop_duplicates()
    n = df.groupby("user")["course"].transform("size")
    return df[n >= min_courses].reset_index(drop=True)


def main():
    uc, cc = load_raw()
    df = clean(uc, cc)
    cc = cc[cc["course"].isin(set(df["course"]))]
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROCESSED / "interactions.csv", index=False)
    cc.to_csv(DATA_PROCESSED / "course_concepts.csv", index=False)
    print("rows:", len(df), "| users:", df["user"].nunique(),
          "| courses:", df["course"].nunique())


if __name__ == "__main__":
    main()