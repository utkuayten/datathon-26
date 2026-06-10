import re
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

TARGET = "career_success_score"

POS_WORDS = [
    "mükemmel", "olağanüstü", "etkileyici", "yüksek", "başarı", "güçlü",
    "dikkat çekici", "umut verici", "değerli", "potansiyel", "sağlam",
    "derin", "ustalık", "üstün", "aday",
]
NEG_WORDS = [
    "ancak", "ama", "fakat", "daha fazla", "gerekiyor", "gerekmekte",
    "gerektiği", "geliştirilmesi", "geliştirmesi", "ihtiyaç", "pratik",
    "çaba", "eksik", "zayıf", "açık",
]


def add_structured_features(df):
    df = df.copy()

    technical_cols = [
        "coding_score", "problem_solving_score", "data_structures_score",
        "sql_score", "machine_learning_score", "backend_score",
        "frontend_score", "cloud_score", "devops_score",
    ]
    soft_cols = [
        "communication_score", "teamwork_score", "leadership_score",
        "presentation_score",
    ]
    readiness_cols = [
        "portfolio_score", "linkedin_profile_score", "cv_quality_score",
        "technical_interview_score", "hr_interview_score",
    ]

    df["technical_avg"]    = df[technical_cols].mean(axis=1)
    df["soft_skill_avg"]   = df[soft_cols].mean(axis=1)
    df["career_readiness"] = df[readiness_cols].mean(axis=1)
    df["interview_avg"]    = (
        df["technical_interview_score"] + df["hr_interview_score"]
    ) / 2

    df["experience_total"] = (
        df["internship_count"].fillna(0)
        + df["freelance_project_count"].fillna(0)
        + df["real_client_project_count"].fillna(0)
        + df["hackathon_count"].fillna(0)
    )
    df["github_power"] = (
        df["github_repo_count"].fillna(0) * df["github_avg_stars"].fillna(0)
    )
    df["application_efficiency"] = (
        df["interviews_attended"] / (df["applications_sent"] + 1)
    )
    return df


def add_lexicon_features(df):
    df = df.copy()
    s = df["mentor_feedback_text"].fillna("").str.lower()

    df["txt_len"]     = s.str.len()
    df["txt_words"]   = s.str.split().apply(len)
    df["pos_cnt"]     = sum(s.str.count(r"\b" + re.escape(w)) for w in POS_WORDS)
    df["neg_cnt"]     = sum(s.str.count(r"\b" + re.escape(w)) for w in NEG_WORDS)
    df["has_ancak"]   = s.str.contains(r"\bancak").astype(int)
    df["has_top_pos"] = s.str.contains(r"\bmükemmel|olağanüstü").astype(int)
    df["sentiment"]   = df["pos_cnt"] - df["neg_cnt"]
    df["sent_ratio"]  = (df["pos_cnt"] + 1) / (df["neg_cnt"] + 1)
    return df


def add_tfidf_features(X_train, X_test, y_train, n_splits=5, random_state=42):
    text_tr = X_train["mentor_feedback_text"].fillna("")
    text_te = X_test["mentor_feedback_text"].fillna("")

    def make_word_vec():
        return TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=2)

    def make_char_vec():
        return TfidfVectorizer(
            max_features=6000, ngram_range=(3, 5), analyzer="char_wb", min_df=3
        )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_w = np.zeros(len(X_train))
    oof_c = np.zeros(len(X_train))

    for tr_idx, va_idx in kf.split(X_train):
        vw = make_word_vec()
        Xw = vw.fit_transform(text_tr.iloc[tr_idx])
        oof_w[va_idx] = (
            Ridge(alpha=2.0)
            .fit(Xw, y_train[tr_idx])
            .predict(vw.transform(text_tr.iloc[va_idx]))
        )

        vc = make_char_vec()
        Xc = vc.fit_transform(text_tr.iloc[tr_idx])
        oof_c[va_idx] = (
            Ridge(alpha=2.0)
            .fit(Xc, y_train[tr_idx])
            .predict(vc.transform(text_tr.iloc[va_idx]))
        )

    vw = make_word_vec()
    rw = Ridge(alpha=2.0).fit(vw.fit_transform(text_tr), y_train)
    te_w = rw.predict(vw.transform(text_te))

    vc = make_char_vec()
    rc = Ridge(alpha=2.0).fit(vc.fit_transform(text_tr), y_train)
    te_c = rc.predict(vc.transform(text_te))

    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train["text_pred_w"] = oof_w
    X_train["text_pred_c"] = oof_c
    X_test["text_pred_w"]  = te_w
    X_test["text_pred_c"]  = te_c

    return X_train, X_test


def prepare_features(X_train, X_test, y_train):
    X_train = add_structured_features(X_train)
    X_test  = add_structured_features(X_test)

    X_train = add_lexicon_features(X_train)
    X_test  = add_lexicon_features(X_test)

    X_train, X_test = add_tfidf_features(X_train, X_test, y_train)

    X_train = X_train.drop(columns=["mentor_feedback_text"])
    X_test  = X_test.drop(columns=["mentor_feedback_text"])

    cat_features = [
        "department", "university_tier", "target_role",
        "hobby", "preferred_social_media_platform",
    ]
    for c in cat_features:
        X_train[c] = X_train[c].fillna("missing").astype(str)
        X_test[c]  = X_test[c].fillna("missing").astype(str)

    assert list(X_train.columns) == list(X_test.columns)
    print("features:", X_train.shape[1])

    return X_train, X_test, cat_features
