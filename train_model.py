"""
train_model.py
==============
Trains a TF-IDF + Logistic Regression intent classifier and saves it.

Intents
-------
  emergency        – user needs urgent help
  contact_mom      – user wants to reach Mom
  contact_dad      – user wants to reach Dad
  contact_doctor   – user wants to reach Doctor
  unknown          – unrecognised / out-of-scope

Run once:  python3 train_model.py
Produces:  model/intent_model.pkl  (classifier)
           model/tfidf_vectorizer.pkl
           model/label_encoder.pkl
"""

import os, pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Training corpus ───────────────────────────────────────────────────────────
TRAINING_DATA = [
    # ── emergency ────────────────────────────────────────────────────────────
    ("help me",                             "emergency"),
    ("help",                                "emergency"),
    ("emergency",                           "emergency"),
    ("I need help",                         "emergency"),
    ("SOS",                                 "emergency"),
    ("please help me",                      "emergency"),
    ("I am in danger",                      "emergency"),
    ("there is an accident",                "emergency"),
    ("I fell down",                         "emergency"),
    ("I have fallen",                       "emergency"),
    ("chest pain",                          "emergency"),
    ("I am having a heart attack",          "emergency"),
    ("I am hurt",                           "emergency"),
    ("save me",                             "emergency"),
    ("call ambulance",                      "emergency"),
    ("I need an ambulance",                 "emergency"),
    ("send help",                           "emergency"),
    ("fire fire",                           "emergency"),
    ("there is a fire",                     "emergency"),
    ("I am bleeding",                       "emergency"),
    ("I can't breathe",                     "emergency"),
    ("I am feeling very sick",              "emergency"),
    ("I am unconscious send help",          "emergency"),
    ("emergency please",                    "emergency"),
    ("urgent help needed",                  "emergency"),
    ("I am in trouble",                     "emergency"),
    ("accident happened",                   "emergency"),
    ("I got injured",                       "emergency"),
    ("something bad happened",              "emergency"),
    ("I am not well send help",             "emergency"),
    ("mujhe madad chahiye",                 "emergency"),
    ("bachao",                              "emergency"),
    ("help send emergency",                 "emergency"),
    ("emergency send message",              "emergency"),
    ("send emergency alert",                "emergency"),
    ("send sos",                            "emergency"),
    ("send help alert",                     "emergency"),

    # ── contact_mom ──────────────────────────────────────────────────────────
    ("call mom",                            "contact_mom"),
    ("call my mom",                         "contact_mom"),
    ("call mother",                         "contact_mom"),
    ("message mom",                         "contact_mom"),
    ("whatsapp mom",                        "contact_mom"),
    ("send message to mom",                 "contact_mom"),
    ("contact mom",                         "contact_mom"),
    ("talk to mom",                         "contact_mom"),
    ("reach mom",                           "contact_mom"),
    ("call mummy",                          "contact_mom"),
    ("send to mummy",                       "contact_mom"),
    ("message mummy",                       "contact_mom"),
    ("call maa",                            "contact_mom"),
    ("maa ko bulao",                        "contact_mom"),
    ("send alert to mom",                   "contact_mom"),
    ("emergency call mom",                  "contact_mom"),
    ("emergency send to mom",               "contact_mom"),
    ("send emergency to mom",               "contact_mom"),
    ("help call mom",                       "contact_mom"),
    ("mom please help",                     "contact_mom"),
    ("need mom",                            "contact_mom"),
    ("i want to call mom",                  "contact_mom"),
    ("connect me to mom",                   "contact_mom"),
    ("mama",                                "contact_mom"),
    ("call mama",                           "contact_mom"),

    # ── contact_dad ──────────────────────────────────────────────────────────
    ("call dad",                            "contact_dad"),
    ("call my dad",                         "contact_dad"),
    ("call father",                         "contact_dad"),
    ("message dad",                         "contact_dad"),
    ("whatsapp dad",                        "contact_dad"),
    ("send message to dad",                 "contact_dad"),
    ("contact dad",                         "contact_dad"),
    ("talk to dad",                         "contact_dad"),
    ("reach dad",                           "contact_dad"),
    ("call papa",                           "contact_dad"),
    ("send to papa",                        "contact_dad"),
    ("message papa",                        "contact_dad"),
    ("call pita",                           "contact_dad"),
    ("papa ko bulao",                       "contact_dad"),
    ("send alert to dad",                   "contact_dad"),
    ("emergency call dad",                  "contact_dad"),
    ("emergency send to dad",               "contact_dad"),
    ("send emergency to dad",               "contact_dad"),
    ("help call dad",                       "contact_dad"),
    ("dad please help",                     "contact_dad"),
    ("need dad",                            "contact_dad"),
    ("i want to call dad",                  "contact_dad"),
    ("connect me to dad",                   "contact_dad"),
    ("bapu",                                "contact_dad"),
    ("call bapu",                           "contact_dad"),

    # ── contact_doctor ───────────────────────────────────────────────────────
    ("call doctor",                         "contact_doctor"),
    ("call the doctor",                     "contact_doctor"),
    ("call my doctor",                      "contact_doctor"),
    ("message doctor",                      "contact_doctor"),
    ("whatsapp doctor",                     "contact_doctor"),
    ("send message to doctor",              "contact_doctor"),
    ("contact doctor",                      "contact_doctor"),
    ("talk to doctor",                      "contact_doctor"),
    ("call dr",                             "contact_doctor"),
    ("send alert to doctor",                "contact_doctor"),
    ("emergency call doctor",               "contact_doctor"),
    ("emergency send to doctor",            "contact_doctor"),
    ("send emergency to doctor",            "contact_doctor"),
    ("need a doctor",                       "contact_doctor"),
    ("i need doctor",                       "contact_doctor"),
    ("get doctor",                          "contact_doctor"),
    ("call physician",                      "contact_doctor"),
    ("call nurse",                          "contact_doctor"),
    ("doctor help",                         "contact_doctor"),
    ("medical help call doctor",            "contact_doctor"),
    ("doctor please",                       "contact_doctor"),
    ("connect me to doctor",                "contact_doctor"),

    # ── unknown ──────────────────────────────────────────────────────────────
    ("hello",                               "unknown"),
    ("hi there",                            "unknown"),
    ("what is the weather",                 "unknown"),
    ("tell me a joke",                      "unknown"),
    ("good morning",                        "unknown"),
    ("what time is it",                     "unknown"),
    ("play music",                          "unknown"),
    ("open youtube",                        "unknown"),
    ("set alarm",                           "unknown"),
    ("remind me later",                     "unknown"),
    ("how are you",                         "unknown"),
    ("what is your name",                   "unknown"),
    ("abcdef",                              "unknown"),
    ("nothing",                             "unknown"),
    ("test",                                "unknown"),
    ("blah blah",                           "unknown"),
]

# ── Build X, y ────────────────────────────────────────────────────────────────
texts  = [t for t, _ in TRAINING_DATA]
labels = [l for _, l in TRAINING_DATA]

# ── Encode labels ─────────────────────────────────────────────────────────────
le = LabelEncoder()
y  = le.fit_transform(labels)

# ── TF-IDF vectoriser (character n-grams catch Hinglish/typos) ────────────────
tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 4),
    max_features=8000,
    sublinear_tf=True,
)
X = tfidf.fit_transform(texts)

# ── Train / eval split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

clf = LogisticRegression(
    max_iter=1000,
    C=5.0,
    class_weight="balanced",
    solver="lbfgs",
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print(f"Test accuracy: {(y_pred == y_test).mean():.2%}\n")

# ── Save artefacts ────────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

with open("model/intent_model.pkl",      "wb") as f: pickle.dump(clf,   f)
with open("model/tfidf_vectorizer.pkl",  "wb") as f: pickle.dump(tfidf, f)
with open("model/label_encoder.pkl",     "wb") as f: pickle.dump(le,    f)

print("✅  Model artefacts saved to  model/")

# ── Quick sanity checks ───────────────────────────────────────────────────────
samples = [
    "help me",
    "emergency send to mom",
    "call doctor",
    "call papa",
    "good morning",
    "send sos to dad",
    "I am hurt",
    "bachao",
]

X_s = tfidf.transform(samples)
preds = le.inverse_transform(clf.predict(X_s))
probs = clf.predict_proba(X_s).max(axis=1)

print("=== Sanity Checks ===")
for txt, pred, conf in zip(samples, preds, probs):
    print(f"  '{txt}'  →  {pred}  ({conf:.0%})")
