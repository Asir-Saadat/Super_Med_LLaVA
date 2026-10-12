import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DOMAINS = {
    "x_ray":           "/home/as5606/projects/Super_LlaVA_Med/dataset/x_ray/chest_xray_filter_decisions.jsonl",
    "mri":             "/home/as5606/projects/Super_LlaVA_Med/dataset/MRI/mri_filter_decisions.jsonl",
    "histology":       "/home/as5606/projects/Super_LlaVA_Med/dataset/histology/histology_filter_decisions.jsonl",
    "gross_pathology": "/home/as5606/projects/Super_LlaVA_Med/dataset/gross_pathology/gross_filter_decisions.jsonl",
    "ct":              "/home/as5606/projects/Super_LlaVA_Med/dataset/CT/ct_filter_decisions.jsonl",
}

# ── Load ALL questions (no KEEP filter) ───────────────────────────────────────
print("Loading questions...")
texts_all, labels_all = [], []
texts_keep, labels_keep = [], []

for domain, path in DOMAINS.items():
    for line in open(path):
        rec = json.loads(line)
        q = rec["question"].replace("<image>", "").strip()
        texts_all.append(q)
        labels_all.append(domain)
        if rec.get("decision") == "KEEP":
            texts_keep.append(q)
            labels_keep.append(domain)

print(f"\nAll questions : {len(texts_all)}")
print(f"KEEP questions: {len(texts_keep)}")
for domain in DOMAINS:
    n_all  = labels_all.count(domain)
    n_keep = labels_keep.count(domain)
    print(f"  {domain:20s}  all={n_all:6d}  keep={n_keep:5d}")


def run_classifier(texts, labels, label):
    print(f"\n{'='*60}")
    print(f"TF-IDF + Logistic Regression  [{label}]")
    print('='*60)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)
    X_train_v = vec.fit_transform(X_train)
    X_test_v  = vec.transform(X_test)

    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', n_jobs=-1)
    clf.fit(X_train_v, y_train)

    y_pred = clf.predict(X_test_v)
    print(classification_report(y_test, y_pred))
    print("Confusion matrix (rows=true, cols=pred):")
    domain_names = list(DOMAINS.keys())
    print("Labels:", domain_names)
    print(confusion_matrix(y_test, y_pred, labels=domain_names))

    # Top discriminative words per class
    print("\nTop 10 words per domain:")
    feature_names = vec.get_feature_names_out()
    for i, domain in enumerate(clf.classes_):
        top = clf.coef_[i].argsort()[-10:][::-1]
        words = [feature_names[j] for j in top]
        print(f"  {domain:20s}: {', '.join(words)}")

    return clf, vec


run_classifier(texts_all, labels_all, "ALL questions")
run_classifier(texts_keep, labels_keep, "KEEP questions only")
