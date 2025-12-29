import random
import csv

random.seed(42)

TOPICS = [
    "education", "healthcare", "technology", "environment", "travel",
    "food", "sports", "finance", "parenting", "product review",
    "history", "art", "music", "workplace", "self improvement"
]

HUMAN_OPENERS = [
    "Honestly,", "I think", "Yesterday,", "In my experience,", "To be fair,",
    "I remember", "Recently,", "When I was younger,", "I’m not sure, but",
    "This might sound weird, but"
]

HUMAN_STYLES = [
    "a short personal story",
    "a casual opinion with small contradictions",
    "a message with a few filler words",
    "a slightly imperfect explanation",
    "a note with uneven sentence lengths"
]

AI_OPENERS = [
    "In this article,", "This section discusses", "It is important to note that",
    "Overall,", "Furthermore,", "In conclusion,", "From a broader perspective,",
    "This analysis highlights", "The following points demonstrate",
    "To summarize,"
]

AI_STYLES = [
    "structured academic tone",
    "formal report style",
    "methodology + implication",
    "neutral and polished explanation",
    "well-balanced multi-point argument"
]

def human_sentence(topic: str) -> str:
    opener = random.choice(HUMAN_OPENERS)
    style = random.choice(HUMAN_STYLES)
    quirks = random.choice([
        "kinda", "maybe", "actually", "like", "sort of", "a bit"
    ])
    add_comma = random.random() < 0.55  # 故意增加逗號機率
    exclaim = "!" if random.random() < 0.18 else "."
    extra = random.choice([
        "and then I changed my mind halfway.",
        "but I could be wrong.",
        "so I wrote it down.",
        "and it surprised me.",
        "and I’m still figuring it out."
    ])
    base = f"{opener} {style} about {topic} {quirks}"
    if add_comma:
        base += ","
    return f"{base} {extra}{exclaim}"

def ai_sentence(topic: str) -> str:
    opener = random.choice(AI_OPENERS)
    style = random.choice(AI_STYLES)
    add_comma = random.random() < 0.75
    multi_point = random.random() < 0.55
    if multi_point:
        points = random.sample([
            "clarity", "efficiency", "sustainability", "risk mitigation",
            "stakeholder alignment", "cost-effectiveness", "scalability",
            "ethical considerations", "data quality", "user experience"
        ], k=3)
        mid = f"{opener} a {style} regarding {topic}"
        if add_comma:
            mid += ","
        return f"{mid} focusing on {points[0]}, {points[1]}, and {points[2]}."
    else:
        mid = f"{opener} {topic} can be evaluated using consistent criteria"
        if add_comma:
            mid += ","
        tail = random.choice([
            "which supports reliable decision-making.",
            "thereby improving overall outcomes.",
            "while maintaining a coherent structure.",
            "to ensure replicable conclusions.",
            "as reflected in the results."
        ])
        return f"{mid} {tail}"

def make_paragraph(generator, topic: str) -> str:
    # 2~5 句組成一段，句長與風格會更像真實資料
    n = random.randint(2, 5)
    sents = [generator(topic) for _ in range(n)]
    # 偶爾插入短句，讓 Human 更人類、AI 更制式
    if generator == human_sentence and random.random() < 0.35:
        sents.insert(random.randint(0, len(sents)), random.choice(["Anyway.", "Not sure.", "That’s it."]))
    if generator == ai_sentence and random.random() < 0.35:
        sents.append(random.choice([
            "This provides a clear basis for further discussion.",
            "These findings align with prior observations.",
            "Future work may refine these estimates."
        ]))
    return " ".join(sents)

def generate_dataset(n_each=150, out_path="sample_data.csv"):
    rows = []
    # Human
    for _ in range(n_each):
        topic = random.choice(TOPICS)
        text = make_paragraph(human_sentence, topic)
        rows.append((text, "Human"))

    # AI
    for _ in range(n_each):
        topic = random.choice(TOPICS)
        text = make_paragraph(ai_sentence, topic)
        rows.append((text, "AI"))

    random.shuffle(rows)

    # ✅ 用 csv 模組寫入，會自動處理逗號與引號（避免你之前的欄位錯誤）
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    print(f"✅ Generated: {out_path} (total rows: {len(rows)})")

if __name__ == "__main__":
    generate_dataset(n_each=150, out_path="sample_data.csv")
