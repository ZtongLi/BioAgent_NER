import json
import os
from openai import OpenAI


def get_test_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_planner_prompt(sentence):
    prompt = f"""
You are a planner for biomedical named entity recognition.

Sentence:
{sentence}

Select only the expert agents that are likely needed to identify named biomedical entities in this sentence.

Experts:
- Molecular Biology Expert: genes, proteins, and related molecular entities
- Clinical Medicine Expert: diseases, disorders, and pathological conditions
- Pharmacology Expert: drugs, chemicals, and pharmacological substances

If no relevant named biomedical entity is likely present, return an empty list.
Multiple experts may be selected.

Return JSON only:
{{"experts": []}}
"""
    return prompt


def QA_KIMI(prompt, model_name):
    client = OpenAI(
        api_key=os.getenv("MOONSHOT_API_KEY"),
        base_url="https://api.moonshot.cn/v1"
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=1
    )

    answer = response.choices[0].message.content

    return answer