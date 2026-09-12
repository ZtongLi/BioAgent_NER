from tool import *

test_file_path = "./bioDataset/bc2gm1/dev.json"

model_name = "kimi-k2.6"


# 读取数据
test_data = get_test_data(test_file_path)

for data in test_data[:10]:

    sentence = data["sentence"]

    print("Sentence:")
    print(sentence)
    print("Gold:", data["entities"])

    # Planner
    planner_prompt = get_planner_prompt(sentence)

    planner_answer = QA_KIMI(
        planner_prompt,
        model_name
    )

    print("\nPlanner Answer:")
    print(planner_answer)