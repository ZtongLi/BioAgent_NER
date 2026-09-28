import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run independent biomedical NER agents.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", default="full", choices=[
        "extract_only", "full", "without_discovery", "without_boundary",
        "without_verification", "without_retrieval", "compare", "all"])
    parser.add_argument("--limit", type=int, help="Override the number of evaluation samples")
    parser.add_argument("--sentence", help="Predict one sentence without saving results")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    if config.get("agent_system") != "ner_agent":
        from candidate_discovery_agent.experiment import run_experiment
        from candidate_discovery_agent.tool import candidate_discovery_agent
        if args.sentence is not None:
            print(candidate_discovery_agent(args.sentence, config))
        else:
            if args.limit is not None:
                config["max_loop"] = args.limit
            run_experiment(config)
        return

    from experiment import predict_sentence, run_experiment
    if args.sentence is not None:
        result = predict_sentence(args.sentence, config, args.variant)
    else:
        result = run_experiment(config, args.variant, args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
