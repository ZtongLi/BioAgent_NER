import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run biomedical expert-agent experiments.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", default="full", choices=[
        "full", "without_retrieval", "compare", "all",
        "single_expert", "candidate_union", "vote_only", "without_voting", "generic_ensemble"])
    parser.add_argument("--limit", type=int, help="Override the number of evaluation samples")
    parser.add_argument("--sentence", help="Predict one sentence without saving results")
    parser.add_argument("--verbose", action="store_true", help="Print full expert experiment JSON instead of the compact summary")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    config["_config_path"] = args.config
    if config.get("agent_system") == "expert_agent":
        from expert_agent.experiment import predict_sentence, run_experiment
        result = (predict_sentence(args.sentence, config, args.variant) if args.sentence is not None
                  else run_experiment(config, args.variant, args.limit))
        if args.verbose or args.sentence is not None:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            from expert_agent.console import print_summary
            print_summary(result)
        return
    parser.error("main.py requires an expert_agent config; run baseline configs with baseline/llm.py")


if __name__ == "__main__":
    main()
