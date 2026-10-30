"""Apply the frozen validation blend to completed final runs, without test reads."""

import json
from simulation import BASE


def main():
    selection = json.loads((BASE / "runs/v2_development/combined_selection.json").read_text())
    tree = json.loads((BASE / "runs/v2_final/selection.json").read_text())
    neural = json.loads((BASE / "runs/v2_neural_final/training_summary.json").read_text())
    assert neural["is_final"] is True and neural["test_used"] is False
    assert selection["iterations"] == tree["iterations"]
    selection["neural_run"] = "v2_neural_final"
    selection["target_mean"] = tree["target_mean"]
    selection["addition_rate"] = tree["addition_rate"]
    (BASE / "runs/v2_final/combined_selection.json").write_text(json.dumps(selection, indent=2))
    print("Final inference configuration frozen; no competition inputs read.")


if __name__ == "__main__":
    main()
