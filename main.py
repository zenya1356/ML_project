import argparse
import yaml
from src.data import load_data, split_features_target, validate_data

def main():
    parser = argparse.ArgumentParser(
        description="Launch of the whole project"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to yaml config file",
    )

    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
        
        print("Task: ", config["task"])
        print("Train: ", config["data"]["train_path"])
        print("Target: ", config['data']['target'])
        print("Path to config yaml file: ", args.config)

    data_config = config["data"]

    train_df, test_df = load_data(
        train_path=data_config["train_path"],
        test_path=data_config["test_path"],
    )

    validate_data(
        train_df = train_df,
        test_df = test_df,
        target_column = data_config["target"],
        id_column = data_config["id_column"],
    )

    print("Data structure check passed")

    X, y = split_features_target(
        train_df=train_df,
        target_column=data_config["target"]
    )

    print("train:", train_df.shape)
    print("test:", test_df.shape)
    print("X:", X.shape)
    print("y:", y.shape)

        

if __name__ == "__main__":
    main()