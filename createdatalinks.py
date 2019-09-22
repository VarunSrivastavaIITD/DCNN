from main import config, construct_data


def main(model_dir=None):
    netconfig, hyperparams_config, data_config = config()
    if model_dir is None:
        model_num = int(netconfig['model_dir'][-1]) - 1
        model_dir = netconfig['model_dir'][:-1] + str(model_num)
    netconfig['model_dir'] = model_dir
    train_data, validate_data, test_data = construct_data(netconfig)


if __name__ == "__main__":
    model_dir = None
    main(model_dir)
