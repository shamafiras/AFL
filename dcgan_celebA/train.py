import config
from models.bank_model import BankModel

opt = config.get_configurations()

if __name__ == "__main__":
    model = BankModel(opt)
    model.init_paths()
    model.write_config()
    if opt.training_scheme == 'all':
        model.train(main_training=True)
        print('Trained main network')
        model.net.loop_count = 2
        model.train(main_training=False)
        print('Trained bank network')
    elif opt.training_scheme == 'bank':
        model.net.loop_count = 2
        model.load_pre_trained()
        model.train(main_training=False)
        print('Trained bank network')
    elif opt.training_scheme == 'main':
        model.train(main_training=True)
        print('Trained main network')
