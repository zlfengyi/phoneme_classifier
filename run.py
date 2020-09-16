import fire
import utils
from torch import optim
from data.prefetch_dataset import VoiceDataset
from torch.utils.data import DataLoader
from settings.hparam import hparam as hp


class Runner:

    IMPLEMENTED_MODELS = ['cbhg', 'mgru']

    def train(self, model, checkpoint='', is_cuda=True, is_multi_gpu=True, logdir='', savedir=''):
        if model not in self.IMPLEMENTED_MODELS:
            raise NotImplementedError('%s model is not implemented !' % model)

        mode = 'train'
        logger = utils.get_logger(mode)

        # initialize hyperparameters
        hp.set_hparam_yaml(mode)
        logger.info('Setup mode as %s, model : %s' % (mode, model))

        # get network
        network = utils.get_networks(model, checkpoint, is_cuda, is_multi_gpu)

        # Setup dataset
        train_dataloader = DataLoader(VoiceDataset(mode='train'), batch_size=hp.train.batch_size,
                                      shuffle=(mode == 'train'), num_workers=hp.num_workers, drop_last=False)
        test_dataloader = DataLoader(VoiceDataset(mode='test'), batch_size=hp.test.batch_size,
                                     shuffle=(mode == 'test'), num_workers=hp.num_workers, drop_last=False)

        # setup optimizer:
        parameters = network.parameters()
        logger.info(network)
        # TODO: Scheduled LR
        lr = getattr(hp, mode).lr
        optimizer = optim.Adam([p for p in parameters if p.requires_grad], lr=lr)


        # pass model, loss, optimizer and dataset to the trainer
        # get trainer
        trainer = utils.get_trainer()(network, optimizer, train_dataloader, test_dataloader, is_cuda, logdir, savedir)

        # train!
        trainer.run(hp.train.num_epochs)

    def eval(self):
        raise NotImplementedError('Evaluation mode is not implemented!')


if __name__ == '__main__':
    fire.Fire(Runner)
