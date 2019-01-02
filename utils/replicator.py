from multiprocessing import Pool

from torch import no_grad, randn
from torch.cuda import set_device, current_device


class ModelReplicator:
    title = 'Replications'

    def __init__(self, modelConstructor, gpusList, logger):
        self.gpuIDs = gpusList
        # init replications list
        self.replications = []
        # init info table rows
        rows = []

        # save current device
        currDevice = current_device()
        # create replications
        for gpu in self.gpuIDs:
            # set device to required gpu
            set_device(gpu)
            # create model new instance
            cModel = modelConstructor()
            # set model to cuda on specific GPU
            cModel = cModel.cuda()
            # set mode to eval mode
            cModel.eval()
            # add model to replications
            self.replications.append((cModel, gpu))

        # reset device back to current device
        set_device(currDevice)

        rows.insert(0, ['#', len(self.replications)])
        # create info table
        logger.addInfoTable(self.title, rows)

# def demo(self, args):
#     (cModel, gpu), nSamples = args
#     # switch to process GPU
#     set_device(gpu)
#     assert (cModel.training is False)
#
#     data = randn(250, 3, 32, 32).cuda()
#
#     print('gpu [{}] start'.format(gpu))
#
#     with no_grad():
#         for _ in range(nSamples):
#             cModel(data)
#             data[0, 0, 0, 0] += 0.001
#
#     print('gpu [{}] end'.format(gpu))
#
# def replicationFunc(self, args):
#     self.demo(args)
#
# def run(self):
#     nCopies = len(self.replications)
#
#     nSamples = int(5000 / nCopies)
#     print('nSamples:[{}]'.format(nSamples))
#     args = ((r, nSamples) for r in self.replications)
#
#     with Pool(processes=nCopies, maxtasksperchild=1) as pool:
#         results = pool.map(self.replicationFunc, args)
#
#     return results
