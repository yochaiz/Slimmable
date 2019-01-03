from datetime import datetime
from numpy import random as nprandom
from sys import exit
from traceback import format_exc

import torch.backends.cudnn as cudnn
from torch.cuda import is_available, set_device
from torch.cuda import manual_seed as cuda_manual_seed
from torch import manual_seed as torch_manual_seed

from trainRegimes.SearchRegime import SearchRegime
from utils.HtmlLogger import HtmlLogger
from utils.emails import sendEmail
from utils.args import parseArgs

if __name__ == '__main__':
    # load command line arguments
    args = parseArgs()
    # init main logger
    logger = HtmlLogger(args.save, 'log')

    if not is_available():
        print('no gpu device available')
        exit(1)

    args.seed = datetime.now().microsecond
    nprandom.seed(args.seed)
    set_device(args.gpu[0])
    cudnn.benchmark = True
    torch_manual_seed(args.seed)
    cudnn.enabled = True
    cuda_manual_seed(args.seed)

    # try:
    #     set_start_method('spawn', force=True)
    # except RuntimeError:
    #     raise ValueError('spawn failed')

    try:
        # build regime for alphas optimization
        alphasRegime = SearchRegime(args, logger)
        # # train according to chosen regime
        # alphasRegime.train()
        logger.addInfoToDataTable('Done !')

    except Exception as e:
        # create message content
        messageContent = '[{}] stopped due to error [{}] \n traceback:[{}]'. \
            format(args.folderName, str(e), format_exc())

        # create data table if exception happened before we create data table
        if logger.dataTableCols is None:
            logger.createDataTable('Exception', ['Error message'])
        # log to logger
        logger.addInfoToDataTable(messageContent, color='lightsalmon')
        # send e-mail with error details
        subject = '[{}] stopped'.format(args.folderName)
        sendEmail(['yochaiz.cs@gmail.com'], subject, messageContent)

        # forward exception
        raise e
