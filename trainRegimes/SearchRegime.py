from .regime import TrainRegime, time


class SearchRegime(TrainRegime):
    def __init__(self, args, logger):
        super(SearchRegime, self).__init__(args, logger)

        # init email time
        self.lastMailTime = time()
        self.secondsBetweenMails = 1 * 3600
