import utils as ut
import pickle as pkl
import os


class SyntheticData:

    def __init__(self, notes):
        self.bn = None
        self.bp = None
        self.data = notes

    def add_bn(self, bn_file):
        self.bn = pkl.load(open(os.path.join(ut.data_folder, bn_file), 'rb'))

    def add_wn(self):
        pass

    def add_bp(self, bp_file):
        self.bp = pkl.load(open(os.path.join(ut.data_folder, bp_file), 'rb'))

    def __remove_redundancy(self, redundancy_type):
        if redundancy_type == 'wn':
            self.nred_wn = self.__drop_duplicates(self.data)
        elif redundancy_type == 'bn':
            self.nred_bn = self.__rred_long(self.data)
        elif redundancy_type == 'bp':
            pass
        else:
            raise NotImplemented('Only within-note, between-note, and between-patient '
                                 'redundancies are allowed. Please one of '
                                 '"wn", "bn", and "bp".')

    @staticmethod
    def __drop_duplicates(notes):
        """

        :param notes: list of lists with sentences.
        :return: list of lists with unique sentences.
        """
        return [(el[0], list(dict.fromkeys(el[1]))) for el in notes]
