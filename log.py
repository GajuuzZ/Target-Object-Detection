import os
import logging


class TrainingLogger:
    def __init__(self, save_path):
        self.save_path = save_path

        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler = logging.FileHandler(self.save_path)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        self.log.addHandler(handler)

        self._pause = 'pause!'
        self.last = ''
        self.last_idx = 0
        self.last_epcoh = 0
        self.last_batch = 0
        self.last_mode = None

    def write(self, mode, epoch, batch, idx, metric):
        self.last = '{}/{}/{}/{} | '.format(mode, epoch, batch, idx)
        for k, v in metric.items():
            self.last += '{}: {:.6f}, '.format(k, v)
        self.log.info(self.last)

    def reset_checkpoint(self, last_check):
        """ reset log to checkpoint interval. """
        path, ext = os.path.splitext(self.save_path)
        new_file = path + '_' + ext
        with open(self.save_path, 'r') as of:
            with open(new_file, 'w') as nf:
                for line in of:
                    if line.strip() == self._pause:
                        continue
                    nf.write(line)
                    if last_check.strip() == line.split(' - ')[1].strip():
                        nf.write(self._pause + '\n')
                        break

        os.replace(new_file, self.save_path)
        # reopen log file.
        self.log.handlers[0].close()
        self.log.handlers[0]._open()

        split = last_check.split('|')[0].strip().split('/')
        self.last_mode = split[0]
        self.last_epcoh = int(split[1])
        self.last_batch = int(split[2])
        self.last_idx = int(split[3])

    def reset_last(self):
        self.last_idx = 0
        self.last_epcoh = 0
        self.last_batch = 0
        self.last_mode = None

    def get_history(self):
        hist = {}
        with open(self.save_path, 'r') as f:
            for line in f:
                if line.strip() == self._pause:
                    continue
                line = line.strip().split(' - ')[1]
                at, metric = line.strip().split(' | ')
                mode, e = at.split('/')[:2]

                if e not in hist.keys():
                    hist[e] = {}
                if mode not in hist[e].keys():
                    hist[e][mode] = {}
                for mt in metric.split(','):
                    if len(mt) == 0: continue
                    n, v = mt.strip().split(': ')
                    if n not in hist[e][mode].keys():
                        hist[e][mode][n] = []
                    hist[e][mode][n].append(float(v))

        return hist
