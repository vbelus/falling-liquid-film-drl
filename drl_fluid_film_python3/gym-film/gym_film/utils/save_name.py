class SaveName():
    def __init__(self, dic=None, name=None):
        # input dictionary should be like this -
        # {"number of epochs":, "model name":, "environment id":}
        if dic is not None:
            self.save_name_dic = dic

        if name is not None:
            self.save_name = name

    @property
    def save_name_dic(self):
        return self._save_name_dic

    @save_name_dic.setter
    def save_name_dic(self, save_name_dic):
        self._save_name_dic = save_name_dic
        self._update_save_name()

    def _update_save_name_dic(self):
        arguments = self.save_name.split("_")
        self._save_name_dic = {"model name": arguments[1],
                               "environment id": arguments[2],
                               "number of epochs": int(arguments[0][:-6])}

    @property
    def save_name(self):
        return self._save_name

    @save_name.setter
    def save_name(self, save_name):
        self._save_name = save_name
        self._update_save_name_dic()

    def _update_save_name(self):
        self._save_name = "{}epochs_{}_{}".format(self.save_name_dic["number of epochs"],
                                                  self.save_name_dic["model name"],
                                                  self.save_name_dic["environment id"])

    def add_epoch(self, nb_epoch):
        self._save_name_dic["number of epochs"] = self._save_name_dic.get(
            "number of epochs", 0) + nb_epoch
        self._update_save_name()


def get_training_name(model_dir):
    # extract training name from model directory
    if 'models/' in model_dir:
        return model_dir.split('models/')[-1].split('/')[0]
    else:
        return ''
