import yaml

class Cfg(dict):
    def __init__(self, config_dict):
        super(Cfg, self).__init__(**config_dict)
        self.__dict__ = self

    @staticmethod
    def load_config_from_file(fname):
        # base_config = download_config(url_config['base'])
        base_config = {}
        # with open("/home/data2/thaitran/Research/OCR/source/vietocr-finetune/vietocr/config/base.yml", encoding='utf-8') as b:
        #     base_config = yaml.safe_load(b)
        with open(fname, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        base_config.update(config)

        return Cfg(base_config)


