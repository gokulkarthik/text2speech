import io
import json
import os
import zipfile
from pathlib import Path
from shutil import copyfile, rmtree
from typing import Dict, Tuple

import requests

from TTS.config import load_config
from TTS.utils.generic_utils import get_user_data_dir

LICENSE_URLS = {
    "cc by-nc-nd 4.0": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
    "mpl": "https://www.mozilla.org/en-US/MPL/2.0/",
    "mpl2": "https://www.mozilla.org/en-US/MPL/2.0/",
    "mpl 2.0": "https://www.mozilla.org/en-US/MPL/2.0/",
    "mit": "https://choosealicense.com/licenses/mit/",
    "apache 2.0": "https://choosealicense.com/licenses/apache-2.0/",
    "apache2": "https://choosealicense.com/licenses/apache-2.0/",
    "cc-by-sa 4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
}


class ModelManager(object):
    """Manage TTS models defined in .models.json.
    It provides an interface to list and download
    models defines in '.model.json'

    Models are downloaded under '.TTS' folder in the user's
    home path.

    Args:
        models_file (str): path to .model.json
    """

    def __init__(self, models_file=None, output_prefix=None):
        super().__init__()
        if output_prefix is None:
            self.output_prefix = get_user_data_dir("tts")
        else:
            self.output_prefix = os.path.join(output_prefix, "tts")
        self.models_dict = None
        if models_file is not None:
            self.read_models_file(models_file)
        else:
            # try the default location
            path = Path(__file__).parent / "../.models.json"
            self.read_models_file(path)

    def read_models_file(self, file_path):
        """Read .models.json as a dict

        Args:
            file_path (str): path to .models.json.
        """
        with open(file_path, "r", encoding="utf-8") as json_file:
            self.models_dict = json.load(json_file)

    def _list_models(self, model_type, model_count=0):
        model_list = []
        for lang in self.models_dict[model_type]:
            for dataset in self.models_dict[model_type][lang]:
                for model in self.models_dict[model_type][lang][dataset]:
                    model_full_name = f"{model_type}--{lang}--{dataset}--{model}"
                    output_path = os.path.join(self.output_prefix, model_full_name)
                    if os.path.exists(output_path):
                        print(f" {model_count}: {model_type}/{lang}/{dataset}/{model} [already downloaded]")
                    else:
                        print(f" {model_count}: {model_type}/{lang}/{dataset}/{model}")
                    model_list.append(f"{model_type}/{lang}/{dataset}/{model}")
                    model_count += 1
        return model_list

    def _list_for_model_type(self, model_type):
        print(" Name format: language/dataset/model")
        models_name_list = []
        model_count = 1
        model_type = "tts_models"
        models_name_list.extend(self._list_models(model_type, model_count))
        return [name.replace(model_type + "/", "") for name in models_name_list]

    def list_models(self):
        print(" Name format: type/language/dataset/model")
        models_name_list = []
        model_count = 1
        for model_type in self.models_dict:
            model_list = self._list_models(model_type, model_count)
            models_name_list.extend(model_list)
        return models_name_list

    def model_info_by_idx(self, model_query):
        """Print the description of the model from .models.json file using model_idx

        Args:
            model_query (str): <model_tye>/<model_idx>
        """
        model_name_list = []
        model_type, model_query_idx = model_query.split("/")
        try:
            model_query_idx = int(model_query_idx)
            if model_query_idx <= 0:
                print("> model_query_idx should be a positive integer!")
                return
        except:
            print("> model_query_idx should be an integer!")
            return
        model_count = 0
        if model_type in self.models_dict:
            for lang in self.models_dict[model_type]:
                for dataset in self.models_dict[model_type][lang]:
                    for model in self.models_dict[model_type][lang][dataset]:
                        model_name_list.append(f"{model_type}/{lang}/{dataset}/{model}")
                        model_count += 1
        else:
            print(f"> model_type {model_type} does not exist in the list.")
            return
        if model_query_idx > model_count:
            print(f"model query idx exceeds the number of available models [{model_count}] ")
        else:
            model_type, lang, dataset, model = model_name_list[model_query_idx - 1].split("/")
            print(f"> model type : {model_type}")
            print(f"> language supported : {lang}")
            print(f"> dataset used : {dataset}")
            print(f"> model name : {model}")
            if "description" in self.models_dict[model_type][lang][dataset][model]:
                print(f"> description : {self.models_dict[model_type][lang][dataset][model]['description']}")
            else:
                print("> description : coming soon")
            if "default_vocoder" in self.models_dict[model_type][lang][dataset][model]:
                print(f"> default_vocoder : {self.models_dict[model_type][lang][dataset][model]['default_vocoder']}")

    def model_info_by_full_name(self, model_query_name):
        """Print the description of the model from .models.json file using model_full_name

        Args:
            model_query_name (str): Format is <model_type>/<language>/<dataset>/<model_name>
        """
        model_type, lang, dataset, model = model_query_name.split("/")
        if model_type in self.models_dict:
            if lang in self.models_dict[model_type]:
                if dataset in self.models_dict[model_type][lang]:
                    if model in self.models_dict[model_type][lang][dataset]:
                        print(f"> model type : {model_type}")
                        print(f"> language supported : {lang}")
                        print(f"> dataset used : {dataset}")
                        print(f"> model name : {model}")
                        if "description" in self.models_dict[model_type][lang][dataset][model]:
                            print(
                                f"> description : {self.models_dict[model_type][lang][dataset][model]['description']}"
                            )
                        else:
                            print("> description : coming soon")
                        if "default_vocoder" in self.models_dict[model_type][lang][dataset][model]:
                            print(
                                f"> default_vocoder : {self.models_dict[model_type][lang][dataset][model]['default_vocoder']}"
                            )
                    else:
                        print(f"> model {model} does not exist for {model_type}/{lang}/{dataset}.")
                else:
                    print(f"> dataset {dataset} does not exist for {model_type}/{lang}.")
            else:
                print(f"> lang {lang} does not exist for {model_type}.")
        else:
            print(f"> model_type {model_type} does not exist in the list.")

    def list_tts_models(self):
        """Print all `TTS` models and return a list of model names

        Format is `language/dataset/model`
        """
        return self._list_for_model_type("tts_models")

    def list_vocoder_models(self):
        """Print all the `vocoder` models and return a list of model names

        Format is `language/dataset/model`
        """
        return self._list_for_model_type("vocoder_models")

    def list_langs(self):
        """Print all the available languages"""
        print(" Name format: type/language")
        for model_type in self.models_dict:
            for lang in self.models_dict[model_type]:
                print(f" >: {model_type}/{lang} ")

    def list_datasets(self):
        """Print all the datasets"""
        print(" Name format: type/language/dataset")
        for model_type in self.models_dict:
            for lang in self.models_dict[model_type]:
                for dataset in self.models_dict[model_type][lang]:
                    print(f" >: {model_type}/{lang}/{dataset}")

    @staticmethod
    def print_model_license(model_item: Dict):
        """Print the license of a model

        Args:
            model_item (dict): model item in the models.json
        """
        if "license" in model_item and model_item["license"].strip() != "":
            print(f" > Model's license - {model_item['license']}")
            if model_item["license"].lower() in LICENSE_URLS:
                print(f" > Check {LICENSE_URLS[model_item['license'].lower()]} for more info.")
            else:
                print(" > Check https://opensource.org/licenses for more info.")
        else:
            print(" > Model's license - No license information available")

    def download_model(self, model_name):
        """Download model files given the full model name.
        Model name is in the format
            'type/language/dataset/model'
            e.g. 'tts_model/en/ljspeech/tacotron'

        Every model must have the following files:
            - *.pth : pytorch model checkpoint file.
            - config.json : model config file.
            - scale_stats.npy (if exist): scale values for preprocessing.

        Args:
            model_name (str): model name as explained above.
        """
        # fetch model info from the dict
        model_type, lang, dataset, model = model_name.split("/")
        model_full_name = f"{model_type}--{lang}--{dataset}--{model}"
        model_item = self.models_dict[model_type][lang][dataset][model]
        # set the model specific output path
        output_path = os.path.join(self.output_prefix, model_full_name)
        if os.path.exists(output_path):
            print(f" > {model_name} is already downloaded.")
        else:
            os.makedirs(output_path, exist_ok=True)
            print(f" > Downloading model to {output_path}")
            # download from github release
            self._download_zip_file(model_item["github_rls_url"], output_path)
            self.print_model_license(model_item=model_item)
        # find downloaded files
        output_model_path, output_config_path = self._find_files(output_path)
        # update paths in the config.json
        self._update_paths(output_path, output_config_path)
        return output_model_path, output_config_path, model_item

    @staticmethod
    def _find_files(output_path: str) -> Tuple[str, str]:
        """Find the model and config files in the output path

        Args:
            output_path (str): path to the model files

        Returns:
            Tuple[str, str]: path to the model file and config file
        """
        model_file = None
        config_file = None
        for file_name in os.listdir(output_path):
            if file_name in ["model_file.pth", "model_file.pth.tar", "model.pth"]:
                model_file = os.path.join(output_path, file_name)
            elif file_name == "config.json":
                config_file = os.path.join(output_path, file_name)
        if model_file is None:
            raise ValueError(" [!] Model file not found in the output path")
        if config_file is None:
            raise ValueError(" [!] Config file not found in the output path")
        return model_file, config_file

    @staticmethod
    def _find_speaker_encoder(output_path: str) -> str:
        """Find the speaker encoder file in the output path

        Args:
            output_path (str): path to the model files

        Returns:
            str: path to the speaker encoder file
        """
        speaker_encoder_file = None
        for file_name in os.listdir(output_path):
            if file_name in ["model_se.pth", "model_se.pth.tar"]:
                speaker_encoder_file = os.path.join(output_path, file_name)
        return speaker_encoder_file

    def _update_paths(self, output_path: str, config_path: str) -> None:
        """Update paths for certain files in config.json after download.

        Args:
            output_path (str): local path the model is downloaded to.
            config_path (str): local config.json path.
        """
        output_stats_path = os.path.join(output_path, "scale_stats.npy")
        output_d_vector_file_path = os.path.join(output_path, "speakers.json")
        output_speaker_ids_file_path = os.path.join(output_path, "speaker_ids.json")
        speaker_encoder_config_path = os.path.join(output_path, "config_se.json")
        speaker_encoder_model_path = self._find_speaker_encoder(output_path)

        # update the scale_path.npy file path in the model config.json
        self._update_path("audio.stats_path", output_stats_path, config_path)

        # update the speakers.json file path in the model config.json to the current path
        self._update_path("d_vector_file", output_d_vector_file_path, config_path)
        self._update_path("model_args.d_vector_file", output_d_vector_file_path, config_path)

        # update the speaker_ids.json file path in the model config.json to the current path
        self._update_path("speakers_file", output_speaker_ids_file_path, config_path)
        self._update_path("model_args.speakers_file", output_speaker_ids_file_path, config_path)

        # update the speaker_encoder file path in the model config.json to the current path
        self._update_path("speaker_encoder_model_path", speaker_encoder_model_path, config_path)
        self._update_path("model_args.speaker_encoder_model_path", speaker_encoder_model_path, config_path)
        self._update_path("speaker_encoder_config_path", speaker_encoder_config_path, config_path)
        self._update_path("model_args.speaker_encoder_config_path", speaker_encoder_config_path, config_path)

    @staticmethod
    def _update_path(field_name, new_path, config_path):
        """Update the path in the model config.json for the current environment after download"""
        if new_path and os.path.exists(new_path):
            config = load_config(config_path)
            field_names = field_name.split(".")
            if len(field_names) > 1:
                # field name points to a sub-level field
                sub_conf = config
                for fd in field_names[:-1]:
                    if fd in sub_conf:
                        sub_conf = sub_conf[fd]
                    else:
                        return
                sub_conf[field_names[-1]] = new_path
            else:
                # field name points to a top-level field
                config[field_name] = new_path
            config.save_json(config_path)

    @staticmethod
    def _download_zip_file(file_url, output_folder):
        """Download the github releases"""
        # download the file
        r = requests.get(file_url)
        # extract the file
        try:
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                z.extractall(output_folder)
        except zipfile.BadZipFile:
            print(f" > Error: Bad zip file - {file_url}")
            raise zipfile.BadZipFile  # pylint: disable=raise-missing-from
        # move the files to the outer path
        for file_path in z.namelist()[1:]:
            src_path = os.path.join(output_folder, file_path)
            dst_path = os.path.join(output_folder, os.path.basename(file_path))
            copyfile(src_path, dst_path)
        # remove the extracted folder
        rmtree(os.path.join(output_folder, z.namelist()[0]))

    @staticmethod
    def _check_dict_key(my_dict, key):
        if key in my_dict.keys() and my_dict[key] is not None:
            if not isinstance(key, str):
                return True
            if isinstance(key, str) and len(my_dict[key]) > 0:
                return True
        return False
