import json
import os
import random as rm
import time
from itertools import combinations
from math import log10

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import requests
from plotly.subplots import make_subplots

pio.templates.default = "plotly_white"

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# raise exceptions for invalid user input


class UnknownArgument(ValueError):
    pass


class UnknownAttribute(ValueError):
    pass


class APIKey(ValueError):
    pass


class InvalidAPIKey(ValueError):
    pass


class APIError(ValueError):
    pass


class Hp:
    def __init__(self):
        self.hps_dict_num = {}  # this dict will be updated at each round
        self.num_int = []
        self.num_float = []
        self.num_log = []

        self.hps_dict_num['hyper_para_99999'] = [
            1.101, 100.101]  # synthetic hp if user supply 1 hp
        self.num_float.append('hyper_para_99999')

    def numrange(self, name, min, max, type='int'):
        if type.lower().strip() == 'int':
            self.hps_dict_num[name.strip()] = [int(min), int(max)]
            self.num_int.append(name.strip())
            # return int(min)
        elif type.lower().strip() == 'float':
            self.hps_dict_num[name.strip()] = [float(min), float(max)]
            self.num_float.append(name.strip())
            # return float(min)
        elif type.lower().strip() == 'log':
            self.hps_dict_num[name.strip()] = [log10(min), log10(max)]
            self.num_log.append(name.strip())
            # return min
        else:
            raise UnknownArgument(
                f'unknown type found "{type}"... type should be either "int", "float" or "log"')

    def remove_hp(self, name):
        if name.strip() in self.hps_dict_num.keys():
            del self.hps_dict_num[name.strip()]

            if name.strip() in self.num_int:
                self.num_int.remove(name.strip())
            elif name.strip() in self.num_float:
                self.num_float.remove(name.strip())
            elif name.strip() in self.num_log:
                self.num_log.remove(name.strip())
        else:
            raise UnknownArgument(f'"{name}" hyperparameter is not defined')

    @property
    def search_space_boundaries(self):
        hps_df = pd.DataFrame(self.hps_dict_num)
        for name in self.num_log:
            hps_df[name] = 10**hps_df[name]
        hps_df.index = ['min', 'max']
        return hps_df

    def lhc_call(self, input_dict, api_key):
        try:
            url = "https://aisara-hyperparameter-tuning.p.rapidapi.com/latin/uniform"
            headers = {
                'content-type': "application/json",
                'x-rapidapi-key': api_key,
                'x-rapidapi-host': "aisara-hyperparameter-tuning.p.rapidapi.com"
            }
            response = requests.request(
                "POST", url, data=json.dumps(input_dict), headers=headers)
            return response.json()
        except Exception:
            raise APIError('Failure in API endpoint, please rerun again')

    def hps_combinations(self, num_trials, api_key):
        dict_input = {}
        dict_input['subsection'] = num_trials
        for key, value in self.hps_dict_num.items():
            dict_input[key] = value
        hps_comb = pd.DataFrame(self.lhc_call(dict_input, api_key))
        for name in self.num_int:
            hps_comb[name] = hps_comb[name].round().astype('int')
        for name in self.num_log:
            hps_comb[name] = 10 ** hps_comb[name]
        return hps_comb

    def uniform_sampling_call(self, num_trials):
        dict_output = {}
        for key, value in self.hps_dict_num.items():
            if key in self.num_int:
                dict_output[key] = [int(round(i)) for i in np.linspace(
                    value[0], value[1], num=num_trials, endpoint=True)]
            if key in self.num_float:
                dict_output[key] = np.linspace(
                    value[0], value[1], num=num_trials, endpoint=True)
            if key in self.num_log:
                dict_output[key] = [
                    10 ** i for i in np.linspace(value[0], value[1], num=num_trials, endpoint=True)]

        output_df = pd.DataFrame(dict_output)
        for col in output_df.columns:
            if col in self.num_int:
                output_df[col] = output_df[col].astype('int')
        return output_df


class HpOptimization:
    def __init__(self, hp_class, model_func, opti_paras, opti_objects, num_trials=5, rounds=3, mode='c', api_key='0000', aisara_seed='variable'):
        self.hp_class = hp_class
        self.model_func = model_func

        if (len(opti_paras) == 1) and (len(opti_objects) == 1):
            self.opti_para_main = opti_paras[0].strip()
            self.opti_para_sup = 'opti_para_sup'
            self.opti_object_main = opti_objects[0].strip()
            self.opti_object_sup = 'max'
        elif (len(opti_paras) == 2) and (len(opti_objects) == 2):
            self.opti_para_main = opti_paras[0].strip()
            self.opti_para_sup = opti_paras[1].strip()
            self.opti_object_main = opti_objects[0].strip()
            self.opti_object_sup = opti_objects[1].strip()
        else:
            raise UnknownAttribute(
                '''length of optimization parameters should be equal to the length of optimization objectives''')

        self.num_trials = num_trials
        self.max_rounds = rounds
        self.run_mode = mode.lower().strip()
        self.api_key = api_key
        self.aisara_seed = aisara_seed.lower().strip()

    def opti_objective_check(self):
        if self.opti_object_main == 'min' or self.opti_object_main == 'max':
            pass
        else:
            raise UnknownAttribute(
                f'''unknown optimization objective... optimization objective should be either "min" or "max"''')
        if self.opti_object_sup == 'min' or self.opti_object_sup == 'max':
            pass
        else:
            raise UnknownAttribute(
                f'''unknown optimization objective... optimization objective should be either "min" or "max"''')

    def api_subscription(self):
        url = "https://aisara-hyperparameter-tuning.p.rapidapi.com/grid/general_pred"
        payload = '{"epoch":[10, 15, 30],"loss":"mae","lr":[0.0001, 0.001, 0.01],"metric":"max","nodes_hidden_1":[32, 128, 256],' \
                  '"nodes_hidden_2":[32, 128, 256 ],"x_shape":[4, 5],"x_train":"60, 116, 0.000178, 19.0, 0.6065573692321777, 116, 60, ' \
                  '0.000562, 16.0, 0.49180328845977783, 228, 172, 0.001778, 11.0, 0.7213114500045776, 172, 228, 0.005623, 14.0, 0.7704917788505554"}'
        headers = {'x-rapidapi-host': "aisara-hyperparameter-tuning.p.rapidapi.com",
                   'x-rapidapi-key': self.api_key,
                   'content-type': "application/json",
                   'accept': "application/json"}
        response = requests.request("POST", url, data=payload, headers=headers)
        for key, value in response.json().items():
            if key == 'message' and value == 'You are not subscribed to this API.':
                raise InvalidAPIKey(
                    'Incorrect API Key!!... please provide valid API key')
            else:
                pass

    def license_mode(self):
        print('\033[1m' + "For commercial use, you can obtain our API from https://rapidapi.com/aisara-technology-aisara-technology-default/api/aisara-hyperparameter-tuning\n"
              'If you are a private user, set the mode parameter in HpOptimization class to "p".\n' + '\033[0m')

        if self.run_mode == 'p':
            # free API key for mode 'p'
            self.api_key = "3b18dfe0e2mshaba41fff27bc31dp1fae09jsnc46d20ec56d2"
            self.api_subscription()
        elif self.run_mode == 'c':
            if self.api_key == "0000":
                raise APIKey(
                    'No API Key is provided!!... please provide the API key')
            elif self.api_key == "3b18dfe0e2mshaba41fff27bc31dp1fae09jsnc46d20ec56d2":
                raise InvalidAPIKey(
                    'Incorrect API Key!!... please provide a valid API key')
            else:
                self.api_subscription()
        else:
            raise UnknownAttribute(
                f'unknown mode "{self.run_mode}"... mode should be either "c" or "p"')

    def set_aisara_seed(self):
        if self.aisara_seed == 'fixed':
            os.environ['PYTHONHASHSEED'] = '0'
            np.random.seed(37)
            rm.seed(1254)
        elif self.aisara_seed == 'variable':
            pass
        else:
            raise UnknownAttribute(
                f'unknown aisara_seed "{self.aisara_seed}"... aisara_seed should be either "fixed" or "variable"')

    def aisara_core_fit_call(self, x, y):
        x_train = []
        for col in x.columns:
            for v in x[col].tolist():
                x_train.append(v)
        x_train_str = str(x_train).replace('[', '').replace(']', '')
        y_train = y.tolist()
        y_train_str = str(y_train).replace('[', '').replace(']', '')
        try:
            url = "https://aisara-hyperparameter-tuning.p.rapidapi.com/developer/dfit"
            payload = {"params": {"ranking_method": "AiSaraRank3", "train_split": "100",
                                  "train_x": x_train_str, "train_y": y_train_str,
                                  "x_shape": f"{[i for i in x.shape]}", "y_shape": f"{[len(y_train),1]}"}}
            headers = {
                'content-type': "application/json",
                'x-rapidapi-key': self.api_key,
                'x-rapidapi-host': "aisara-hyperparameter-tuning.p.rapidapi.com"
            }
            response = requests.request(
                "POST", url, data=json.dumps(payload), headers=headers)
            return response.json()['result']
        except Exception:
            raise APIError('Failure in API endpoint, please rerun again')

    def aisara_core_predict_call(self, model_id, x):
        x_shape = [i for i in x.shape]
        x_train = []
        for index, row in x.iterrows():
            for col in x.columns:
                x_train.append(row[col])
        x_train_str = str(x_train).replace('[', '').replace(']', '')
        try:
            url = "https://aisara-hyperparameter-tuning.p.rapidapi.com/developer/dpred"
            payload = {"params": {"model_id": model_id, "test_x": x_train_str,
                                  "x_shape": f"{x_shape}", "y_selected": "1"}}
            headers = {
                'content-type': "application/json",
                'x-rapidapi-key': self.api_key,
                'x-rapidapi-host': "aisara-hyperparameter-tuning.p.rapidapi.com"
            }
            response = requests.request(
                "POST", url, data=json.dumps(payload), headers=headers)
            prediction = response.json()['result']
            impact_mod = response.json()['impacts']
            impact_discimal = [sum(i) for i in impact_mod]
            impact = [round(i * 100, 3) for i in impact_discimal]
            impact.reverse()
            return prediction, impact
        except Exception:
            raise APIError('Failure in API endpoint, please rerun again')

    def df_to_sting_x(self, results_log_df, target):
        df = results_log_df.copy()
        if target == 'error':
            del df[self.opti_para_sup]
            del df[self.opti_para_main]
            del df[f'{self.opti_para_main}_predict']
        else:
            del df[self.opti_para_sup]
            del df[f'{self.opti_para_main}_predict']
            del df['error']
            del df['aisara_max_error']

        x_train = []
        for index, row in df.iterrows():
            for col in df.columns:
                x_train.append(row[col])
        x_train_str = str(x_train).replace('[', '').replace(']', '')
        return df, x_train_str

    def gen_pred_call(self, results_log_df, target, min_max):
        df_tostr, df_srt = self.df_to_sting_x(results_log_df, target)
        rows, colums = df_tostr.shape
        x_shape = [rows, colums]
        dict_api_input = {}
        for col in df_tostr.columns[:-1]:
            if col in self.hp_class.num_int:
                dict_api_input[col] = [int(i) for i in np.linspace(
                    self.hp_class.hps_dict_num[col][0], self.hp_class.hps_dict_num[col][1], 17)]
            else:
                dict_api_input[col] = [i for i in np.linspace(
                    self.hp_class.hps_dict_num[col][0], self.hp_class.hps_dict_num[col][1], 17)]

        dict_api_input["x_train"] = df_srt
        dict_api_input["x_shape"] = x_shape
        dict_api_input["metric"] = f"{min_max}"
        dict_api_input["loss"] = "mae"
        try:
            url = "https://aisara-hyperparameter-tuning.p.rapidapi.com/grid/general_pred"
            headers = {
                'content-type': "application/json",
                'x-rapidapi-key': self.api_key,
                'x-rapidapi-host': "aisara-hyperparameter-tuning.p.rapidapi.com"
            }
            api_results = requests.request(
                "POST", url, data=json.dumps(dict_api_input), headers=headers)

            impact = api_results.json()['Impact']
            best_comp = api_results.json()['Best combination']
            best_comp_df = pd.DataFrame(best_comp)
            best_comp_df.rename(
                columns={'Prediction': f'aisara_{min_max}_{target}'}, inplace=True)
            return best_comp_df, impact
        except Exception:
            raise APIError('Failure in API endpoint, please rerun again')

    def general_grid_call(self, input):
        try:
            url = "https://aisara-hyperparameter-tuning.p.rapidapi.com/grid/general"
            headers = {
                'content-type': "application/json",
                'x-rapidapi-key': self.api_key,
                'x-rapidapi-host': "aisara-hyperparameter-tuning.p.rapidapi.com"
            }
            response = requests.request(
                "POST", url, data=json.dumps(input), headers=headers)
            return response.json()
        except Exception:
            raise APIError('Failure in API endpoint, please rerun again')

    def run_opti(self):
        # optimization objective check
        self.opti_objective_check()

        # License mode
        self.license_mode()

        # fixed seed
        self.set_aisara_seed()

        # dictionary contains rounds results
        self.rounds_dict = {}
        # choosing between user's rounds or aisara_auto round
        if self.max_rounds == 'aisara_auto':
            self.max_rounds = 1000

        # check how many if hps >=1
        if len(self.hp_class.hps_dict_num) == 2:
            self.one_hp_check = True
        else:
            self.one_hp_check = False
            del self.hp_class.hps_dict_num['hyper_para_99999']

        # list contains best hps at each run
        rounds_best_hps = []
        self.search_space_dict = {}
        for round in range(self.max_rounds):
            print('\033[1m'+'\033[94m'+f'Round-{round+1}:'+'\033[0m')
            model_training_time = []
            if self.one_hp_check:
                self.comp = self.hp_class.uniform_sampling_call(
                    self.num_trials)  # uniform sampling if case of 1 hyperparameter is given
            else:
                self.comp = self.hp_class.hps_combinations(
                    self.num_trials, self.api_key)  # create lhc combinations
            self.opti_param_values = []  # list contains opti_apara main value at each run
            self.opti_paras_values = []  # list contains opti_apara sup value at each run

            for num in range(self.num_trials):
                print('\n'+'\033[1m'+'\033[92m'+f'  Run-{num+1}:'+'\033[0m')

                if self.one_hp_check:
                    comp_print = {self.comp.columns[1]: self.comp.iloc[num, 1]}
                else:
                    comp_print = self.comp.iloc[num, :].to_dict()
                for key, value in comp_print.items():  # to solve the int issue in training log
                    if key in self.hp_class.num_int:
                        comp_print[key] = int(value)

                t_start = time.time()
                function_output = self.model_func(self.comp, num)
                t_end = time.time()
                train_time = t_end-t_start
                model_training_time.append(train_time)
                if isinstance(function_output, tuple):
                    return_main, return_support = function_output
                    self.opti_param_values.append(return_main)
                    self.opti_paras_values.append(return_support)
                    print({self.opti_para_main: return_main,
                          self.opti_para_sup: return_support, 'hyperparameters': comp_print})
                else:
                    return_main = function_output
                    return_support = 99999
                    self.opti_param_values.append(return_main)
                    self.opti_paras_values.append(return_support)
                    print({self.opti_para_main: return_main,
                          'hyperparameters': comp_print})

            if self.one_hp_check:
                print('\033[1m'+'\n\nsearch space boundaries:\n'+'\033[0m',
                      self.hp_class.search_space_boundaries.iloc[:, 1].to_frame(), '\n')
                print('\033[1m'+'hyperparameters combinations (uniform sampling):\n'+'\033[0m',
                      self.comp.iloc[:, 1].to_frame().to_string(index=False), '\n')
            else:
                print('\033[1m'+'\n\nsearch space boundaries:\n' +
                      '\033[0m', self.hp_class.search_space_boundaries, '\n')
                print('\033[1m'+'hyperparameters combinations (lHC sampling):\n' +
                      '\033[0m', self.comp.to_string(index=False), '\n')
            to_search_space_dict = self.hp_class.search_space_boundaries
            to_search_space_dict.index = [0, 1]
            to_search_space_dict['Round'] = f'Round_{round+1}'
            self.search_space_dict[f'Round_{round + 1}'] = to_search_space_dict

            self.round_results = self.comp.copy()
            self.round_results[self.opti_para_sup] = self.opti_paras_values
            self.round_results[self.opti_para_main] = self.opti_param_values

            round_results_columns = self.round_results.columns.to_list()
            if self.opti_paras_values[0] == 99999:
                round_results_columns.remove(self.opti_para_sup)
            else:
                pass
            if self.one_hp_check:
                print('\033[1m'+f'models results:\n''\033[0m'+'\033[0m',
                      self.round_results[round_results_columns].iloc[:, 1:].to_string(index=False))  # to remove the synthetic hp
            else:
                print('\033[1m' + f'models results:\n''\033[0m' + '\033[0m',
                      self.round_results[round_results_columns].to_string(index=False))

            to_rounds_dict = self.round_results.copy()
            to_rounds_dict['Round'] = f'Round_{round + 1}'
            to_rounds_dict['training time (sec)'] = model_training_time
            self.rounds_dict[f'Round_{round + 1}'] = to_rounds_dict

            if round < self.max_rounds-1:
                # creation of new search space boundaries
                # convert log column to log values
                self.round_results_log = self.round_results.copy()
                for name in self.hp_class.num_log:
                    if name in self.round_results_log.columns:
                        self.round_results_log[name] = np.log10(
                            self.round_results_log[name])

                dict_poped_col = {}  # it will be added later to self.hps_dict_num
                # excluding opti_para_main and opti_para_sup
                self.round_results_log_hps_col = self.round_results_log.columns[:-2]
                stop_cols = [x for x in self.round_results_log_hps_col.copy()]
                for col_name in self.round_results_log_hps_col:
                    # checking if the col contains similar values
                    if len(self.round_results_log[col_name].unique()) == 1:
                        dict_poped_col[col_name] = [self.round_results_log[col_name].unique(
                        )[0], self.round_results_log[col_name].unique()[0]]
                        del self.round_results_log[col_name]
                        stop_cols.remove(col_name)

                # checking that aisara will receive min of 1 col with non unique values
                if len(stop_cols) >= 2:
                    print(
                        '\033[1m'+'\nworking on the creation of reduced search space boundaries for the next round, pls wait this might take some time'+'\033[0m')
                # hps of best round run:
                    if self.opti_object_main == 'max':
                        mask1 = self.round_results_log[self.opti_para_main] == self.round_results_log[self.opti_para_main].max(
                        )
                    else:
                        mask1 = self.round_results_log[self.opti_para_main] == self.round_results_log[self.opti_para_main].min(
                        )
                    trans = self.round_results_log[mask1]
                    if self.opti_object_sup == 'max':
                        mask2 = trans[self.opti_para_sup] == trans[self.opti_para_sup].max(
                        )
                    else:
                        mask2 = trans[self.opti_para_sup] == trans[self.opti_para_sup].min(
                        )

                    # saving round best results for comp
                    to_rounds_best_hps = trans[mask2].iloc[:1, :].copy()
                    to_rounds_best_hps.index = [0]
                    rounds_best_hps.append(to_rounds_best_hps)

                    if round > 0:
                        for key, value in dict_poped_col.items():
                            if key in rounds_best_hps[round-1].columns:
                                del rounds_best_hps[round-1][key]
                        self.round_results_log = pd.concat(
                            [self.round_results_log, rounds_best_hps[round-1]], ignore_index=True)

                        if self.opti_object_main == 'max':
                            mask1_1 = self.round_results_log[self.opti_para_main] == self.round_results_log[self.opti_para_main].max(
                            )
                        else:
                            mask1_1 = self.round_results_log[self.opti_para_main] == self.round_results_log[self.opti_para_main].min(
                            )
                        trans_1 = self.round_results_log[mask1_1]
                        if self.opti_object_sup == 'max':
                            mask2_1 = trans_1[self.opti_para_sup] == trans_1[self.opti_para_sup].max(
                            )
                        else:
                            mask2_1 = trans_1[self.opti_para_sup] == trans_1[self.opti_para_sup].min(
                            )

                        best_hps_round = trans_1[mask2_1].iloc[:1, :-2].copy()
                        best_hps_round.index = ['round_best_hps']
                        best_hps_round_t = best_hps_round.T

                    else:
                        best_hps_round = trans[mask2].iloc[:1, :-2].copy()
                        best_hps_round.index = ['round_best_hps']
                        best_hps_round_t = best_hps_round.T

                # 1) multi_error_calc:
                    all_index = list(self.round_results_log.index)
                    index_predic = []
                    for m in combinations(self.round_results_log.index, len(self.round_results_log.index) - 1):
                        comb_multi = list(m)
                        not_in = [l for l in all_index if l not in comb_multi]
                        ref = self.aisara_core_fit_call(
                            self.round_results_log.iloc[comb_multi, :-2], self.round_results_log.iloc[comb_multi, -1])
                        predict, _ = self.aisara_core_predict_call(
                            ref, self.round_results_log.iloc[not_in, :-2])
                        index_predic.append((not_in[0], predict[0]))

                    index_predic.sort(key=lambda x: x[0])
                    self.round_results_log[f'{self.opti_para_main}_predict'] = [
                        i[1] for i in index_predic]
                    self.round_results_log['error'] = abs(self.round_results_log[f'{self.opti_para_main}']
                                                          - self.round_results_log[f'{self.opti_para_main}_predict']) \
                        / self.round_results_log[f'{self.opti_para_main}']

                # 2) aisara_max error:
                    best_comp_e, _ = self.gen_pred_call(
                        self.round_results_log, 'error', 'max')
                    self.round_results_log = pd.concat(
                        [self.round_results_log, best_comp_e], ignore_index=True)
                    for key, value in dict_poped_col.items():
                        best_comp_e[key] = value[0]
                    del best_comp_e['aisara_max_error']
                    for name in self.hp_class.num_log:
                        best_comp_e[name] = 10**(best_comp_e[name])

                    print('\n'+'\033[1m'+'\033[95m' +
                          'Aisara_max_error Run:'+'\033[0m')
                    if self.one_hp_check:
                        comp_print_e = {
                            best_comp_e.columns[1]: best_comp_e.iloc[0, 1]}
                    else:
                        comp_print_e = best_comp_e.iloc[0, :].to_dict()
                    for key, value in comp_print_e.items():  # to solve the int issue in training log
                        if key in self.hp_class.num_int:
                            comp_print_e[key] = int(value)

                    function_output = self.model_func(best_comp_e, 0)
                    if isinstance(function_output, tuple):
                        return_main, return_support = function_output
                        if round == 0:
                            self.round_results_log.at[self.num_trials,
                                                      self.opti_para_sup] = return_support
                            self.round_results_log.at[self.num_trials,
                                                      self.opti_para_main] = return_main
                        else:
                            self.round_results_log.at[self.num_trials +
                                                      1, self.opti_para_sup] = return_support
                            self.round_results_log.at[self.num_trials +
                                                      1, self.opti_para_main] = return_main
                        print({self.opti_para_main: return_main,
                              self.opti_para_sup: return_support, 'hyperparameters': comp_print_e})
                    else:
                        return_main = function_output
                        return_support = 99999
                        if round == 0:
                            self.round_results_log.at[self.num_trials,
                                                      self.opti_para_sup] = return_support
                            self.round_results_log.at[self.num_trials,
                                                      self.opti_para_main] = return_main
                        else:
                            self.round_results_log.at[self.num_trials +
                                                      1, self.opti_para_sup] = return_support
                            self.round_results_log.at[self.num_trials +
                                                      1, self.opti_para_main] = return_main
                        print({self.opti_para_main: return_main,
                              'hyperparameters': comp_print_e})

                # 3) aisara_max acc/mae:
                    best_comp_a_m, impact_a_m = self.gen_pred_call(
                        self.round_results_log, self.opti_para_main, self.opti_object_main)
                    self.round_results_log = pd.concat(
                        [self.round_results_log, best_comp_a_m], ignore_index=True)

                # aisara best hps of best round run:
                    best_hps_aisara = best_comp_a_m.iloc[:, :-1]
                    best_hps_aisara.index = ['aisara_best_hps']
                    best_hps_aisara_t = best_hps_aisara.T

                # 4) round best and aisara best (error, modified_error, min and max):
                    error_perc_df = pd.concat(
                        [best_hps_round_t, best_hps_aisara_t], axis=1)
                    error_perc_df['error'] = abs(
                        (error_perc_df['round_best_hps'] - error_perc_df['aisara_best_hps'])) / error_perc_df['round_best_hps']
                    impact_df = pd.DataFrame(impact_a_m, index=[0])
                    impact_df.index = ['impact_factor']
                    impact_df_t = impact_df.T
                    error_impact_df = pd.concat(
                        [error_perc_df, impact_df_t], axis=1)
                    error_impact_df['modified_error'] = np.where(error_impact_df['impact_factor'] < 1, error_impact_df['error'] * (1 - error_impact_df['impact_factor']),
                                                                 error_impact_df['error'])
                    error_impact_df['min_*'] = error_impact_df['round_best_hps'] * \
                        (1 - error_impact_df['modified_error'])
                    error_impact_df['max_*'] = error_impact_df['round_best_hps'] * \
                        (1 + error_impact_df['modified_error'])

                # 5) new boundaries min, max:
                    list_min = []
                    for row in error_impact_df.index:
                        if row in self.hp_class.num_int:
                            list_min.append(np.round(
                                max(error_impact_df.loc[row, 'min_*'], self.hp_class.hps_dict_num[row][0])))
                        else:
                            if row == 'hyper_para_99999':
                                list_min.append(1.101)
                            else:
                                list_min.append(
                                    max(error_impact_df.loc[row, 'min_*'], self.hp_class.hps_dict_num[row][0]))
                    error_impact_df['min'] = list_min
                    list_max = []
                    for row in error_impact_df.index:
                        if row in self.hp_class.num_int:
                            list_max.append(np.round(
                                min(error_impact_df.loc[row, 'max_*'], self.hp_class.hps_dict_num[row][1])))
                        else:
                            if row == 'hyper_para_99999':
                                list_max.append(100.101)
                            else:
                                list_max.append(
                                    min(error_impact_df.loc[row, 'max_*'], self.hp_class.hps_dict_num[row][1]))
                    error_impact_df['max'] = list_max

                    dict_hps_new_ss_log = error_impact_df.iloc[:, -2:].T
                    for col in dict_hps_new_ss_log.columns:
                        if col in self.hp_class.num_int:
                            dict_hps_new_ss_log[col] = dict_hps_new_ss_log[col].astype(
                                'int')

                # update self.hp_class.hps_dict_num
                    for key, value in dict_hps_new_ss_log.to_dict().items():
                        self.hp_class.hps_dict_num[key] = [
                            value['min'], value['max']]
                    for key, value in dict_poped_col.items():
                        self.hp_class.hps_dict_num[key] = [
                            value[0].item(), value[0].item()]
                    print()
                else:
                    print(
                        '\033[1m' + '\033[91m' + '\nmaximum search space reduction has been achieved, not future optimization can be done.' + '\033[0m')
                    break

            else:
                print('\033[1m'+'\noptimization has ended'+'\033[0m')
                break

    def gen_all_rounds_results(self):
        all_rounds_results = pd.concat(
            [i for i in self.rounds_dict.values()], ignore_index=True)
        if self.opti_paras_values[0] == 99999:
            del all_rounds_results['opti_para_sup']
        if self.one_hp_check:
            return all_rounds_results.iloc[:, 1:]
        else:
            return all_rounds_results

    @property
    def opti_results(self):
        return self.gen_all_rounds_results().to_string(index=False)

    def plot_opti_results(self):
        para_plot_opti = self.gen_all_rounds_results()
        to_fig_opti_results = {}
        to_fig_opti_results['Round'] = para_plot_opti['Round'].unique()
        if self.opti_object_main == 'max':
            min_max_opti_para_main = [para_plot_opti.iloc[self.num_trials * (i):self.num_trials * (
                i + 1), -3].max() for i in range(int(to_fig_opti_results['Round'][-1].split('_')[-1]))]
            min_max_opti_para_ploting = [
                max(min_max_opti_para_main[:i + 1]) for i in range(len(min_max_opti_para_main))]
        else:
            min_max_opti_para_main = [para_plot_opti.iloc[self.num_trials * (i):self.num_trials * (
                i + 1), -3].min() for i in range(int(to_fig_opti_results['Round'][-1].split('_')[-1]))]
            min_max_opti_para_ploting = [
                min(min_max_opti_para_main[:i+1]) for i in range(len(min_max_opti_para_main))]
        to_fig_opti_results[f'{self.opti_object_main} {self.opti_para_main}'] = min_max_opti_para_ploting
        if self.one_hp_check:
            selected_col_str = [
                key for key in self.hp_class.hps_dict_num.keys()][1]
            selected_col = [selected_col_str, self.opti_para_main]
        else:
            selected_col = [key for key in self.hp_class.hps_dict_num.keys()]
            selected_col.extend([self.opti_para_main])
        run_hps_perf = para_plot_opti[selected_col].copy()
        # print(run_hps_perf)

        fig = make_subplots(rows=1, cols=2, subplot_titles=(f'{self.opti_object_main} {self.opti_para_main} at each round',
                                                            f'{self.opti_para_main} distribution at each round'))
        fig_opti_para = px.line(to_fig_opti_results, x="Round",
                                y=f'{self.opti_object_main} {self.opti_para_main}')
        fig_dist = px.violin(para_plot_opti, x="Round",
                             y=self.opti_para_main, points='all')
        fig_hps_perf = px.parallel_coordinates(run_hps_perf)
        trace1 = fig_opti_para['data'][0]
        trace2 = fig_dist['data'][0]
        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(trace2, row=1, col=2)
        fig.update_layout(title_text="Optimization Results")
        fig.show()
        fig_hps_perf.show()

    def plot_search_space(self):
        # hps boundaries
        all_boundaries = pd.concat(
            [i for i in self.search_space_dict.values()], ignore_index=True)
        if self.one_hp_check:
            subs_title = [
                [key for key in self.hp_class.hps_dict_num.keys()][1]]
            fig1 = make_subplots(rows=1, cols=1)
            subs = [px.box(all_boundaries, x="Round", y=key)
                    for key in subs_title]
            for i in range(1):
                fig1.add_trace(subs[i]['data'][0], row=1, col=i+1)
        else:
            subs_title = [key for key in self.hp_class.hps_dict_num.keys()]
            fig1 = make_subplots(rows=1, cols=len(
                self.hp_class.hps_dict_num.keys()))
            subs = [px.box(all_boundaries, x="Round", y=key)
                    for key in self.hp_class.hps_dict_num.keys()]
            for i in range(len(self.hp_class.hps_dict_num.keys())):
                fig1.add_trace(subs[i]['data'][0], row=1, col=i+1)

        fig1.update_layout(
            title_text="hyperparameters min, max value at each round")
        for i in range(len(subs_title)):
            fig1.update_yaxes(title_text=subs_title[i], row=1, col=i+1)
        fig1.show()

        if self.one_hp_check:
            pass
        else:
            # to add 3D search space
            columns_3d = [key for key in self.hp_class.hps_dict_num.keys()]
            columns_3d.extend([self.opti_para_main])
            all_rounds_df_3d = self.gen_all_rounds_results().loc[:, columns_3d]
            for col in all_rounds_df_3d.columns:
                if col in self.hp_class.num_log:
                    all_rounds_df_3d[col] = np.log10(all_rounds_df_3d[col])

            model_3d_ref = self.aisara_core_fit_call(
                all_rounds_df_3d.iloc[:, :-1], all_rounds_df_3d.iloc[:, -1])
            dict_api_input_3d = {}
            initial_boundaries_3d = all_boundaries[all_boundaries['Round'] == 'Round_1']
            for col in all_rounds_df_3d.columns[:-1]:
                if col in self.hp_class.num_int:
                    dict_api_input_3d[col] = [int(i) for i in np.linspace(min(all_rounds_df_3d.loc[:, col].unique()),
                                                                          max(all_rounds_df_3d.loc[:, col].unique()), 17)]
                else:
                    dict_api_input_3d[col] = [i for i in np.linspace(min(all_rounds_df_3d.loc[:, col].unique()),
                                                                     max(all_rounds_df_3d.loc[:, col].unique()), 17)]

            chunk = 10000
            grid_df = pd.DataFrame(self.general_grid_call(dict_api_input_3d))
            prediction = []
            for i in range(round(len(grid_df) / chunk) + 1):
                lower_bound = i * chunk
                upper_bound = (i + 1) * chunk
                pred_df = grid_df.iloc[lower_bound:upper_bound, :]
                pre_predict, weight = self.aisara_core_predict_call(
                    model_3d_ref, pred_df)
                prediction.extend(pre_predict)
            grid_df['Prediction'] = prediction
            weight = pd.DataFrame(
                dict(zip(grid_df.columns[:-1], weight)), index=['impact'])
            grid_3d = grid_df.copy()
            impact_3d = weight.copy()

            impact_3d_sorted_dict = {k: v for k, v in sorted(
                impact_3d.T.to_dict()['impact'].items(), key=lambda x: x[1], reverse=True)}
            sorted_hps_list = [key for key in impact_3d_sorted_dict.keys()]
            surface_df = grid_3d.pivot_table(index=sorted_hps_list[0], columns=sorted_hps_list[1],
                                             values='Prediction')

            # x_range and y_range
            for col in all_rounds_df_3d.columns[:-1]:
                if col == sorted_hps_list[0]:
                    y_range = np.linspace(min(all_rounds_df_3d.loc[:, col].unique()), max(
                        all_rounds_df_3d.loc[:, col].unique()), surface_df.shape[0])
                elif col == sorted_hps_list[1]:
                    x_range = np.linspace(min(all_rounds_df_3d.loc[:, col].unique()), max(
                        all_rounds_df_3d.loc[:, col].unique()), surface_df.shape[1])
                else:
                    pass

            fig_3d = go.Figure(data=[go.Surface(
                z=surface_df.values, x=x_range, y=y_range, name='Aisara Surface')])
            fig_3d.update_traces(showlegend=True, showscale=False)
            fig_3d.add_trace(go.Scatter3d(x=all_rounds_df_3d.loc[:, sorted_hps_list[1]], y=all_rounds_df_3d.loc[:, sorted_hps_list[0]],
                             z=all_rounds_df_3d.loc[:, self.opti_para_main], mode='markers', marker=dict(color='grey'), name='Optimization Trials'))
            renamed_sorted_hps_list = []
            for col in sorted_hps_list[:2]:
                if (col in self.hp_class.num_int) or (col in self.hp_class.num_float):
                    renamed_sorted_hps_list.append(col)
                else:
                    renamed_sorted_hps_list.append('log_'+col)

            fig_3d.update_layout(
                scene=dict(
                    xaxis_title=renamed_sorted_hps_list[1],
                    yaxis_title=renamed_sorted_hps_list[0],
                    zaxis_title=self.opti_para_main))

            fig_3d.show()

    def gen_best_model_df(self):
        para_gen_best = self.gen_all_rounds_results()
        if self.opti_para_sup not in para_gen_best.columns:
            if self.opti_object_main == 'max':
                mask_model_1 = para_gen_best[self.opti_para_main] == para_gen_best[self.opti_para_main].max(
                )
            else:
                mask_model_1 = para_gen_best[self.opti_para_main] == para_gen_best[self.opti_para_main].min(
                )
            self.best_model_df = para_gen_best[mask_model_1]
        else:
            if self.opti_object_main == 'max':
                mask_model_1 = para_gen_best[self.opti_para_main] == para_gen_best[self.opti_para_main].max(
                )
            else:
                mask_model_1 = para_gen_best[self.opti_para_main] == para_gen_best[self.opti_para_main].min(
                )
            model_trans = para_gen_best[mask_model_1]
            if self.opti_object_sup == 'max':
                mask_model_2 = model_trans[self.opti_para_sup] == model_trans[self.opti_para_sup].max(
                )
            else:
                mask_model_2 = model_trans[self.opti_para_sup] == model_trans[self.opti_para_sup].min(
                )
            self.best_model_df = model_trans[mask_model_2]
        return self.best_model_df

    @property
    def best_model_hps(self):
        if self.one_hp_check:
            num_hps_best_model_hps = 1
        else:
            num_hps_best_model_hps = len(self.hp_class.hps_dict_num.keys())
        best_model_hp = self.gen_best_model_df(
        ).iloc[:1, :num_hps_best_model_hps]

        # export best parameters in dict
        best_parameters_dict = {}
        for i in range(len(best_model_hp.columns)):
            best_parameters_dict[best_model_hp.columns.tolist()[
                i]] = best_model_hp.iloc[0, i]

        return best_parameters_dict
