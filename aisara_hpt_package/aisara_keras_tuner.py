from math import log10
import requests
import pandas as pd
import numpy as np
import os
import json
from itertools import combinations
import random as rm
from getpass import getpass
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import tensorflow as tf
pio.templates.default = "plotly_white"



# raise exceptions for invalid user input
class UnknownArgument(ValueError):
    pass
class UnknownAttribute(ValueError):
    pass
class APIKey(ValueError):
    pass
class InvalidAPIKey(ValueError):
    pass

class Hp:
    def __init__(self):
        self.hps_dict_num = {} # this dict will be updated at each round
        self.hps_dict_str = {}
        self.num_linear=[]
        self.num_log=[]
        self.name_str=[]

    def numrange(self, name, min, max, type='linear'):
        if type.lower().strip()== 'linear':
            self.hps_dict_num[name.strip()] = [min, max]
            self.num_linear.append(name.strip())
            return min
        elif type.lower().strip() == 'log':
            self.hps_dict_num[name.strip()] = [log10(min),log10(max)]
            self.num_log.append(name.strip())
            return min
        else:
            raise UnknownArgument(f'unknown type found "{type}"... the type for numerical hyperparameters should be either "linear" or "log"')

    def strvalues(self, name, values):
        self.hps_dict_str[name.strip()]=values
        self.name_str.append(name.strip())
        return values[0]

    def remove_hp (self, name):
        if name.strip() in self.hps_dict_num.keys():
            try:
                del self.hps_dict_num[name.strip()]
            except:
                pass
            try:
                if name.strip() in self.num_linear:
                    self.num_linear.remove(name.strip())
                elif name.strip() in self.num_log:
                    self.num_log.remove(name.strip())
            except:
                pass
        elif name.strip() in self.hps_dict_str.keys():
            try:
                del self.hps_dict_str[name.strip()]
            except:
                pass
            try:
                self.name_str.remove(name.strip())
            except:
                pass
        else:
            raise UnknownArgument(f'"{name}" hyperparameter is not defined')

    @property
    def search_space_boundaries (self):
        hps_df=pd.DataFrame(self.hps_dict_num)
        for name in self.num_log:
            hps_df[name]=10**hps_df[name]
        hps_df.index=['min', 'max']
        return hps_df

    def lhc_call(self, input_dict, api_key):
        url = "https://aisara-hyperparameter-tuning.p.rapidapi.com/latin/uniform"
        headers = {
            'content-type': "application/json",
            'x-rapidapi-key': api_key,
            'x-rapidapi-host': "aisara-hyperparameter-tuning.p.rapidapi.com"
        }
        response = requests.request("POST", url, data=json.dumps(input_dict), headers=headers)
        return response.json()

    def hps_combinations(self, num_trials, api_key):
        dict_input = {}
        dict_input['subsection'] = num_trials
        for key, value in self.hps_dict_num.items():
            dict_input[key]=value
        hps_comb = pd.DataFrame(self.lhc_call(dict_input, api_key))
        for name in self.num_linear:
            hps_comb[name] = round(hps_comb[name]).astype('int')
        for name in self.num_log:
            hps_comb[name] = 10 ** hps_comb[name]
        return hps_comb


class HpOptimization:
    def __init__ (self, hp_class, model_func, opti_paras, opti_objects, num_trials=5, rounds=3, mode = 'c', api_key ='0000', aisara_seed = 'variable'):
        self.hp_class = hp_class
        self.model_func = model_func
        self.opti_para_main = opti_paras[0].strip()
        self.opti_para_sup = opti_paras[1].strip()
        self.opti_object_main = opti_objects[0].strip()
        self.opti_object_sup = opti_objects[1].strip()
        self.num_trials = num_trials
        self.max_rounds = rounds
        self.aisara_seed = aisara_seed.lower().strip()
        self.run_mode = mode.lower().strip()
        self.api_key = api_key


    def opti_objective_check(self):
        if self.opti_object_main == 'min' or self.opti_object_main == 'max':
            pass
        else:
            raise UnknownAttribute(f'''unknown optimization objective for the main parameter "{self.opti_object_main}"... optimization objective should be either "min" or "max"''')
        if self.opti_object_sup == 'min' or self.opti_object_sup == 'max':
            pass
        else:
            raise UnknownAttribute(f'''unknown optimization objective for the supporting parameter "{self.opti_object_sup}"... optimization objective should be either "min" or "max"''')

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
            if key == 'message' and value  == 'You are not subscribed to this API.':
                raise InvalidAPIKey('Incorrect API Key!!... please provide valid API key')
            else:
                pass

    def license_mode(self):
        print('\033[1m' + "For commercial use, you can obtain our API from https://rapidapi.com/aisara-technology-aisara-technology-default/api/aisara-hyperparameter-tuning\n"
                        'If you are a private user, set the mode parameter in HpOptimization class to "p".\n' + '\033[0m')

        if self.run_mode == 'p':
            self.api_key = "3b18dfe0e2mshaba41fff27bc31dp1fae09jsnc46d20ec56d2"  # free API key for mode 'p'
            self.api_subscription()
        elif self.run_mode == 'c':
            if self.api_key == "0000":
                raise APIKey('No API Key is provided!!... please provide the API key')
            elif self.api_key == "3b18dfe0e2mshaba41fff27bc31dp1fae09jsnc46d20ec56d2":
                raise InvalidAPIKey('Incorrect API Key!!... please provide a valid API key')
            else:
                self.api_subscription()
        else:
            raise UnknownAttribute(f'unknown mode "{self.run_mode}"... mode should be either "c" or "p"')

    def set_aisara_seed(self):
        if self.aisara_seed == 'fixed':
            os.environ['PYTHONHASHSEED'] = '0'
            np.random.seed(37)
            rm.seed(1254)
            tf.random.set_seed(89)
            sess = tf.compat.v1.Session(target='', graph=tf.compat.v1.get_default_graph())
            tf.compat.v1.keras.backend.set_session(sess)
        elif self.aisara_seed == 'variable':
            pass
        else:
            raise UnknownAttribute(f'unknown aisara_seed "{self.aisara_seed}"... aisara_seed should be either "fixed" or "variable"')

    def aisara_core_fit_call(self, x, y):
        x_train = []
        for col in x.columns:
            for v in x[col].tolist():
                x_train.append(v)
        x_train_str = str(x_train).replace('[', '').replace(']', '')
        y_train = y.tolist()
        y_train_str = str(y_train).replace('[', '').replace(']', '')

        url = "https://aisara-hyperparameter-tuning.p.rapidapi.com/developer/dfit"
        payload = {"params": {"ranking_method": "AiSaraRank3", "train_split": "100",
                              "train_x": x_train_str, "train_y": y_train_str,
                              "x_shape": f"{[i for i in x.shape]}", "y_shape": f"{[len(y_train),1]}"}}
        headers = {
            'content-type': "application/json",
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': "aisara-hyperparameter-tuning.p.rapidapi.com"
        }
        response = requests.request("POST", url, data=json.dumps(payload), headers=headers)
        return response.json()['result']

    def aisara_core_predict_call(self, model_id, x):
        x_shape = [i for i in x.shape]
        x_train = []
        for index, row in x.iterrows():
            for col in x.columns:
                x_train.append(row[col])
        x_train_str = str(x_train).replace('[', '').replace(']', '')
        url = "https://aisara-hyperparameter-tuning.p.rapidapi.com/developer/dpred"
        payload = {"params": {"model_id": model_id, "test_x": x_train_str, "x_shape": f"{x_shape}", "y_selected": "1"}}
        headers = {
            'content-type': "application/json",
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': "aisara-hyperparameter-tuning.p.rapidapi.com"
        }
        response = requests.request("POST", url, data=json.dumps(payload), headers=headers)
        prediction = response.json()['result']
        impact_mod = response.json()['impacts']
        impact_discimal = [sum(i) for i in impact_mod]
        impact = [round(i * 100, 3) for i in impact_discimal]
        impact.reverse()
        return prediction, impact

    def df_to_sting_x(self, results_log_df, target):
        df = results_log_df.copy()
        if target == 'error':
            del df[self.opti_para_sup]
            del df[self.opti_para_main]
            del df[f'{self.opti_para_main}_predict']
            if len(self.his_keys) > 2:
                del df[self.not_in_keys[0]]
                del df[self.not_in_keys[1]]
        else:
            del df[self.opti_para_sup]
            del df[f'{self.opti_para_main}_predict']
            del df['error']
            del df['aisara_max_error']
            if len(self.his_keys) > 2:
                del df[self.not_in_keys[0]]
                del df[self.not_in_keys[1]]

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
            if col in self.hp_class.num_linear:
                dict_api_input[col] = [int(i) for i in np.linspace(self.hp_class.hps_dict_num[col][0], self.hp_class.hps_dict_num[col][1], 17)]
            else:
                dict_api_input[col] = [i for i in np.linspace(self.hp_class.hps_dict_num[col][0], self.hp_class.hps_dict_num[col][1], 17)]

        dict_api_input["x_train"] = df_srt
        dict_api_input["x_shape"] = x_shape
        dict_api_input["metric"] = f"{min_max}"
        dict_api_input["loss"] = "mae"

        url = "https://aisara-hyperparameter-tuning.p.rapidapi.com/grid/general_pred"
        headers = {
            'content-type': "application/json",
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': "aisara-hyperparameter-tuning.p.rapidapi.com"
        }
        api_results = requests.request("POST", url, data=json.dumps(dict_api_input), headers=headers)

        impact = api_results.json()['Impact']
        best_comp = api_results.json()['Best combination']
        best_comp_df = pd.DataFrame(best_comp)
        best_comp_df.rename(columns={'Prediction': f'aisara_{min_max}_{target}'}, inplace=True)
        return best_comp_df, impact

    def general_grid_call(self, input):
        url = "https://aisara-hyperparameter-tuning.p.rapidapi.com/grid/general"
        headers = {
            'content-type': "application/json",
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': "aisara-hyperparameter-tuning.p.rapidapi.com"
        }
        response = requests.request("POST", url, data=json.dumps(input), headers=headers)
        return response.json()

    def run_opti(self):
        # optimization objective check
        self.opti_objective_check()

        # License mode
        self.license_mode()

        # fixed seed
        self.set_aisara_seed()

        # making a folder to save models
        try:
            os.makedirs('models')
        except:
            pass
        folder = './models'
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)

        # dictionary contains rounds results
        self.rounds_dict={}
        # choosing between user's rounds or aisara_auto round
        if self.max_rounds == 'aisara_auto':
            self.max_rounds = 1000

        # list contains best hps at each run
        rounds_best_hps=[]
        self.search_space_dict = {}
        for round in range(self.max_rounds):
            print('\033[1m'+'\033[94m'+f'Round-{round+1}:'+'\033[0m')
            model_id=[] # names of the model id at each round [round, run], this will be used to get the best model from models folder
            self.comp = self.hp_class.hps_combinations(self.num_trials, self.api_key)  # create lhc combinations
            opti_param_values = []  # list contains opti_apara main value at each run
            opti_paras_values = []  # list contains opti_apara sup value at each run
            remaining_hist_keys_1 = [] # remaining metrics
            remaining_hist_keys_2 = []  # remaining metrics
            for num in range(self.num_trials):
                print('\n'+'\033[1m'+'\033[92m'+f'  Run-{num+1}:'+'\033[0m')
                run_model, run_hist=self.model_func(self.comp, num)
                run_model.save(f'./models/model_{round+1}_{num+1}.h5')
                self.his_keys = run_hist.history.keys()

                # check optimization parameter main and sub
                if self.opti_para_main in self.his_keys:
                    pass
                else:
                    raise UnknownAttribute(f'"{self.opti_para_main}" is not in history keys')
                if self.opti_para_sup in self.his_keys:
                    pass
                else:
                    raise UnknownAttribute(f'"{self.opti_para_sup}" is not in history keys')

                self.not_in_keys = [key for key in self.his_keys if key not in [self.opti_para_sup, self.opti_para_main]]
                if len(self.his_keys) > 2:
                    remaining_hist_keys_1.append(run_hist.history[self.not_in_keys[0]][-1])
                    remaining_hist_keys_2.append(run_hist.history[self.not_in_keys[1]][-1])
                opti_param_values.append(run_hist.history[self.opti_para_main][-1])
                opti_paras_values.append(run_hist.history[self.opti_para_sup][-1])
                model_id.append(f'model_{round+1}_{num+1}')

            print('\033[1m'+'\n\nsearch space boundaries:\n'+'\033[0m',self.hp_class.search_space_boundaries,'\n')
            print('\033[1m'+'hyperparameters combinations (lHC):\n'+'\033[0m',self.comp.to_string(index=False),'\n')
            to_search_space_dict = self.hp_class.search_space_boundaries
            to_search_space_dict.index = [0,1]
            to_search_space_dict['Round'] = f'Round_{round+1}'
            self.search_space_dict[f'Round_{round + 1}']= to_search_space_dict

            self.round_results = self.comp.copy()
            if len(self.his_keys) > 2:
                self.round_results[self.not_in_keys[0]] = remaining_hist_keys_1
                self.round_results[self.not_in_keys[1]] = remaining_hist_keys_2
            self.round_results[self.opti_para_sup] = opti_paras_values
            self.round_results[self.opti_para_main] = opti_param_values
            print('\033[1m'+f'models results:\n''\033[0m'+'\033[0m',self.round_results.to_string(index=False))

            to_rounds_dict = self.round_results.copy()
            to_rounds_dict['Round'] = f'Round_{round + 1}'
            to_rounds_dict['model ID'] = model_id
            self.rounds_dict[f'Round_{round + 1}'] = to_rounds_dict

            if round < self.max_rounds-1:
                # creation of new search space boundaries
                self.round_results_log = self.round_results.copy()
                for name in self.hp_class.num_log:
                    if name in self.round_results_log.columns:
                        self.round_results_log[name] = np.log10(self.round_results_log[name])

                dict_poped_col={} # it will be added later to self.hps_dict_num
                if len(self.his_keys) == 2:
                    self.round_results_log_hps_col = self.round_results_log.columns[:-2]
                else:
                    self.round_results_log_hps_col = self.round_results_log.columns[:-4]
                stop_cols = [x for x in self.round_results_log_hps_col.copy()]
                for col_name in self.round_results_log_hps_col: # excluding opti_para_main and opti_para_sup
                    if len(self.round_results_log[col_name].unique()) == 1: # checking if the col contains similar values
                        dict_poped_col[col_name]=[self.round_results_log[col_name].unique()[0], self.round_results_log[col_name].unique()[0]]
                        del self.round_results_log[col_name]
                        stop_cols.remove(col_name)

                if len(stop_cols) >= 2: # checking that aisara will receive min of 1 col with non unique values
                    print('\033[1m'+'\nworking on the creation of reduced search space boundaries for the next round, pls wait this might take some time'+'\033[0m')
                # hps of best round run:
                    if self.opti_object_main == 'max':
                        mask1= self.round_results_log[self.opti_para_main] == self.round_results_log[self.opti_para_main].max()
                    else:
                        mask1 = self.round_results_log[self.opti_para_main] == self.round_results_log[self.opti_para_main].min()
                    trans = self.round_results_log[mask1]
                    if self.opti_object_sup == 'max':
                        mask2 = trans[self.opti_para_sup] == trans[self.opti_para_sup].max()
                    else:
                        mask2 = trans[self.opti_para_sup] == trans[self.opti_para_sup].min()

                    # saving round best results for comp
                    to_rounds_best_hps = trans[mask2].iloc[:1, : ].copy()
                    to_rounds_best_hps.index = [0]
                    rounds_best_hps.append(to_rounds_best_hps)

                    if round > 0:
                        for key, value in dict_poped_col.items():
                            if key in rounds_best_hps[round-1].columns:
                                del rounds_best_hps[round-1][key]
                        self.round_results_log = pd.concat([self.round_results_log, rounds_best_hps[round-1]], ignore_index=True)

                        if self.opti_object_main == 'max':
                            mask1_1= self.round_results_log[self.opti_para_main] == self.round_results_log[self.opti_para_main].max()
                        else:
                            mask1_1 = self.round_results_log[self.opti_para_main] == self.round_results_log[self.opti_para_main].min()
                        trans_1 = self.round_results_log[mask1_1]
                        if self.opti_object_sup == 'max':
                            mask2_1 = trans_1[self.opti_para_sup] == trans_1[self.opti_para_sup].max()
                        else:
                            mask2_1 = trans_1[self.opti_para_sup] == trans_1[self.opti_para_sup].min()
                        if len(self.his_keys) == 2:
                            best_hps_round = trans_1[mask2_1].iloc[:1, :-2].copy()
                        else:
                            best_hps_round = trans_1[mask2_1].iloc[:1, :-4].copy()
                        best_hps_round.index = ['round_best_hps']
                        best_hps_round_t = best_hps_round.T
                    else:
                        if len(self.his_keys) == 2:
                            best_hps_round = trans[mask2].iloc[:1, :-2].copy()
                        else:
                            best_hps_round = trans[mask2].iloc[:1, :-4].copy()
                        best_hps_round.index = ['round_best_hps']
                        best_hps_round_t = best_hps_round.T



                # 1) multi_error_calc:
                    all_index = list(self.round_results_log.index)
                    index_predic = []
                    for m in combinations(self.round_results_log.index, len(self.round_results_log.index) - 1):
                        comb_multi = list(m)
                        not_in = [l for l in all_index if l not in comb_multi]
                        if len(self.his_keys) == 2:
                            ref = self.aisara_core_fit_call(self.round_results_log.iloc[comb_multi, :-2], self.round_results_log.iloc[comb_multi, -1])
                        else:
                            ref = self.aisara_core_fit_call(self.round_results_log.iloc[comb_multi, :-4], self.round_results_log.iloc[comb_multi, -1])
                        if len(self.his_keys) == 2:
                            predict, _ = self.aisara_core_predict_call(ref, self.round_results_log.iloc[not_in, :-2])
                        else:
                            predict, _ = self.aisara_core_predict_call(ref, self.round_results_log.iloc[not_in, :-4])
                        index_predic.append((not_in[0], predict[0]))

                    index_predic.sort(key=lambda x: x[0])
                    self.round_results_log[f'{self.opti_para_main}_predict'] = [i[1] for i in index_predic]
                    self.round_results_log['error'] = abs(self.round_results_log[f'{self.opti_para_main}']
                                                      - self.round_results_log[f'{self.opti_para_main}_predict']) \
                                                      / self.round_results_log[f'{self.opti_para_main}']

                # 2) aisara_max error:
                    best_comp_e, _ =self.gen_pred_call(self.round_results_log,'error','max')
                    self.round_results_log = pd.concat([self.round_results_log, best_comp_e], ignore_index=True)
                    for key, value in dict_poped_col.items():
                        best_comp_e[key]= value[0]
                    del best_comp_e['aisara_max_error']
                    for name in self.hp_class.num_log:
                        best_comp_e[name]=10**(best_comp_e[name])

                    print('\n'+'\033[1m'+'\033[95m'+'Aisara_max_error Run:'+'\033[0m')
                    _, run_hist_e = self.model_func(best_comp_e, 0)
                    if round == 0:
                        self.round_results_log.at[self.num_trials, self.opti_para_sup] = run_hist_e.history[self.opti_para_sup][-1]
                        self.round_results_log.at[self.num_trials, self.opti_para_main] = run_hist_e.history[self.opti_para_main][-1]
                    else:
                        self.round_results_log.at[self.num_trials+1, self.opti_para_sup] = run_hist_e.history[self.opti_para_sup][-1]
                        self.round_results_log.at[self.num_trials+1, self.opti_para_main] = run_hist_e.history[self.opti_para_main][-1]

                # 3) aisara_max acc/mae:
                    best_comp_a_m, impact_a_m = self.gen_pred_call(self.round_results_log, self.opti_para_main, self.opti_object_main)
                    self.round_results_log = pd.concat([self.round_results_log, best_comp_a_m], ignore_index=True)

                # aisara best hps of best round run:
                    best_hps_aisara = best_comp_a_m.iloc[:,:-1]
                    best_hps_aisara.index = ['aisara_best_hps']
                    best_hps_aisara_t = best_hps_aisara.T

                # 4) round best and aisara best (error, modified_error, min and max):
                    error_perc_df = pd.concat([best_hps_round_t, best_hps_aisara_t], axis=1)
                    error_perc_df['error'] = abs((error_perc_df['round_best_hps'] - error_perc_df['aisara_best_hps'])) / error_perc_df['round_best_hps']
                    impact_df = pd.DataFrame(impact_a_m, index=[0])
                    impact_df.index = ['impact_factor']
                    impact_df_t = impact_df.T
                    error_impact_df = pd.concat([error_perc_df, impact_df_t], axis=1)
                    error_impact_df['modified_error'] = np.where(error_impact_df['impact_factor'] < 1, error_impact_df['error'] * (1 - error_impact_df['impact_factor']),
                                                                                                        error_impact_df['error'])
                    error_impact_df['min_*'] = error_impact_df['round_best_hps'] * (1 - error_impact_df['modified_error'])
                    error_impact_df['max_*'] = error_impact_df['round_best_hps'] * (1 + error_impact_df['modified_error'])

                # 5) new boundaries min, max:
                    list_min=[]
                    for row in error_impact_df.index:
                        if row in self.hp_class.num_linear:
                            list_min.append(np.round(max(error_impact_df.loc[row, 'min_*'], self.hp_class.hps_dict_num[row][0])))
                        elif row in self.hp_class.num_log:
                            list_min.append(max(error_impact_df.loc[row, 'min_*'], self.hp_class.hps_dict_num[row][0]))
                    error_impact_df['min'] = list_min
                    list_max=[]
                    for row in error_impact_df.index:
                        if row in self.hp_class.num_linear:
                            list_max.append(np.round(min(error_impact_df.loc[row, 'max_*'], self.hp_class.hps_dict_num[row][1])))
                        elif row in self.hp_class.num_log:
                            list_max.append(min(error_impact_df.loc[row, 'max_*'], self.hp_class.hps_dict_num[row][1]))
                    error_impact_df['max']=list_max

                    dict_hps_new_ss_log = error_impact_df.iloc[:,-2:].T
                    for col in dict_hps_new_ss_log.columns:
                        if col in self.hp_class.num_linear:
                            dict_hps_new_ss_log[col] = dict_hps_new_ss_log[col].astype('int')

                # update self.hp_class.hps_dict_num
                    for key, value in dict_hps_new_ss_log.to_dict().items():
                        self.hp_class.hps_dict_num[key] = [value['min'], value['max']]
                    for key, value in dict_poped_col.items():
                        self.hp_class.hps_dict_num[key] = [value[0].item(), value[0].item()]
                    print()
                else:
                    print('\033[1m' + '\033[91m' + '\nmaximum search space reduction has been achieved, not future optimization can be done.' + '\033[0m')
                    break

            else:
                print('\033[1m'+'\noptimization has ended'+'\033[0m')
                break

    def gen_all_rounds_results(self):
        self.all_rounds_results = pd.concat([i for i in self.rounds_dict.values()], ignore_index=True)
        return self.all_rounds_results

    @property
    def opti_results(self):
        return self.gen_all_rounds_results().to_string(index=False)


    def plot_opti_results(self):
        para_plot_opti = self.gen_all_rounds_results()
        to_fig_opti_results={}
        to_fig_opti_results['Round'] = para_plot_opti['Round'].unique()
        if self.opti_object_main == 'max':
            min_max_opti_para_main = [para_plot_opti.iloc[self.num_trials * (i):self.num_trials * (i + 1), -3].max() for i in range(int(to_fig_opti_results['Round'][-1].split('_')[-1]))]
            min_max_opti_para_ploting = [max(min_max_opti_para_main[:i + 1]) for i in range(len(min_max_opti_para_main))]
        else:
            min_max_opti_para_main = [para_plot_opti.iloc[self.num_trials * (i):self.num_trials * (i + 1), -3].min() for i in range(int(to_fig_opti_results['Round'][-1].split('_')[-1]))]
            min_max_opti_para_ploting =[min(min_max_opti_para_main[:i+1]) for i in range(len(min_max_opti_para_main))]
        to_fig_opti_results[f'{self.opti_object_main} {self.opti_para_main}'] = min_max_opti_para_ploting
        selected_col =[key for key in self.hp_class.hps_dict_num.keys()]
        selected_col.extend([self.opti_para_main])
        run_hps_perf = para_plot_opti[selected_col].copy()

        fig = make_subplots(rows=1, cols=2,subplot_titles=(f'{self.opti_object_main} {self.opti_para_main} at each round',
                                                           f'{self.opti_para_main} distribution at each round'))
        fig_opti_para = px.line(to_fig_opti_results, x="Round", y=f'{self.opti_object_main} {self.opti_para_main}')
        fig_dist = px.violin(para_plot_opti, x="Round", y=self.opti_para_main, points='all')
        fig_hps_perf = px.parallel_coordinates(run_hps_perf)
        trace1 = fig_opti_para ['data'][0]
        trace2 = fig_dist['data'][0]
        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(trace2, row=1, col=2)
        fig.update_layout(title_text="Optimization Results")
        fig.show()
        fig_hps_perf.show()

    def plot_search_space(self):
        # hps boundaries
        all_boundaries = pd.concat([i for i in self.search_space_dict.values()], ignore_index=True)
        subs_title=[key for key in self.hp_class.hps_dict_num.keys()]
        fig1= make_subplots(rows=1, cols=len(self.hp_class.hps_dict_num.keys()))
        subs = [px.box(all_boundaries, x="Round", y=key) for key in self.hp_class.hps_dict_num.keys()]
        for i in range(len(self.hp_class.hps_dict_num.keys())):
            fig1.add_trace(subs[i]['data'][0], row=1, col=i+1)

        fig1.update_layout(title_text="hyperparameters min, max value at each round")
        for i in range(len(subs_title)):
            fig1.update_yaxes(title_text=subs_title[i], row=1, col=i+1)
        fig1.show()

        # to add 3D search space
        columns_3d = [key for key in self.hp_class.hps_dict_num.keys()]
        columns_3d.extend([self.opti_para_main])
        all_rounds_df_3d = self.gen_all_rounds_results().loc[:, columns_3d]
        for col in all_rounds_df_3d.columns:
            if col in self.hp_class.num_log:
                all_rounds_df_3d[col] = np.log10(all_rounds_df_3d[col])

        model_3d_ref = self.aisara_core_fit_call(all_rounds_df_3d.iloc[:,:-1], all_rounds_df_3d.iloc[:,-1])
        dict_api_input_3d = {}
        initial_boundaries_3d = all_boundaries[all_boundaries['Round'] == 'Round_1']
        for col in all_rounds_df_3d.columns[:-1]:
            if col in self.hp_class.num_linear:
                dict_api_input_3d[col] = [int(i) for i in np.linspace(min(all_rounds_df_3d.loc[:,col].unique()),
                                                                      max(all_rounds_df_3d.loc[:,col].unique()), 17)]
            else:
                dict_api_input_3d[col] = [i for i in np.linspace(min(all_rounds_df_3d.loc[:,col].unique()),
                                                                 max(all_rounds_df_3d.loc[:,col].unique()), 17)]

        chunk=10000
        grid_df = pd.DataFrame(self.general_grid_call(dict_api_input_3d))
        prediction = []
        for i in range(round(len(grid_df) / chunk) + 1):
            lower_bound = i * chunk
            upper_bound = (i + 1) * chunk
            pred_df = grid_df.iloc[lower_bound:upper_bound, :]
            pre_predict, weight = self.aisara_core_predict_call(model_3d_ref, pred_df)
            prediction.extend(pre_predict)
        grid_df['Prediction'] = prediction
        weight = pd.DataFrame(dict(zip(grid_df.columns[:-1], weight)), index=['impact'])
        grid_3d = grid_df.copy()
        impact_3d = weight.copy()

        impact_3d_sorted_dict = {k:v for k,v in sorted(impact_3d.T.to_dict()['impact'].items(), key= lambda x: x[1], reverse= True)}
        sorted_hps_list = [key for key in impact_3d_sorted_dict.keys()]
        surface_df = grid_3d.pivot_table(index=sorted_hps_list[0], columns=sorted_hps_list[1],
                                         values='Prediction')

        # x_range and y_range
        for col in all_rounds_df_3d.columns[:-1]:
            if col == sorted_hps_list[0]:
                 y_range = np.linspace(min(all_rounds_df_3d.loc[:,col].unique()), max(all_rounds_df_3d.loc[:,col].unique()), surface_df.shape[0])
            elif col == sorted_hps_list[1]:
                x_range = np.linspace(min(all_rounds_df_3d.loc[:,col].unique()), max(all_rounds_df_3d.loc[:,col].unique()), surface_df.shape[1])
            else:
                pass

        fig_3d = go.Figure(data=[go.Surface(z=surface_df.values, x=x_range, y=y_range, name='Aisara Surface')])
        fig_3d.update_traces(showlegend=True, showscale=False)
        fig_3d.add_trace(go.Scatter3d(x=all_rounds_df_3d.loc[:, sorted_hps_list[1]], y=all_rounds_df_3d.loc[:, sorted_hps_list[0]],
                         z=all_rounds_df_3d.loc[:, self.opti_para_main], mode='markers', marker=dict(color='grey'), name='Optimization Trials'))
        renamed_sorted_hps_list = []
        for col in sorted_hps_list[:2]:
            if col in self.hp_class.num_linear:
                renamed_sorted_hps_list.append(col)
            else:
                renamed_sorted_hps_list.append('log_'+col)

        # log_table_name_list = []
        # for col in sorted_hps_list[:2]:
        #     if col in self.hp_class.num_log:
        #         log_table_name_list.append(col)
        # if len(log_table_name_list) > 0:
        #     log_log_table_v_list = np.linspace(np.log10(initial_boundaries_3d.loc[0,log_table_name_list[0]]),
        #                                        np.log10(initial_boundaries_3d.loc[1,log_table_name_list[0]]), 3)
        #     log_table_v_list  = [10**i for i in log_log_table_v_list]
        #     table_fig = go.Table(
        #             header=dict(
        #                 values=[log_table_name_list[0], "log_"+log_table_name_list[0]],
        #                 font=dict(size=15)),
        #             cells=dict(
        #                 values=[log_table_v_list, log_log_table_v_list]))
        #     fig_3d.add_trace(table_fig)

        fig_3d.update_layout(
            scene = dict(
                xaxis_title=renamed_sorted_hps_list[1],
                yaxis_title=renamed_sorted_hps_list[0],
                zaxis_title=self.opti_para_main))

        fig_3d.show()

    def gen_best_model_df(self):
        para_gen_best = self.gen_all_rounds_results()
        if self.opti_object_main == 'max':
            mask_model_1 = para_gen_best[self.opti_para_main] == para_gen_best[self.opti_para_main].max()
        else:
            mask_model_1 = para_gen_best[self.opti_para_main] == para_gen_best[self.opti_para_main].min()
        model_trans = para_gen_best[mask_model_1]
        if self.opti_object_sup == 'max':
            mask_model_2 = model_trans[self.opti_para_sup] == model_trans[self.opti_para_sup].max()
        else:
            mask_model_2 = model_trans[self.opti_para_sup] == model_trans[self.opti_para_sup].min()
        self.best_model_df = model_trans[mask_model_2]

        return self.best_model_df

    @property
    def best_model_hps(self):
        num_hps_best_model_hps = len(self.hp_class.hps_dict_num.keys())
        best_model_hp = self.gen_best_model_df().iloc[:1,:num_hps_best_model_hps]
        return best_model_hp.to_string(index=False)

    @property
    def best_model(self):
        best_model_id = self.gen_best_model_df().iloc[:1,-1]
        best_model_id_list = best_model_id.tolist()
        return tf.keras.models.load_model(f'./models/{best_model_id_list[0]}.h5')



