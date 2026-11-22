
from src.Logger import Logger
import utils_v1
from icecream import ic
import numpy as np
import os
import time

from sklearn import metrics
from sklearn.metrics import f1_score
from src import metrics as _metrics
from enum import Enum
import matplotlib.pyplot as plt
from scipy import optimize  
from src.manager.base import Manager
import src.uncertainty as uncertainty
import pathlib
import pdb

from src.modelGenerator import generate_model

import torch
from numpy import inf
import torch.nn as nn
from tqdm import tqdm

from src.trainer import Trainer
class ManagerMultiOutput(Manager):
    def __init__(self, config, dataset, patchesHandler, logger, grid_idx=0):
        super().__init__(config, dataset, patchesHandler, logger, grid_idx=grid_idx)
        # self.network_architecture = utils_v1.build_resunet_dropout_spatial
        self.pred_entropy_single_idx = 0
        

    def train(self):
        for tm in range(self.config['training_times']):
            if self.config['training_times'] == 1:
                trainer = Trainer(self.config)
            else:
                trainer = Trainer(self.config, str(tm))
            t0 = time.time()
            trainer.train(self.train_dataloader, self.val_dataloader)
            print("Training time: {:.2f} s".format(time.time() - t0))
    def getMeanProb(self):
        self.mean_prob = np.mean(self.prob_rec, axis = 0)
        if self.classes_mode == True:
            self.mean_prob = self.mean_prob[...,1]            

    def preprocessProbRec(self):
        if self.classes_mode == False:
            print(f"preprocessProbRec=> self.prob_rec.shape: {self.prob_rec.shape}")
            #self.prob_rec = np.transpose(self.prob_rec, (2, 0, 1))
            self.prob_rec = np.expand_dims(self.prob_rec, axis = -1)
            print(f"preprocessProbRec=> self.prob_rec.shape: {self.prob_rec.shape}")
        else:
            # this np.transpose is untested, but since the the other one needed to be removed, I think this one is also wrong. 
            # The shape of prob_rec should be (inference_times, h, w)
            self.prob_rec = np.transpose(self.prob_rec, (3, 0, 1, 2))

    def setUncertainty(self):

        if self.config['uncertainty_method'] == "pred_var":
            self.uncertainty_map = uncertainty.predictive_variance(self.prob_rec).astype(np.float32)

        elif self.config['uncertainty_method'] == "MI":
            self.uncertainty_map = uncertainty.mutual_information(self.prob_rec).astype(np.float32)

        elif self.config['uncertainty_method'] == "pred_entropy":
            self.uncertainty_map = uncertainty.predictive_entropy(self.prob_rec, self.classes_mode).astype(np.float32)

        elif self.config['uncertainty_method'] == "KL":
            self.uncertainty_map = uncertainty.expected_KL_divergence(self.prob_rec).astype(np.float32)

        elif self.config['uncertainty_method'] == "pred_entropy_single":
            self.uncertainty_map = uncertainty.single_experiment_entropy(
                self.prob_rec[self.pred_entropy_single_idx], self.classes_mode).astype(np.float32)

    def getPOIValues(self):
        self.snippet_poi_results = []

        lims_snippets = [self.dataset.previewLims1, self.dataset.previewLims2]
        for snippet_id, lims in enumerate(lims_snippets):
            for coord in self.dataset.snippet_coords["snippet_id{}".format(snippet_id)]:
                dict_ = {"snippet_id": snippet_id,
                        "coords": coord, # 10,1 alpha
                        "reference": self.label_mask[lims[0]:lims[1], lims[2]:lims[3]][coord[0], coord[1]]}
                
                predicted_coord = []
                for idx in range(self.prob_rec.shape[0]):
                    predicted_coord.append(self.prob_rec[idx][lims[0]:lims[1], lims[2]:lims[3]][coord[0], coord[1]])
                predicted_coord = np.array(predicted_coord)
                dict_["predicted"] = predicted_coord

                self.snippet_poi_results.append(dict_)

        return self.snippet_poi_results
class ManagerMCDropout(ManagerMultiOutput):
    def __init__(self, config, dataset, patchesHandler, logger, grid_idx=0):
        config['dropout_training'] = True
        super().__init__(config, dataset, patchesHandler, logger, grid_idx)
        self.default_log_name = 'output/log/log_mcd.pkl'

class ManagerSingleRun(ManagerMultiOutput):
    def __init__(self, config, dataset, patchesHandler, logger, grid_idx=0):
        config['dropout_training'] = False
        super().__init__(config, dataset, patchesHandler, logger, grid_idx)
        self.default_log_name = 'output/log/log_single_run.pkl'

# class ManagerEnsemble(ManagerMCDropout):
class ManagerEnsemble(ManagerMultiOutput):
    def __init__(self, config, dataset, patchesHandler, logger, grid_idx=0):
        config['dropout_training'] = False
        super().__init__(config, dataset, patchesHandler, logger, grid_idx)
        self.default_log_name = 'output/log/log_ensemble.pkl'

    def infer(self, device="cpu"):
        self.h, self.w, self.c = self.image1_pad.shape
        self.c = self.channels
        patch_size_rows = self.h//self.n_rows
        patch_size_cols = self.w//self.n_cols
        num_patches_x = int(self.h/patch_size_rows)
        num_patches_y = int(self.w/patch_size_cols)


        #self.device = torch.device(self.config['General']['device'] if torch.cuda.is_available() else "cpu")
        self.device = device
        
        print("device: %s" % self.device)     

        class_n = 3
        
        new_model = generate_model(self.config)

        new_model.eval()
        
        
        if self.config["loadInference"] == False:
            if self.classes_mode == False:
                self.prob_rec = np.zeros((self.image1_pad.shape[0],self.image1_pad.shape[1], self.config["inference_times"]), dtype = np.float32)
            else:
                self.prob_rec = np.zeros((self.image1_pad.shape[0],self.image1_pad.shape[1], class_n - 1, self.config["inference_times"]), dtype = np.float32)

            self.patchesHandler.class_n = class_n
            # pathlib.Path(self.path_maps).mkdir(parents=True, exist_ok=True)
            metrics_all =[]

            
            for tm in range(0,self.config["inference_times"]):
                print('time: ', tm)
                
                # Recinstructing predicted map
                start_test = time.time()

                path_exp = self.dataset.paths.experiment + 'exp' + str(self.exp) # exp_ids[tm]
                path_models = path_exp + '/models'
                
                # ic(path_models+ '/' + method +'_'+str(0)+'.h5')
                model_name = new_model.__class__.__name__
                if self.config['General']['model_type'] == 'resunet':
                    model_name = model_name.replace("Unet","ResUNet")
                path_model = os.path.join(self.config['General']['path_model'], 
                    model_name + 
                    '_' + str(self.config['General']['exp_id']) + str(tm) + '.pth')
                
                ic(path_model)

                with open(path_model, "rb") as f:
                    new_model.load_state_dict(torch.load(f), strict = True)
                    print(f"Model weights loaded from {path_model}")

                """
                model = utils_v1.load_model(path_models+ '/' + self.method +'_'+str(tm)+'.h5', compile=False)
                for l in range(1, len(model.layers)): #WHY 1?
                    new_model.layers[l].set_weights(model.layers[l].get_weights())
                """
                '''
                args_network = {'patch_size_rows': patch_size_rows,
                    'patch_size_cols': patch_size_cols,
                    'c': c,
                    'nb_filters': nb_filters,
                    'class_n': class_n,
                    'dropout_seed': inference_times}
                '''
                prob_reconstructed = self.patchesHandler.infer(
                        new_model, self.image1_pad, self.h, self.w, 
                        num_patches_x, num_patches_y, patch_size_rows, 
                        patch_size_cols, classes_mode = self.classes_mode, device = self.device)
                        
                ts_time =  time.time() - start_test

                if self.config["save_probabilities"] == True:     
                    np.save(os.path.join(self.path_maps, model_name +'prob_'+str(tm)+'.npy'),prob_reconstructed)
                    print(f"Saved inference results at {os.path.join(self.path_maps, model_name +'prob_'+str(tm)+'.npy')}")
                else:
                    print(f"config['save_probabilities'] is {self.config['save_probabilities']} so not saving the inference result")
                    
                self.prob_rec[...,tm] = prob_reconstructed

                metrics_all.append(ts_time)
                del prob_reconstructed
            metrics_ = np.asarray(metrics_all)
            # Saving test time
            np.save(self.path_exp+'/metrics_ts.npy', metrics_)
            
        else: # load the inference result instead of redoing it again
            if self.classes_mode == False:
                self.prob_rec = np.zeros((self.image1_pad.shape[0],self.image1_pad.shape[1], self.config["inference_times"]), dtype = np.float32)
            else:
                self.prob_rec = np.zeros((self.image1_pad.shape[0],self.image1_pad.shape[1], class_n - 1, self.config["inference_times"]), dtype = np.float32)

            for tm in range(0,self.config["inference_times"]):
                self.prob_rec[...,tm] = np.load(os.path.join(self.path_maps, 'prob_'+str(tm)+'.npy'))

        self.backup_prob_rec = self.prob_rec.copy()
        del self.image1_pad

    def run_predictor_repetition_single_entropy(self):
        # self.setExperimentPath()
        # self.createLogFolders()        
        self.setPadding()
        self.infer()
        self.loadPredictedProbabilities()
        self.getMeanProb()
        self.unpadMeanProb()
        self.squeezeLabel()
        self.setMeanProbNotConsideredAreas()
        self.getLabelTest()
        # self.getMAP()
        self.preprocessProbRec()
        # self.getUncertaintyToShow(self.pred_entropy)
        self.getLabelCurrentDeforestation()
        
        
        # min_metric = np.inf
        # max_metric = 0
        self.config['uncertainty_method'] = "pred_entropy_single"
        results = {}
        for idx in range(self.config['inference_times']):
            self.pred_entropy_single_idx = idx
            self.applyProbabilityThreshold() # start from here for single entropy loop
            self.getTestValues()
            self.removeSmallPolygons()
            self.calculateMetrics()
            self.getValidationValuesForMetrics()
            self.calculateMetricsValidation()
            calculateMAPWithoutSmallPolygons = False
            if calculateMAPWithoutSmallPolygons == True:
                self.calculateMAPWithoutSmallPolygons()
            self.getErrorMask()
            self.getErrorMaskToShowRGB()

            self.setUncertainty()
            self.getValidationValues2()
            self.getTestValues2()
            self.getOptimalUncertaintyThreshold()

            results["pred_entropy_single_{}".format(idx)] = self.getUncertaintyMetricsFromOptimalThreshold()
            
            '''
            results_tmp = self.getUncertaintyMetricsFromOptimalThreshold()
            metric = self.f1
            if metric > max_metric:
                max_metric = metric
                results["pred_entropy_single_max"] = results_tmp
            if metric < min_metric:
                min_metric = metric
                results["pred_entropy_single_min"] = results_tmp
            '''
        
        print("results", results)
        return results

        
    def defineExperiment(self, exp_id):
        self.exp = exp_id

    def getPOIValues(self):
        self.snippet_poi_results = []

        lims_snippets = [self.dataset.previewLims1, self.dataset.previewLims2]
        for snippet_id, lims in enumerate(lims_snippets):
            for coord in self.dataset.snippet_coords["snippet_id{}".format(snippet_id)]:
                dict_ = {"snippet_id": snippet_id,
                        "coords": coord, # 10,1 alpha
                        "reference": self.label_mask[lims[0]:lims[1], lims[2]:lims[3]][coord[0], coord[1]]}
                
                predicted_coord = []
                for idx in range(self.prob_rec.shape[0]):
                    predicted_coord.append(self.prob_rec[idx][lims[0]:lims[1], lims[2]:lims[3]][coord[0], coord[1]])
                predicted_coord = np.array(predicted_coord)
                dict_["predicted"] = predicted_coord

                self.snippet_poi_results.append(dict_)

        return self.snippet_poi_results