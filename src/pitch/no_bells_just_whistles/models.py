from typing import Dict, List, Tuple
import yaml
import torch
import torchvision.transforms.functional as f
import torchvision.transforms as transforms
from PIL import Image
import cv2
import torch.nn.functional as F 
import copy
from scipy.stats import linregress

from .cls_hrnet import HighResolutionNet
from .cls_hrnet_l import HighResolutionNet as HighResolutionNetLine

class NoBellsJustWhistles:
    def __init__(self, config: Dict, device = "cpu", **kwargs):
            
        self.device = device
        
        #  Initialize dicts
        self.kp_dicts = self.init_dicts()

        # Initialize models
        self.model = self.initiate_kp_model(config, **kwargs)
        self.model_l = self.initiate_lines_model(config, **kwargs)

    def initiate_kp_model(self, config, **kwargs):

        self.kp_threshold = config['kp_threshold']
        
        model_config =yaml.safe_load(open(config['config_kp_path'], 'r'))
        model = HighResolutionNet(model_config, **kwargs)
        loaded_state = torch.load(config['model_kp_path'], map_location=self.device)
        model.load_state_dict(loaded_state)
        model.to(self.device)
        model.eval()
        return model
    
    def initiate_lines_model(self, config, **kwargs):

        self.lines_threshold = config['lines_threshold']

        model_config = yaml.safe_load(open(config['config_lines_path'], 'r'))
        model = HighResolutionNetLine(model_config, **kwargs)
        loaded_state = torch.load(config['model_lines_path'], map_location=self.device)
        model.load_state_dict(loaded_state)
        model.to(self.device)
        model.eval()
        return model
    
    def init_dicts(self):
        """ Load keypoints and lines dictionaries from YAML file"""
        kp_dicts = yaml.safe_load(open('src/pitch/no_bells_just_whistles/keypoints.yaml', 'r'))
        return kp_dicts

    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        frame = f.to_tensor(frame).float().unsqueeze(0)
        _, _, h_original, w_original = frame.size()
            
        # Ensure correct size
        if frame.size()[-1] != 960:
            frame = self.transform_frame(frame)
            
        return frame.to(self.device)

    def transform_frame(self, frame):
        """Transform frame to required size"""
        return F.resize(frame, [480, 960])

    def predict_heatmaps(self, frame):
        """Get heatmap predictions from both models"""
        with torch.no_grad():
            kp_heatmaps = self.model(frame)
            lines_heatmaps = self.model_l(frame)
        return kp_heatmaps, lines_heatmaps

    def process_keypoints(self, kp_heatmaps, lines_heatmaps, kp_threshold, lines_threshold):
        """Process heatmaps to get keypoints"""
        kp_coords = self.get_keypoints_from_heatmap_batch_maxpool(kp_heatmaps[:,:-1,:,:])
        lines_coords = self.get_keypoints_from_heatmap_batch_maxpool_l(lines_heatmaps[:,:-1,:,:])
        
        kp_dict = self.coords_to_dict(kp_coords, threshold=kp_threshold)
        lines_dict = self.coords_to_dict(lines_coords, threshold=lines_threshold)
        
        return self.complete_keypoints(kp_dict, lines_dict, w= self.last_w, h= self.last_h)

    def get_keypoints_from_heatmap_batch_maxpool(self, heatmap: torch.Tensor, scale: int = 2,
                                                max_keypoints: int = 1, min_keypoint_pixel_distance: int = 15,
                                                return_scores: bool = True,
                                                ) -> torch.Tensor:
        """Fast extraction of keypoints from a batch of heatmaps using maxpooling.

        Inspired by mmdetection and CenterNet:
        https://mmdetection.readthedocs.io/en/v2.13.0/_modules/mmdet/models/utils/gaussian_target.html

        Args:
            heatmap (torch.Tensor): NxCxHxW heatmap batch
            max_keypoints (int, optional): max number of keypoints to extract, lowering will result in faster execution times. Defaults to 20.
            min_keypoint_pixel_distance (int, optional): _description_. Defaults to 1.

            Following thresholds can be used at inference time to select where you want to be on the AP curve. They should ofc. not be used for training
            abs_max_threshold (Optional[float], optional): _description_. Defaults to None.
            rel_max_threshold (Optional[float], optional): _description_. Defaults to None.

        Returns:
            The extracted keypoints for each batch, channel and heatmap; and their scores
        """
        batch_size, n_channels, _, width = heatmap.shape

        # obtain max_keypoints local maxima for each channel (w/ maxpool)

        kernel = min_keypoint_pixel_distance * 2 + 1
        pad = min_keypoint_pixel_distance
        # exclude border keypoints by padding with highest possible value
        # bc the borders are more susceptible to noise and could result in false positives
        padded_heatmap = torch.nn.functional.pad(heatmap, (pad, pad, pad, pad), mode="constant", value=1.0)
        max_pooled_heatmap = torch.nn.functional.max_pool2d(padded_heatmap, kernel, stride=1, padding=0)
        # if the value equals the original value, it is the local maximum
        local_maxima = max_pooled_heatmap == heatmap
        # all values to zero that are not local maxima
        heatmap = heatmap * local_maxima

        # extract top-k from heatmap (may include non-local maxima if there are less peaks than max_keypoints)
        scores, indices = torch.topk(heatmap.view(batch_size, n_channels, -1), max_keypoints, sorted=True)
        indices = torch.stack([torch.div(indices, width, rounding_mode="floor"), indices % width], dim=-1)
        # at this point either score > 0.0, in which case the index is a local maximum
        # or score is 0.0, in which case topk returned non-maxima, which will be filtered out later.

        #  remove top-k that are not local maxima and threshold (if required)
        # thresholding shouldn't be done during training

        #  moving them to CPU now to avoid multiple GPU-mem accesses!
        indices = indices.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        filtered_indices = [[[] for _ in range(n_channels)] for _ in range(batch_size)]
        filtered_scores = [[[] for _ in range(n_channels)] for _ in range(batch_size)]

        # have to do this manually as the number of maxima for each channel can be different
        for batch_idx in range(batch_size):
            for channel_idx in range(n_channels):
                candidates = indices[batch_idx, channel_idx]
                locs = []
                for candidate_idx in range(candidates.shape[0]):
                    # convert to (u,v)
                    loc = candidates[candidate_idx][::-1] * scale
                    loc = loc.tolist()
                    if return_scores:
                        loc.append(scores[batch_idx, channel_idx, candidate_idx])
                    locs.append(loc)
                filtered_indices[batch_idx][channel_idx] = locs

        return torch.tensor(filtered_indices)


    def get_keypoints_from_heatmap_batch_maxpool_l(self, heatmap: torch.Tensor, scale: int = 2,
                                                  max_keypoints: int = 2, min_keypoint_pixel_distance: int = 10,
                                                    return_scores: bool = True,
                                                ) -> torch.Tensor:
        """Fast extraction of keypoints from a batch of heatmaps using maxpooling.

        Inspired by mmdetection and CenterNet:
        https://mmdetection.readthedocs.io/en/v2.13.0/_modules/mmdet/models/utils/gaussian_target.html

        Args:
            heatmap (torch.Tensor): NxCxHxW heatmap batch
            max_keypoints (int, optional): max number of keypoints to extract, lowering will result in faster execution times. Defaults to 20.
            min_keypoint_pixel_distance (int, optional): _description_. Defaults to 1.

            Following thresholds can be used at inference time to select where you want to be on the AP curve. They should ofc. not be used for training
            abs_max_threshold (Optional[float], optional): _description_. Defaults to None.
            rel_max_threshold (Optional[float], optional): _description_. Defaults to None.

        Returns:
            The extracted keypoints for each batch, channel and heatmap; and their scores
        """
        batch_size, n_channels, _, width = heatmap.shape
        kernel = min_keypoint_pixel_distance * 2 + 1
        pad = int((kernel-1)/2)

        max_pooled_heatmap = torch.nn.functional.max_pool2d(heatmap, kernel, stride=1, padding=pad)
        # if the value equals the original value, it is the local maximum
        local_maxima = max_pooled_heatmap == heatmap

        # all values to zero that are not local maxima
        heatmap = heatmap * local_maxima

        # extract top-k from heatmap (may include non-local maxima if there are less peaks than max_keypoints)
        scores, indices = torch.topk(heatmap.view(batch_size, n_channels, -1), max_keypoints, sorted=True)
        indices = torch.stack([torch.div(indices, width, rounding_mode="floor"), indices % width], dim=-1)
        # at this point either score > 0.0, in which case the index is a local maximum
        # or score is 0.0, in which case topk returned non-maxima, which will be filtered out later.

        #  remove top-k that are not local maxima and threshold (if required)
        # thresholding shouldn't be done during training

        #  moving them to CPU now to avoid multiple GPU-mem accesses!
        indices = indices.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        filtered_indices = [[[] for _ in range(n_channels)] for _ in range(batch_size)]
        filtered_scores = [[[] for _ in range(n_channels)] for _ in range(batch_size)]

        # have to do this manually as the number of maxima for each channel can be different
        for batch_idx in range(batch_size):
            for channel_idx in range(n_channels):
                candidates = indices[batch_idx, channel_idx]
                locs = []
                for candidate_idx in range(candidates.shape[0]):
                    # convert to (u,v)
                    loc = candidates[candidate_idx][::-1] * scale
                    loc = loc.tolist()
                    if return_scores:
                        loc.append(scores[batch_idx, channel_idx, candidate_idx])
                    locs.append(loc)
                filtered_indices[batch_idx][channel_idx] = locs

        return torch.tensor(filtered_indices)


    def coords_to_dict(self, coords, threshold=0.05, ground_plane_only=False):
        kp_list = []
        for batch in range(coords.size()[0]):
            keypoints = {}
            for count, c in enumerate(range(coords.size(1))):
                if coords.size(2) == 1:
                    if ground_plane_only and c+1 in [12,15,16,19]:
                        continue
                    if coords[batch, c, 0, -1] > threshold:
                        keypoints[count+1] = {
                            'x': coords[batch, c, 0, 0].item(),
                            'y': coords[batch, c, 0, 1].item(),
                            'p': coords[batch, c, 0, 2].item()
                        }
                else:
                    if ground_plane_only and c+1 in [7,8,9,10,11,12]:
                        continue
                    if coords[batch, c, 0, -1] > threshold and coords[batch, c, 1, -1] > threshold:
                        keypoints[count+1] = {
                            'x_1': coords[batch, c, 0, 0].item(),
                            'y_1': coords[batch, c, 0, 1].item(),
                            'p_1': coords[batch, c, 0, 2].item(),
                            'x_2': coords[batch, c, 1, 0].item(),
                            'y_2': coords[batch, c, 1, 1].item(),
                            'p_2': coords[batch, c, 1, 2].item()
                        }
            kp_list.append(keypoints)
        return kp_list

    def complete_keypoints(self, kp_dict, lines_dict, w, h, normalize=False):
        def lines_intersection(x1, y1, x2, y2):
            x1[-1] += 1e-7  # Small offset to handle identical coordinates
            x2[-1] += 1e-7
            slope1, intercept1, r1, p1, se1 = linregress(x1, y1)
            slope2, intercept2, r2, p2, se2 = linregress(x2, y2)

            x_intersection = (intercept2 - intercept1) / (slope1 - slope2 + 1e-7)
            y_intersection = slope1 * x_intersection + intercept1

            return x_intersection, y_intersection

        lines_list = self.kp_dicts['lines_list']
        keypoints_lines_list = self.kp_dicts['keypoints_lines_list']
        keypoint_aux_pair_list = self.kp_dicts['keypoint_aux_pair_list']

        w_extra = 0.5 * w
        h_extra = 0.5 * h

        complete_list = []
        for batch in range(len(kp_dict)):
            complete_dict = copy.deepcopy(kp_dict[batch])
            
            # Process main keypoints
            for key in range(1, 31):
                if key not in kp_dict[batch].keys():
                    lines_keys = keypoints_lines_list[key-1]
                    lines_key1 = lines_list.index(lines_keys[0]) + 1
                    lines_key2 = lines_list.index(lines_keys[1]) + 1
                    
                    if all(lines_key in lines_dict[batch].keys() for lines_key in [lines_key1, lines_key2]):
                        x1 = [lines_dict[batch][lines_key1]['x_1'], lines_dict[batch][lines_key1]['x_2']]
                        y1 = [lines_dict[batch][lines_key1]['y_1'], lines_dict[batch][lines_key1]['y_2']]
                        x2 = [lines_dict[batch][lines_key2]['x_1'], lines_dict[batch][lines_key2]['x_2']]
                        y2 = [lines_dict[batch][lines_key2]['y_1'], lines_dict[batch][lines_key2]['y_2']]
                        
                        new_kp = lines_intersection(x1, y1, x2, y2)
                        if -w_extra < new_kp[0] < w_extra + w and -h_extra < new_kp[1] < h_extra + h:
                            complete_dict[key] = {
                                'x': round(new_kp[0], 0),
                                'y': round(new_kp[1], 0),
                                'p': 1.0
                            }

            # Process auxiliary keypoints
            for key in range(1, len(keypoint_aux_pair_list)):
                lines_keys = keypoint_aux_pair_list[key-1]
                lines_key1 = lines_list.index(lines_keys[0]) + 1
                lines_key2 = lines_list.index(lines_keys[1]) + 1
                
                if all(lines_key in lines_dict[batch].keys() for lines_key in [lines_key1, lines_key2]):
                    x1 = [lines_dict[batch][lines_key1]['x_1'], lines_dict[batch][lines_key1]['x_2']]
                    y1 = [lines_dict[batch][lines_key1]['y_1'], lines_dict[batch][lines_key1]['y_2']]
                    x2 = [lines_dict[batch][lines_key2]['x_1'], lines_dict[batch][lines_key2]['x_2']]
                    y2 = [lines_dict[batch][lines_key2]['y_1'], lines_dict[batch][lines_key2]['y_2']]
                    
                    new_kp = lines_intersection(x1, y1, x2, y2)
                    if -w_extra < new_kp[0] < w_extra + w and -h_extra < new_kp[1] < h_extra + h:
                        complete_dict[key+57] = {
                            'x': round(new_kp[0], 0),
                            'y': round(new_kp[1], 0),
                            'p': 1.0
                        }

            # Normalize coordinates if requested
            if normalize:
                for kp in complete_dict.keys():
                    complete_dict[kp]['x'] /= w
                    complete_dict[kp]['y'] /= h

            complete_dict = dict(sorted(complete_dict.items()))
            complete_list.append(complete_dict)

        return complete_list
    
    def inference(self, frame, kp_threshold = None, lines_threshold = None):
        """Main inference pipeline"""

        # Set thresholds if not provided
        kp_threshold = kp_threshold or self.kp_threshold
        lines_threshold = lines_threshold or self.lines_threshold
        
        # Preprocess
        processed_frame = self.preprocess_frame(frame)
        self.last_frame_size = processed_frame.size()

        self.last_w =  processed_frame.size(3)
        self.last_h =  processed_frame.size(2)
        
        # Get predictions
        kp_heatmaps, lines_heatmaps = self.predict_heatmaps(processed_frame)
        
        
        # Process keypoints
        final_dict = self.process_keypoints(kp_heatmaps, lines_heatmaps, 
                                          kp_threshold, lines_threshold)
        
        return final_dict
    
        # Update camera and get final parameters
        # cam.update(final_dict[0])
        # return cam.heuristic_voting()