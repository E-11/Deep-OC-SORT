import torch
import numpy as np
import motmetrics as mm
from .association import *
from .embedding import EmbeddingComputer
from .cmc import CMCComputer
from external.ArTIST.utils.clustering import load_clusters
from external.ArTIST.utils.utils_ar import infer_log_likelihood
from external.ArTIST.models.ar import motion_ar
from external.ArTIST.models.ae import motion_ae
from utils import xyxy2ltwh, ltwh2xyxy

class ArTISTTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, bbox, orig=False, emb=None, alpha=0, start=0):
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model

        self.emb = emb

        # ArTIST Tracker
        self.id = ArTISTTracker.count
        ArTISTTracker.count += 1
        self.alive = True
        self.sequence = bbox  ## (1, track_len, 4)
        self.social = None    ## (1, track_len, 256)
        self.ae_hs = None       ## hidden state in ae from last frame: (1, 1, 256)
        self.gap = 0
        self.start = start
        self.end = start + 1

    def update(self, bbox, end, inpainted_info):
        """
        Updates embedding and social feature with observed bbox.
        bbox: (1, seq_len, 4)
        """
        if bbox is not None:
            if inpainted_info is not None:  ## for inpainted tracklet
                self.gap = 0
                self.sequence = torch.cat((self.sequence, inpainted_info[0]), 1)
                self.social = torch.cat((self.social, inpainted_info[1]), 1)
                self.ae_hs = inpainted_info[2]

            self.sequence = torch.cat((self.sequence, bbox), 1)
            self.end = end
        else:
            self.gap += 1

    def update_emb(self, emb, alpha=0.9):
        self.emb = alpha * self.emb + (1 - alpha) * emb
        self.emb /= np.linalg.norm(self.emb)

    def get_emb(self):
        return self.emb

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.x_to_bbox_func(self.kf.x)

    def mahalanobis(self, bbox):
        """Should be run after a predict() call for accuracy."""
        return self.kf.md_for_measurement(self.bbox_to_z_func(bbox))


"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note 
    that we hardly normalize the cost by all methods to (0,1) which may not be 
    the best practice.
"""
ASSO_FUNCS = {
    "iou": iou_batch,
    "giou": giou_batch,
    "ciou": ciou_batch,
    "diou": diou_batch,
    "ct_dist": ct_dist,
}


class ArTIST(object):
    def __init__(
        self,
        det_thresh,
        max_age=30,
        asso_func="iou",
        inertia=0.2,
        w_association_emb=0.75,
        alpha_fixed_emb=0.95,
        aw_param=0.5,
        embedding_off=False,
        cmc_off=False,
        aw_off=False,
        new_kf_off=False,
        grid_off=False,
        n_sampling = 50,
        num_cluster = 1024,
        t_trs = 2,
        iou_thre_trs = 0.5,
        hidden_state = 256,
        init_frame_num = 3, 
        **kwargs,
    ):
        """
        Sets key parameters for SORT
        """
        self.trackers = []  ## list of tracklets
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.w_association_emb = w_association_emb
        self.alpha_fixed_emb = alpha_fixed_emb
        self.aw_param = aw_param
        ArTISTTracker.count = 0

        self.embedder = EmbeddingComputer(kwargs["args"].dataset, kwargs["args"].test_dataset, grid_off)
        self.cmc = CMCComputer()
        self.embedding_off = embedding_off
        self.cmc_off = cmc_off
        self.aw_off = aw_off
        self.grid_off = grid_off

        # ArTIST
        self.model_ae = motion_ae(256).cuda()  ## moving agent autoencoder network
        self.model_ae.load_state_dict(torch.load('/home/share/model/DVAE-UMOT/ArTIST/ae/ae_8.pth'))
        self.model_ae.eval()
        self.model_ar = motion_ar(512, 1024).cuda()  ## ArTIST Model
        self.model_ar.load_state_dict(torch.load('/home/share/model/DVAE-UMOT/ArTIST/ar/ar_110.pth'))
        self.model_ar.eval()
        self.n_sampling = n_sampling
        centroid_x, centroid_y, centroid_w, centroid_h = load_clusters()
        self.centroids = [centroid_x, centroid_y, centroid_w, centroid_h]
        self.num_cluster = num_cluster
        self.inpainted_social = torch.zeros(1, 1, hidden_state).cuda()  ## (1, frame_count, 256)
        self.tmp_inpaint_seq_dict = {}  ## temporary inpainted sequence for TRS
        self.t_trs = t_trs
        self.iou_thre_trs = iou_thre_trs
        self.det_last2_frame_list = [None]  ## detections in prior 2 frames, 2×(num_det, 4)
        self.max_age = max_age
        self.init_frame_num = init_frame_num

    def update(self, output_results, img_tensor, img_numpy, tag):
        """
        Params:
          output_results - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        if output_results is None:
            return np.empty((0, 5))
        if not isinstance(output_results, np.ndarray):
            output_results = output_results.cpu().numpy()
        self.frame_count += 1
        if output_results.shape[1] == 5:
            det_scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results
            det_scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        dets = np.concatenate((bboxes, np.expand_dims(det_scores, axis=-1)), axis=1)
        remain_inds = det_scores > self.det_thresh
        dets = dets[remain_inds]

        # Rescale
        scale = min(img_tensor.shape[2] / img_numpy.shape[0], img_tensor.shape[3] / img_numpy.shape[1])
        dets[:, :4] /= scale

        # Generate embeddings
        self.dets_embs = np.ones((dets.shape[0], 1))
        if not self.embedding_off and dets.shape[0] != 0:
            # Shape = (num detections, 3, 512) if grid
            self.dets_embs = self.embedder.compute_embedding(img_numpy, dets[:, :4], tag)

        # CMC
        if not self.cmc_off:
            transform = self.cmc.compute_affine(img_numpy, dets[:, :4], tag)
            for trk in self.trackers:
                trk.apply_affine_correction(transform)

        # dets_alpha
        trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)  ## ∈ (0,1)
        af = self.alpha_fixed_emb
        # From [self.alpha_fixed_emb, 1], goes to 1 as detector is less confident
        self.dets_alpha = af + (1 - af) * (1 - trust)

        # tracking in current frame with motion feature
        detection_current_frame = torch.tensor(xyxy2ltwh(dets[:, :4])).cuda()
        detection_current_frame = detection_current_frame[~torch.any(detection_current_frame.isnan(),dim=1)]
        if self.frame_count <= self.init_frame_num:
            assignment_fullseq, assignment_inpaint, inpaint_seq_list, uT_fullseq, uT_inpaint, not_assign_seq_idx, uD_inpaint = self.track_first_n_frame(detection_current_frame)
        else:
            assignment_fullseq, assignment_inpaint, inpaint_seq_list, uT_fullseq, uT_inpaint, not_assign_seq_idx, uD_inpaint = self.tracking_framet(detection_current_frame)

        # post-process
        self.post_process(detection_current_frame,
                          assignment_fullseq, 
                          assignment_inpaint, 
                          inpaint_seq_list, 
                          uT_fullseq, 
                          uT_inpaint, 
                          not_assign_seq_idx, 
                          uD_inpaint)
        
        ret = []
        for trk in self.trackers:
            d = ltwh2xyxy(trk.sequence[0, -1:, :].cpu().detach().numpy()).squeeze()
            if trk.gap == 0:
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
    
    def track_first_n_frame(self, detection_current_frame):
        # tracking based on iou
        detection_current_frame = detection_current_frame[~torch.any(detection_current_frame.isnan(),dim=1)]
        det_bbox = np.asfarray(detection_current_frame.cpu().detach().numpy())
        num_det = detection_current_frame.shape[0]
        num_tracklet = len(self.trackers)

        track_bbox = []
        cost_matrix = torch.zeros(num_tracklet, num_det).cuda()
        full_seq_idx = []
        not_assign_seq_idx = []

        for tidx in range(num_tracklet):
            track_bbox = self.trackers[tidx].sequence[:, -1, :]
            track_bbox = np.asfarray(track_bbox.cpu().detach().numpy())
            iou = mm.distances.boxiou(track_bbox[:, None], det_bbox[None, :])
            iou = torch.from_numpy(iou).cuda()
            if torch.sum(iou>self.iou_thre_trs) == 0:
                not_assign_seq_idx.append(tidx)
                cost_matrix[tidx, :] = -1000 * torch.ones(num_det).cuda()
            else:
                full_seq_idx.append(tidx)
                cost_matrix[tidx:tidx+1, :] = iou
            
            if self.frame_count == self.init_frame_num:
                print('\n\n', tidx)
                print(self.trackers[tidx].sequence)

        detection_idx = np.arange(num_det)
        # Assignment for full sequences
        assignment_fullseq, uD_fullseq, uT_fullseq = Munkres(cost_matrix, detection_idx, full_seq_idx)
        inpaint_seq_idx = []
        inpaint_seq_list = []
        assignment_inpaint, uD_inpaint, uT_inpaint = [], uD_fullseq, inpaint_seq_idx
        return assignment_fullseq, assignment_inpaint, inpaint_seq_list, uT_fullseq, uT_inpaint, not_assign_seq_idx, uD_inpaint    

    def tracking_framet(self, detection_current_frame):
        # detection_current_frame: (num_det, 4), xywh
        num_det = detection_current_frame.shape[0]
        num_tracklet = len(self.trackers)

        cost_matrix = torch.zeros(num_tracklet, num_det).cuda()
        full_seq_idx = []
        inpaint_seq_idx = []
        not_assign_seq_idx = []  ## sequences not join in assignment
        inpaint_seq_list = []

        gaussian_kernel = np.load("external/ArTIST/data/kernel.npy")
        gaussian_kernel = torch.autograd.Variable(torch.from_numpy(gaussian_kernel).float()).cuda()
        gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0)

        for tidx in range(num_tracklet):
            sequence = self.trackers[tidx].sequence  ## sequence: (1, track_len, 4)
            gap = self.trackers[tidx].gap
            start = self.trackers[tidx].start
            end = self.trackers[tidx].end
            social = self.trackers[tidx].social  ## (1, track_len, 256)

            # computing the motion velocity
            tracklet_delta = torch.zeros(sequence.shape).cuda()
            if sequence.shape[1] > 1:
                tracklet_delta_tmp = sequence[:, 1:, :] - sequence[:, :-1, :]
                tracklet_delta[:, 1:, :] = tracklet_delta_tmp  ## (1, track_len, 4)

            # Inpainting and compute the cost matrix
            if gap == 0:
                y_g1d_x, y_g1d_y, y_g1d_w, y_g1d_h, sampled_boxes, sampled_deltas, sampled_detection = self.generate_distrib(self.trackers[tidx], gaussian_kernel)

                distribs = [y_g1d_x, y_g1d_y, y_g1d_w, y_g1d_h]
                last_bbox = sequence[0, -1, :]
                scores = self.scoring(last_bbox, tracklet_delta, sampled_deltas, detection_current_frame, distribs)
                inpaint_seq_list.append(sampled_boxes)  ## gap = 0, sampled_bboxes is []
                # if torch.sum(scores>-50) == 0:
                #     not_assign_seq_idx.append(tidx)
                # else:
                #     full_seq_idx.append(tidx)
                full_seq_idx.append(tidx)
                
                # if self.frame_count == 5:
                #     print('\n\n', tidx)
                #     print(distribs)
                #     print(scores)
                #     print(sequence)

            elif gap >= self.t_trs:  ## tracklet inpainting
                with torch.no_grad():
                    dist_x, dist_y, dist_w, dist_h, sampled_boxes, sampled_deltas, sampled_detection = self.model_ar.batch_inference(
                        sequence.repeat(self.n_sampling, 1, 1), 
                        social.repeat(self.n_sampling, 1, 1), 
                        gap + 1, 
                        self.centroids, 
                        self.inpainted_social[:, end:, :].repeat(self.n_sampling, 1, 1))

                det_last3_frame_list = self.det_last2_frame_list + [detection_current_frame]
                best_inpainting, best_detection, best_deltas, best_dist_x, best_dist_y, best_dist_w, best_dist_h = self.trs(
                    sampled_boxes,       ## (n_sampling, gap, 4)
                    sampled_detection,   ## (n_sampling, 1, 4)
                    sampled_deltas,      ## (n_sampling, gap, 4)
                    dist_x, dist_y, dist_w, dist_h,  ## (n_sampling, gap+1, 1024)
                    det_last3_frame_list)

                if len(best_inpainting) != 0:
                    # making it a probability distribution
                    dist_x = torch.nn.Softmax(dim=-1)(best_dist_x).permute(1, 0, 2)
                    dist_y = torch.nn.Softmax(dim=-1)(best_dist_y).permute(1, 0, 2)
                    dist_w = torch.nn.Softmax(dim=-1)(best_dist_w).permute(1, 0, 2)
                    dist_h = torch.nn.Softmax(dim=-1)(best_dist_h).permute(1, 0, 2)

                    # smoothing the distributions using a gaussian kernel
                    y_g1d_x = torch.nn.functional.conv1d(dist_x, gaussian_kernel,
                                                        padding=24).permute(1, 0, 2)
                    y_g1d_y = torch.nn.functional.conv1d(dist_y, gaussian_kernel,
                                                        padding=24).permute(1, 0, 2)
                    y_g1d_w = torch.nn.functional.conv1d(dist_w, gaussian_kernel,
                                                        padding=24).permute(1, 0, 2)
                    y_g1d_h = torch.nn.functional.conv1d(dist_h, gaussian_kernel,
                                                        padding=24).permute(1, 0, 2)
                    distribs = [y_g1d_x, y_g1d_y, y_g1d_w, y_g1d_h]
                    last_bbox = best_inpainting[0, -1, :]  ## not predicted tracking bbox in current frame
                    scores = self.scoring(last_bbox, tracklet_delta, best_deltas, detection_current_frame, distribs)
                    # if torch.sum(scores>-50) == 0:
                    #     not_assign_seq_idx.append(tidx)
                    # else:
                    #     inpaint_seq_idx.append(tidx)
                    inpaint_seq_idx.append(tidx)

                else:
                    not_assign_seq_idx.append(tidx)
                    scores = -1000 * torch.ones(num_det).cuda()
                inpaint_seq_list.append(best_inpainting)
            else:
                # 0 < gap < 2
                not_assign_seq_idx.append(tidx)
                scores = -1000 * torch.ones(num_det).cuda()
                inpaint_seq_list.append([])

            cost_matrix[tidx, :] = scores

        detection_idx = np.arange(num_det)
        # Assignment for full sequences
        assignment_fullseq, uD_fullseq, uT_fullseq = Munkres(cost_matrix, detection_idx, full_seq_idx)

        # Assignment for inpainted sequences
        assignment_inpaint, uD_inpaint, uT_inpaint = [], uD_fullseq, inpaint_seq_idx
        if (uD_fullseq.size != 0) & (np.sum([len(j) for j in inpaint_seq_list]) != 0):
            assignment_inpaint, uD_inpaint, uT_inpaint = Munkres(cost_matrix, uD_fullseq, inpaint_seq_idx)
        
        return assignment_fullseq, assignment_inpaint, inpaint_seq_list, uT_fullseq, uT_inpaint, not_assign_seq_idx, uD_inpaint

    def post_process(self, 
                     detection_current_frame, 
                     assignment_fullseq, 
                     assignment_inpaint,
                     inpaint_seq_list, 
                     uT_fullseq, 
                     uT_inpaint, 
                     not_assign_seq_idx, 
                     uD_inpaint):

        # Update existing tracklets:
        tid_current_frame = []  ## tracklet idx in current frame

        # For assigned fullseq tracklets
        if len(assignment_fullseq) != 0:
            for ass in assignment_fullseq:
                tracklet_idx = ass[0]
                det_idx = ass[1]
                detection = detection_current_frame[det_idx].unsqueeze(0).unsqueeze(0)  ## (1, 1, 4)
                self.trackers[tracklet_idx].update(detection, self.frame_count + 1, None)
                tid_current_frame.append(tracklet_idx)
                
        # For assigned inpainted tracklets
        if len(assignment_inpaint) != 0:
            # generate ae_hs for each inpainted tracklet in frame gap
            for ass in assignment_inpaint:
                tracklet_idx = ass[0]
                det_idx = ass[1]
                tracklet_end = self.trackers[tracklet_idx].end
                detection = detection_current_frame[det_idx].unsqueeze(0).unsqueeze(0)
                inpainted_seq = inpaint_seq_list[ass[0]]
                inpainted_social = self.inpainted_social[:, tracklet_end:, :]

                inpainted_vel = inpainted_seq[:, 1:, :] - inpainted_seq[:, :-1, :]  ## inpaint tracklet till last frame
                inpainted_vel_0 = inpainted_seq[:, 0:1, :] - self.trackers[tracklet_idx].sequence[:, -1:, :]
                inpainted_vel = torch.cat((inpainted_vel_0, inpainted_vel), 1)  ## (1, gap, 4)
                hs = self.trackers[tracklet_idx].ae_hs
                inpainted_hs = self.model_ae.inference(inpainted_vel, hs)[:, -1:, :]  ## (1, 1, 256)
                
                inpainted_info = [inpainted_seq, inpainted_social, inpainted_hs]
                self.trackers[tracklet_idx].update(detection, self.frame_count + 1, inpainted_info)
                tid_current_frame.append(tracklet_idx)

        # For unassigned tracklets and not joint assignment tracklets
        for uT_idx in uT_fullseq:
            self.trackers[uT_idx].update(None, None, None)
        
        # For unassigned inpainted tracklets
        dead_track_idx = []
        if len(uT_inpaint) != 0:
            for unassigned_idx in list(uT_inpaint) + list(not_assign_seq_idx):
                self.trackers[unassigned_idx].update(None, None, None)
                # remove dead tracklets
                if self.trackers[unassigned_idx].gap >= self.max_age:
                    dead_track_idx.append(unassigned_idx)

        # update self.det_last2_frame_list, max_len=2
        if len(self.det_last2_frame_list) == 1:
            self.det_last2_frame_list.append(detection_current_frame)
        else:
            self.det_last2_frame_list[0] = self.det_last2_frame_list[1]
            self.det_last2_frame_list[1] = detection_current_frame

        # Generate and initialize new tracklets for unmatched detections
        if len(uD_inpaint) != 0:
            for uD_idx in uD_inpaint:
                trk = ArTISTTracker(
                    detection_current_frame[uD_idx].unsqueeze(0).unsqueeze(0), 
                    emb=self.dets_embs[uD_idx], 
                    alpha=self.dets_alpha[uD_idx], 
                    start=self.frame_count
                )
                self.trackers.append(trk)
                tid_current_frame.append(len(self.trackers)-1)

        # update hidden state and social feature for assigned tracklets in current frame
        self.update_social_hs(tid_current_frame)

        # remove dead tracklets
        for tidx in dead_track_idx:
            self.trackers.pop(tidx)

    def update_social_hs(self, tid_current_frame):
        # computing I_j with T_j excluded

        hs_current_frame = []
        # for exiting tracklets
        for tidx in tid_current_frame:
            tid_sequence = self.trackers[tidx].sequence
            if len(tid_sequence) == 1:
                social_vel = torch.zeros(1, 1, 4)
            else:
                social_vel = tid_sequence[:, -1:, :] - tid_sequence[:, -2:-1, :]  ## (1, 1, 4)
            hs = self.trackers[tidx].ae_hs
            with torch.no_grad():
                social_vel = social_vel.float().cuda()
                hs_current_frame.append(self.model_ae.inference_per_frame(social_vel, hs))  ## (1, 1, 256)
        hs_current_frame = torch.cat(hs_current_frame, 0)  ## (track_num, 1, 256)
        
        # get social feature of tracklets in current frame
        for j, tidx in enumerate(tid_current_frame):  ## for T_j
            hs_wo_tid = torch.cat((hs_current_frame[:j, :, :], hs_current_frame[j+1:, :, :]), dim=0)  ## hidden state without tid: (track_num-1, 1, 256)
            I_j = torch.max(hs_wo_tid, dim=0)[0].unsqueeze(0)  ## (1, 1, 256)
            if self.trackers[tidx].social is None:
                self.trackers[tidx].social = I_j
            else:
                self.trackers[tidx].social = torch.cat((self.trackers[tidx].social, I_j), 1)
            self.trackers[tidx].ae_hs = hs_current_frame[j:j+1, :, :]  ## (1, 1, 256)
        
        # social in current frame for inpainted tracklet
        inpaint_s_current_frame = torch.max(hs_current_frame, dim=0)[0].unsqueeze(0)  ## (1, 1, 256)
        self.inpainted_social = torch.cat((self.inpainted_social, inpaint_s_current_frame), 1)

    def generate_distrib(self, tracker, gaussian_kernel):

        # sequence: (1, track_len, 4)
        # social: (1, track_len, 256)

        ## computing the distribution over the next plausible bounding box
        # dist_*: equation(3) in paper
        # sampled_boxes: inpainting bboxes
        # sampled_deltas: inpainting deltas
        # sampled_detection: predicted tracking bbox
        sequence = tracker.sequence
        social = tracker.social
        gap = tracker.gap
        dist_x, dist_y, dist_w, dist_h, sampled_boxes, sampled_deltas, sampled_detection = self.model_ar.inference(
            sequence,
            social,
            gap,
            self.centroids,
            self.inpainted_social[:, tracker.end:, :])

        # making it a probability distribution
        dist_x = torch.nn.Softmax(dim=-1)(dist_x).permute(1, 0, 2)
        dist_y = torch.nn.Softmax(dim=-1)(dist_y).permute(1, 0, 2)
        dist_w = torch.nn.Softmax(dim=-1)(dist_w).permute(1, 0, 2)
        dist_h = torch.nn.Softmax(dim=-1)(dist_h).permute(1, 0, 2)

        # smoothing the distributions using a gaussian kernel
        y_g1d_x = torch.nn.functional.conv1d(dist_x, gaussian_kernel,
                                            padding=24).permute(1,0,2)
        y_g1d_y = torch.nn.functional.conv1d(dist_y, gaussian_kernel,
                                            padding=24).permute(1,0,2)
        y_g1d_w = torch.nn.functional.conv1d(dist_w, gaussian_kernel,
                                            padding=24).permute(1,0,2)
        y_g1d_h = torch.nn.functional.conv1d(dist_h, gaussian_kernel,
                                            padding=24).permute(1,0,2)

        return y_g1d_x, y_g1d_y, y_g1d_w, y_g1d_h, sampled_boxes, sampled_deltas, sampled_detection

    def scoring(self, 
                last_bbox, 
                tracklet_delta, 
                sampled_delta, 
                true_detections, 
                distribs):
        # tracklet_delta: (1, track_len, 4)
        # true_dections: (num_det, 4)
        # sampled_delta: (1, gap, 4)
        # last_bbox: (4,)

        # scores: (num_det)  ## for each tracklet

        num_det = true_detections.shape[0]
        extended_track = torch.zeros(1, tracklet_delta.shape[1] + sampled_delta.shape[1] + 1, 4).cuda()
        extended_track[0, :tracklet_delta.shape[1], :] = tracklet_delta[0, :, :]
        extended_track[0, tracklet_delta.shape[1]:-1, :] = sampled_delta

        scores = torch.zeros(num_det).cuda()
        for i in range(num_det):
            last_delta = true_detections[i, :] - last_bbox.detach()
            extended_track[0, -1, :] = last_delta
            likelihoods_smooth = infer_log_likelihood(distribs[0], distribs[1], distribs[2], distribs[3],
                                                    extended_track[:, 1:, 0:1],
                                                    extended_track[:, 1:, 1:2],
                                                    extended_track[:, 1:, 2:3],
                                                    extended_track[:, 1:, 3:4],
                                                    self.centroids[0], self.centroids[1], self.centroids[2], self.centroids[3], self.num_cluster)
            all_scores = np.array(likelihoods_smooth[-1])
            likelihoods_smooth = np.sum(all_scores)
            score = torch.tensor(likelihoods_smooth.astype(np.float64))
            scores[i] = score

        return scores

    def trs(self, 
            sampled_boxes, 
            sampled_detection, 
            sampled_deltas, 
            dist_x, dist_y, dist_w, dist_h, 
            det_last3_frame_list):
        ## Tracklet Rejection Scheme: to pick the optimum one from S inpainted tracklets
        # sampled_boxes: inpainting bboxes, (n_sample, gap, 4)
        # sampled_detection: predicted tracking bbox, (n_sample, 1, 4)
        # sampled_deltas: inpainting deltas, (n_sample, gap, 4)
        # dist_x: (n_sample, gap+1, 1024)
        # det_last3_frame_list: last 3 frame detection results, (trs+1)×(num_det, 4), trs_default=2

        n_sampling = sampled_boxes.shape[0]
        detection_len = len(det_last3_frame_list)

        iou_list = []
        inpainted_boxes = torch.cat((sampled_boxes, sampled_detection), 1)  ## (n_sample, gap+1, 4)

        for n in range(n_sampling):
            max_sum_iou_n = 0
            for f in range(detection_len):
                max_iou_n_f = 0
                current_inpaint_box = inpainted_boxes[n, f-detection_len, :].cpu().detach().numpy()
                num_det = det_last3_frame_list[f].shape[0]
                detection_current_frame = det_last3_frame_list[f]
                for i in range(num_det):
                    if torch.isnan(det_last3_frame_list[f][i, 0]):  ## no object is detected in that frame
                        continue
                    else:
                        current_det_box = detection_current_frame[i, :].cpu().detach().numpy()
                        current_iou = mm.distances.boxiou(current_inpaint_box, current_det_box)
                        if current_iou > max_iou_n_f:
                            max_iou_n_f = current_iou
                if max_iou_n_f < self.iou_thre_trs and f == 0:
                    break
                elif max_iou_n_f >= self.iou_thre_trs:
                    max_sum_iou_n += max_iou_n_f
            iou_list.append(max_sum_iou_n)    
        if np.max(iou_list) > 0:
            max_iou_idx = np.argmax(iou_list)
            best_inpainting = sampled_boxes[max_iou_idx, :, :].unsqueeze(0)  ## (1, gap, 4)
            best_detection = sampled_detection[max_iou_idx, :, :].unsqueeze(0)
            best_deltas = sampled_deltas[max_iou_idx, :, :].unsqueeze(0)     ## (1, gap, 4)
            best_dist_x = dist_x[max_iou_idx, :-1, :].unsqueeze(0)  ## (1, gap, 1024)
            best_dist_y = dist_y[max_iou_idx, :-1, :].unsqueeze(0)
            best_dist_w = dist_w[max_iou_idx, :-1, :].unsqueeze(0)
            best_dist_h = dist_h[max_iou_idx, :-1, :].unsqueeze(0)
        elif np.max(iou_list) == 0:
            best_inpainting = []
            best_detection = []
            best_deltas = []
            best_dist_x = []
            best_dist_y = []
            best_dist_w = []
            best_dist_h = []

        return best_inpainting, best_detection, best_deltas, best_dist_x, best_dist_y, best_dist_w, best_dist_h

    
    def dump_cache(self):
        self.cmc.dump_cache()
        self.embedder.dump_cache()
