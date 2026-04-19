# Copyright (c) OpenMMLab. All rights reserved.
import math
import copy
import torch
import torch.nn.functional as F

from typing import List, Optional, Tuple
from torch import Tensor

from mmengine import MessageHub
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_project, bbox2roi, bbox_overlaps
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.structures import InstanceData
from mmdet.models.detectors.soft_teacher import SoftTeacher


@MODELS.register_module()
class PGDT(SoftTeacher):
    r"""Implementation of Physics-Guided Dual-Teacher (PGDT) Framework.

    This class introduces the Physics Teacher and the Dual-Arbitration Mechanism
    (Rules 1, 2, and 3) tailored for SAR ship detection.
    """

    def __init__(self,
                 detector: ConfigType,
                 physics_cfg: dict = None,
                 semi_train_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            detector=detector,
            semi_train_cfg=semi_train_cfg,
            semi_test_cfg=semi_test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        self.physics_cfg = physics_cfg or dict(
            k=3.0, window_size=7,
            tau_high=0.7, tau_low=0.3
        )

    @torch.no_grad()
    def compute_physics_prior(self, img: Tensor) -> Tensor:
        """
        Compute Physical Prior Map (P_phys) via Vectorized IS-Transform.
        Note: The current framework strictly supports single-polarization (C=1).
        """
        B, C, H, W = img.shape
        p_phys_list = []

        assert C == 1, "PGDT currently supports single-polarization (C=1) SAR images via IS-Transform."

        for i in range(B):
            single_img = img[i]

            # Vectorized approximation of IS-Transform (AvgPool)
            k = self.physics_cfg.get('k', 3.0)
            win = self.physics_cfg.get('window_size', 7)
            pad = win // 2

            local_mean = F.avg_pool2d(single_img.unsqueeze(0), win, stride=1, padding=pad)
            feat = single_img * (1.0 + k * local_mean)

            # Image-wise normalization to [0, 1]
            if feat.std() > 0:
                feat_norm = (feat - feat.mean()) / feat.std()
            else:
                feat_norm = feat

            p_phys = torch.sigmoid(feat_norm)
            p_phys_list.append(p_phys)

        return torch.stack(p_phys_list)

    @torch.no_grad()
    def get_pseudo_instances(
            self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Tuple[SampleList, Optional[dict]]:
        """PGDT Dual-Arbitration Mechanism"""
        assert self.teacher.with_bbox, 'Bbox head must be implemented.'

        # === Dynamic Curriculum Scheduling ===
        message_hub = MessageHub.get_current_instance()
        current_iter = message_hub.get_info('iter') if message_hub else 0
        MAX_ITERS = 90000.0  # Align with config max_iters

        T_low_init, T_low_end = 0.3, 0.05
        T_high_init, T_high_end = 0.7, 0.85

        progress = min(current_iter / MAX_ITERS, 1.0)
        alpha = 0.5 * (1 - math.cos(math.pi * progress))

        self.physics_cfg['tau_low'] = T_low_init - (T_low_init - T_low_end) * alpha
        self.physics_cfg['tau_high'] = T_high_init + (T_high_end - T_high_init) * alpha

        # 1. Semantic Teacher Feature Extraction
        x = self.teacher.extract_feat(batch_inputs)

        # 2. Physics Teacher Generates Prior Map
        p_phys_map = self.compute_physics_prior(batch_inputs)

        # 3. RPN Predictions (Candidates for Rule 3 Mining)
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.teacher.rpn_head.predict(x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [data.proposals for data in batch_data_samples]

        # 4. Semantic Teacher Predictions (Rule 1 & 2 Candidates)
        sem_results_list = self.teacher.roi_head.predict(x, rpn_results_list, batch_data_samples, rescale=False)

        final_pseudo_instances = []
        tau_high = self.physics_cfg['tau_high']
        tau_low = self.physics_cfg['tau_low']

        for i, (sem_results, rpn_results) in enumerate(zip(sem_results_list, rpn_results_list)):
            sem_bboxes = sem_results.bboxes
            sem_scores = sem_results.scores
            sem_labels = sem_results.labels

            sem_mask = sem_scores > self.semi_train_cfg.cls_pseudo_thr
            keep_indices = []

            def compute_sar_phys_score(feats):
                return 0.7 * feats.amax(dim=[1, 2, 3]) + 0.3 * feats.std(dim=[1, 2, 3])

            # --- Rule 1 & 2: Golden Positives & Physics Veto ---
            if sem_mask.sum() > 0:
                valid_inds = sem_mask.nonzero(as_tuple=True)[0]
                temp_bboxes = sem_bboxes[valid_inds]

                rois = bbox2roi([temp_bboxes])
                phys_feats = self.teacher.roi_head.bbox_roi_extractor([p_phys_map[i:i + 1]], rois)
                s_phys = compute_sar_phys_score(phys_feats)

                tau_veto = self.physics_cfg['tau_low']
                safe_score_thr = 0.9  # Safety Lock

                veto_mask = (s_phys > tau_veto) | (sem_scores[valid_inds] > safe_score_thr)
                keep_indices = valid_inds[veto_mask]

            final_bboxes = sem_bboxes[keep_indices]
            final_labels = sem_labels[keep_indices]
            final_scores = sem_scores[keep_indices]
            is_mining = torch.zeros_like(final_labels, dtype=torch.bool)

            # --- Rule 3: Physics Mining & Teacher-Guided Refinement ---
            rpn_boxes = rpn_results.bboxes[:1000]

            if len(rpn_boxes) > 0:
                rois_rpn = bbox2roi([rpn_boxes])
                phys_feats_rpn = self.teacher.roi_head.bbox_roi_extractor([p_phys_map[i:i + 1]], rois_rpn)
                rpn_s_phys = compute_sar_phys_score(phys_feats_rpn)

                tau_mining = self.physics_cfg['tau_high']
                mining_mask = rpn_s_phys > tau_mining
                mined_inds = mining_mask.nonzero(as_tuple=True)[0]

                if len(mined_inds) > 0:
                    raw_mined_bboxes = rpn_boxes[mined_inds]

                    with torch.no_grad():
                        current_img_feat = [lvl[i:i + 1] for lvl in x]
                        rois_refine = bbox2roi([raw_mined_bboxes])
                        bbox_feats = self.teacher.roi_head.bbox_roi_extractor(current_img_feat, rois_refine)

                        if self.teacher.roi_head.with_shared_head:
                            bbox_feats = self.teacher.roi_head.shared_head(bbox_feats)

                        _, rpn_bbox_pred = self.teacher.roi_head.bbox_head(bbox_feats)
                        bbox_head = self.teacher.roi_head.bbox_head

                        if bbox_head.reg_class_agnostic:
                            deltas = rpn_bbox_pred
                            refined_bboxes = bbox_head.bbox_coder.decode(raw_mined_bboxes, deltas)
                        else:
                            deltas = rpn_bbox_pred.view(rpn_bbox_pred.size(0), -1, 4)[:, 0, :]
                            refined_bboxes = bbox_head.bbox_coder.decode(raw_mined_bboxes, deltas)

                        img_shape = batch_data_samples[i].img_shape
                        refined_bboxes[:, 0::2] = refined_bboxes[:, 0::2].clamp(min=0, max=img_shape[1])
                        refined_bboxes[:, 1::2] = refined_bboxes[:, 1::2].clamp(min=0, max=img_shape[0])
                        mined_bboxes = refined_bboxes

                    is_new = torch.ones(len(mined_bboxes), dtype=torch.bool, device=batch_inputs.device)
                    if len(final_bboxes) > 0:
                        ious = bbox_overlaps(mined_bboxes, final_bboxes)
                        max_ious, _ = ious.max(dim=1)
                        is_new = max_ious < 0.1  # IoU filtering constraint
                        mined_bboxes = mined_bboxes[is_new]

                    if len(mined_bboxes) > 0:
                        mined_labels = torch.zeros(len(mined_bboxes), dtype=torch.long, device=batch_inputs.device)
                        current_phys_scores = rpn_s_phys[mining_mask][is_new]
                        mined_scores = torch.clamp(current_phys_scores, min=0.8, max=1.0)
                        mined_flags = torch.ones(len(mined_bboxes), dtype=torch.bool, device=batch_inputs.device)

                        final_bboxes = torch.cat([final_bboxes, mined_bboxes])
                        final_labels = torch.cat([final_labels, mined_labels])
                        final_scores = torch.cat([final_scores, mined_scores])
                        is_mining = torch.cat([is_mining, mined_flags])

            pseudo_instance = InstanceData()
            pseudo_instance.bboxes = final_bboxes
            pseudo_instance.labels = final_labels
            pseudo_instance.scores = final_scores
            pseudo_instance.is_mining = is_mining
            final_pseudo_instances.append(pseudo_instance)

        for data_samples, results in zip(batch_data_samples, final_pseudo_instances):
            data_samples.gt_instances = results

        reg_uncs_list = self.compute_uncertainty_with_aug(x, batch_data_samples)

        for data_samples, reg_uncs in zip(batch_data_samples, reg_uncs_list):
            data_samples.gt_instances['reg_uncs'] = reg_uncs
            data_samples.gt_instances.bboxes = bbox_project(
                data_samples.gt_instances.bboxes,
                torch.from_numpy(data_samples.homography_matrix).inverse().to(
                    self.data_preprocessor.device), data_samples.ori_shape)

        batch_info = {
            'feat': x,
            'img_shape': [],
            'homography_matrix': [],
            'metainfo': []
        }
        for data_samples in batch_data_samples:
            batch_info['img_shape'].append(data_samples.img_shape)
            batch_info['homography_matrix'].append(
                torch.from_numpy(data_samples.homography_matrix).to(
                    self.data_preprocessor.device))
            batch_info['metainfo'].append(data_samples.metainfo)

        return batch_data_samples, batch_info

    def rcnn_reg_loss_by_pseudo_instances(
            self, x: Tuple[Tensor], unsup_rpn_results_list: List,
            batch_data_samples: SampleList) -> dict:
        """
        Rewrite Regression Loss: Ensures Rule 3 (Mining) samples are excluded
        from the regression loss computation.
        """
        valid_reg_samples = []
        for data_sample in batch_data_samples:
            valid_sample = copy.deepcopy(data_sample)

            if len(valid_sample.gt_instances) > 0:
                if hasattr(valid_sample.gt_instances, 'is_mining'):
                    valid_mask = ~valid_sample.gt_instances.is_mining
                    valid_sample.gt_instances = valid_sample.gt_instances[valid_mask]

            valid_reg_samples.append(valid_sample)

        return super().rcnn_reg_loss_by_pseudo_instances(
            x, unsup_rpn_results_list, valid_reg_samples)