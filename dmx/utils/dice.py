import numpy as np

from skimage.measure import regionprops

def box_intersection(boxes_a, boxes_b):
    """Compute pairwise intersection areas between boxes.

    Args:
      boxes_a: [..., N, 2d]
      boxes_b: [..., M, 2d]

    Returns:
      its: [..., N, M] representing pairwise intersections.
    """
    minimum = np.minimum
    maximum = np.maximum
    boxes_a = np.array(boxes_a)
    boxes_b = np.array(boxes_b)

    ndim = boxes_a.shape[-1] // 2
    assert ndim * 2 == boxes_a.shape[-1]
    assert ndim * 2 == boxes_b.shape[-1]

    min_vals_1 = boxes_a[..., None, :ndim]  # [..., N, 1, d]
    max_vals_1 = boxes_a[..., None, ndim:]
    min_vals_2 = boxes_b[..., None, :, :ndim]  # [..., 1, M, d]
    max_vals_2 = boxes_b[..., None, :, ndim:]

    min_max = minimum(max_vals_1, max_vals_2)  # [..., N, M, d]
    max_min = maximum(min_vals_1, min_vals_2)

    intersects = maximum(0, min_max - max_min)  # [..., N, M, d]

    return intersects.prod(axis=-1)


class Dice:
    """Compute instance Dice values"""

    def __init__(self):
        self.pred_areas = []
        self.gt_areas = []
        self.pred_scores = []
        self.gt_scores = []


    def _update(self, pred_its, pred_areas, gt_areas):
        self.pred_areas.append(pred_areas)
        self.gt_areas.append(gt_areas)

        pred_best = pred_its.max(axis=1)
        pred_best_matches = pred_its.argmax(axis=1)
        pred_dice = pred_best * 2 / (pred_areas + gt_areas[pred_best_matches])
        self.pred_scores.append(pred_dice)

        gt_best = pred_its.max(axis=0)
        gt_best_matches = pred_its.argmax(axis=0)
        gt_dice = gt_best * 2 / (gt_areas + pred_areas[gt_best_matches])
        self.gt_scores.append(gt_dice)


    def update(self, pred_mask, gt_mask):
        gt_rps = regionprops(gt_mask)
        pred_rps = regionprops(pred_mask)

        if len(gt_rps) == 0 and len(pred_rps) == 0:
            return

        if len(gt_rps) == 0:
            pred_areas = np.stack([rp.area for rp in pred_rps])
            self.pred_areas.append(pred_areas)
            self.pred_scores.append(np.zeros([pred_areas.shape[0]]))
            return
        
        if len(pred_rps) == 0:
            gt_areas = np.stack([rp.area for rp in gt_rps])
            self.gt_areas.append(gt_areas)
            self.gt_scores.append(np.zeros([gt_areas.shape[0]]))
            return

        pred_bboxes = np.stack([rp.bbox for rp in pred_rps])
        pred_areas = np.stack([rp.area for rp in pred_rps])
        gt_bboxes = np.stack([rp.bbox for rp in gt_rps])
        gt_areas = np.stack([rp.area for rp in gt_rps])
        labels = [rp.label for rp in gt_rps]

        box_its = box_intersection(pred_bboxes, gt_bboxes)

        def _get_its(pid, gid):
            y0,x0,y1,x1 = pred_bboxes[pid]
            return np.count_nonzero(gt_mask[y0:y1, x0:x1] == labels[gid])

        mask_its = np.zeros_like(box_its, dtype=int)
        ids = np.where(box_its > 0)
        mask_its[ids] = [_get_its(pid, gid) for pid, gid in zip(*ids)]

        self._update(mask_its, pred_areas, gt_areas)


    def compute(self):
        pred_areas = np.concatenate(self.pred_areas)
        gt_areas = np.concatenate(self.gt_areas)
        pred_scores = np.concatenate(self.pred_scores)
        gt_scores = np.concatenate(self.gt_scores)

        pred_dice = (pred_areas / pred_areas.sum() * pred_scores).sum()
        gt_dice = (gt_areas / gt_areas.sum() * gt_scores).sum()

        dice = (pred_dice + gt_dice) / 2

        return dice
