def overlap(a, b, max_allowed_iou):
    """Find if two bounding boxes are overlapping or not. This is determined by maximum allowed
       IOU between bounding boxes. If IOU is less than the max allowed IOU then bounding boxes
       don't overlap

    Args:
        a(Rectangle): Bounding box 1
        b(Rectangle): Bounding box 2
    Returns:
        bool: True if boxes overlap else False
    """
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)

    if (
        dx >= 0
        and dy >= 0
        and (
            float(dx * dy) > max_allowed_iou * (a.xmax - a.xmin) * (a.ymax - a.ymin)
            or float(dx * dy) > max_allowed_iou * (b.xmax - b.xmin) * (b.ymax - b.ymin)
        )
    ):
        return True
    else:
        return False
