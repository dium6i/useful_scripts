def non_max_suppression(boxes, scores, iou_thresh):
    """
    Perform Non-Maximum Suppression to filter out overlapping bounding boxes.

    Args:
        boxes (list): Bounding boxes in the format [[x1, y1, x2, y2], ...].
        scores (list): Confidence scores corresponding to each bounding box.
        iou_thresh (float): Threshold for Intersection over Union.

    Returns:
        keep_indices (list): Indices to keep after NMS.
    """
    if len(boxes) == 0:
        return []

    # Sort boxes by their scores in descending order
    sorted_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )

    keep_indices = []
    while len(sorted_indices) > 0:
        # Pick the box with the highest confidence score
        max_index = sorted_indices[0]
        keep_indices.append(max_index)

        # Compute IoU between the picked box and all other boxes
        selected_box = boxes[max_index]
        other_boxes = [boxes[i] for i in sorted_indices[1:]]
        iou_scores = [
            calculate_iou(selected_box, other_box)
            for other_box in other_boxes
        ]

        # Filter out boxes with IoU greater than threshold
        filtered_indices = [
            i for i, iou in zip(sorted_indices[1:], iou_scores)
            if iou <= iou_thresh
        ]
        sorted_indices = filtered_indices

    return keep_indices


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (list): Bounding box in the format [x1, y1, x2, y2].
        box2 (list): Bounding box in the format [x1, y1, x2, y2].

    Returns:
        iou: Intersection over Union (IoU) score.
    """
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # Calculate the coordinates of the intersection rectangle
    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)

    # Calculate intersection area
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * \
        max(0, intersection_y2 - intersection_y1 + 1)

    # Calculate areas of each bounding box
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x4 - x3 + 1) * (y4 - y3 + 1)

    # Calculate Union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou
