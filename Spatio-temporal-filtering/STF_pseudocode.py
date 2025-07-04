temporal_filtering_length = 5
video_predictions = [
    # ...{"frame_number", "xmin", "xmax", "ymin", "ymax", "predicted_behaviour"}
]
updated_video_prediction = [
    # ...{"frame_number", "xmin", "xmax", "ymin", "ymax", "predicted_behaviour"}
]

for current_prediction in video_predictions:
    # find temporally-linked predictions
    temporally_linked_predictions = []
    for prediction in video_predictions:
        is_within_link_length_range = (
            abs(prediction.frame_number, current_prediction.frame_number) < link_length
        )
        if is_within_link_length_range:
            temporally_linked_predictions.append(prediction)

    # find spatially-overlapping predictions
    overlapping_predictions = []
    for prediction in temporally_linked_predictions:
        # calculate overlapping area of 2 bounding boxes
        a = current_prediction
        b = prediction
        overlap_x = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
        overlap_y = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
        if (overlap_x >= 0) and (overlap_y >= 0):
            overlap_area = overlap_x * overlap_y
            overlapping_predictions.append(prediction)

    linked_predictions = overlapping_predictions

    # get predominant predicted behaviour for linked annotations
    predicted_behaviours = [
        prediction["predicted_behaviour"] for prediction in linked_predictions
    ]
    # See Python Counter API
    c = Counter(predicted_behaviours)
    most_common = c.most_common()[0][0]
    
    # update prediction with predominant predicted behaviour of linked predictions
    updated_prediction = current_prediction.copy()
    updated_prediction["predicted_behaviour"] = most_common
    updated_video_prediction.append(updated_prediction)

