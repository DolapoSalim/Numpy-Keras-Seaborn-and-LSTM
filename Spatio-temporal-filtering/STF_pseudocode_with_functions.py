# calculate overlapping area of 2 bounding boxes
def calculate_overlap_area(a, b):
    overlap_x = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    overlap_y = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (overlap_x >= 0) and (overlap_y >= 0):
        return overlap_x * overlap_y


def get_linked_predictions(selected_prediction, video_predictions, link_length=5):
    # find temporally-linked predictions
    temporally_linked_predictions = []
    for prediction in video_predictions:
        is_within_link_length_range = (
            abs(selected_prediction.frame_number, prediction.frame_number) < link_length
        )
        if is_within_link_length_range:
            temporally_linked_predictions.append(prediction)

    # find spatially-overlapping predictions
    overlapping_predictions = []
    for prediction in temporally_linked_predictions:
        overlap_area = calculate_overlap_area(prediction, selected_prediction)
        if overlap_area > 0:
            overlapping_predictions.append(prediction)

    return overlapping_predictions


temporal_filtering_length = 5
video_predictions = [
    # ...{"frame_number", "xmin", "xmax", "ymin", "ymax", "predicted_behaviour"}
]
updated_video_prediction = [
    # ...{"frame_number", "xmin", "xmax", "ymin", "ymax", "predicted_behaviour"}
]

for prediction in video_predictions:
    # find temporally and spatially linked annotations
    linked_predictions = get_linked_predictions(
        selected_prediction=prediction,
        video_predictions=video_predictions,
        link_length=temporal_filtering_length,
    )

    # get predominant predicted behaviour for linked annotations
    predicted_behaviours = [
        prediction["predicted_behaviour"] for prediction in linked_predictions
    ]
    # See Python Counter API
    c = Counter(predicted_behaviours)
    most_common = c.most_common()[0][0]

    # update prediction with predominant predicted behaviour of linked predictions  
    updated_prediction = prediction.copy()
    updated_prediction["predicted_behaviour"] = most_common
    updated_video_prediction.append(updated_prediction)