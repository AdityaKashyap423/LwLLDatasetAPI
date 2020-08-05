from dataclasses import dataclass
from typing import Optional, List


# METRICS = ['accuracy']
DATASET_TYPES = ['image_classification', 'object_detection', 'machine_translation']

# @dataclass
# class ProblemDoc:
#     problem_id: str
#     base_dataset: str
#     base_label_budget: int
#     base_evaluation_metrics: List[str]
#     adaptation_dataset: str
#     adaptation_label_budget: int
#     adaptation_evalutation_metrics: List[str]
#     base_can_use_pretrained_model: bool = True
#     adaptation_can_use_pretrained_model: bool = True
#     blacklisted_resources: List[str] = field(default_factory=lambda: [])

#     def __post_init__(self) -> None:
#         self._validate_metrics(self.base_evaluation_metrics)
#         self._validate_metrics(self.adaptation_evalutation_metrics)

#     @staticmethod
#     def _validate_metrics(metrics: List[str]) -> None:
#         for metric in metrics:
#             if metric not in METRICS:
#                 raise Exception(f'Metric: {metric} not supported yet.')

@dataclass
class DatasetDoc:
    name: str
    dataset_type: str
    sample_number_of_samples_train: int
    sample_number_of_samples_test: int
    sample_number_of_classes: Optional[int]  # Only in image problems
    full_number_of_samples_train: int
    full_number_of_samples_test: int
    full_number_of_classes: Optional[int]  # Only in image problems
    number_of_channels: Optional[int]  # Only in image problems
    classes: Optional[List[str]]
    language_to: Optional[str]  # Only in MT problems
    language_from: Optional[str]  # Only in MT problems
    sample_total_codecs: Optional[int]  # Only in MT problems
    full_total_codecs: Optional[int]  # Only in MT problems
    license_requirements: str
    license_link: str
    license_citation: str

    def __post_init__(self) -> None:
        self._validate_dataset_type(self.dataset_type)

    @staticmethod
    def _validate_dataset_type(dataset_type: str) -> None:
        if dataset_type not in DATASET_TYPES:
            raise Exception(f'Dataset Type: {dataset_type} not supported yet.')
