import os
import json
import datasets


_CITATION = """"""
_DESCRIPTION = """"""

task = "topic_segmentation"
version = datasets.Version("1.1.1")
desc = "vt_ssum_30min_clean2_stagerealigned_relabel_only"


class TopicSegmentationConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(TopicSegmentationConfig, self).__init__(**kwargs)


class VtSsum30MinClean2StagerealignedRelabelOnlyMultiModalityForTopicSegmentation(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        TopicSegmentationConfig(
            name="VtSsum30MinClean2StagerealignedRelabelOnlyMultiModalityForTopicSegmentation",
            version=version,
            description=desc,
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "example_id": datasets.Value("int32"),
                    "lecture": datasets.Value("string"),
                    "sentences": datasets.Sequence(datasets.Value("string")),
                    "labels": datasets.Sequence(datasets.Value("string")),
                    "stet": datasets.Sequence(datasets.Sequence(datasets.Value("float"))),
                    "topic_end_seconds": datasets.Sequence(datasets.Value("float")),
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_root = os.environ.get("DATA_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
        dataset_dir = os.path.normpath(
            os.path.join(data_root, "vt_ssum_30min_clean2_stagerealigned_relabel_only", "processed_jsonl")
        )
        urls_to_download = {
            "train": os.path.join(dataset_dir, "train.jsonl"),
            "dev": os.path.join(dataset_dir, "dev.jsonl"),
            "test": os.path.join(dataset_dir, "test.jsonl"),
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        print(task)
        label_map = {"1": "B-EOP", "0": "O", 1: "B-EOP", 0: "O"}
        with open(filepath, "r") as f:
            for index, line in enumerate(f.readlines()):
                example = json.loads(line.strip())
                sentences = example["text"]
                labels = [label_map[v] if v in label_map else -100 for v in example["labels"]]
                yield index, {
                    "example_id": index,
                    "lecture": example["example_id"],
                    "sentences": sentences,
                    "labels": labels,
                    "stet": example.get("stet", []),
                    "topic_end_seconds": example.get("topic_end_seconds", []),
                }
