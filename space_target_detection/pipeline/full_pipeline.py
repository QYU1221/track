"""Public API for running the complete sequence pipeline."""


class FullPipeline:
    """A small explicit boundary around the complete sequence processor.

    The injected processor is the behavior-compatible ``process_sequence``
    function.  Injection avoids hidden imports and keeps this orchestration API
    usable by tests and future front ends.
    """

    def __init__(self, sequence_processor):
        self._sequence_processor = sequence_processor

    def run(self, measurements_dir, output_dir=None, end_frame=None,
            verbose=True, dataset_name=None, snapshot_dir=None):
        return self._sequence_processor(
            measurements_dir,
            output_dir=output_dir,
            end_frame=end_frame,
            verbose=verbose,
            dataset_name=dataset_name,
            snapshot_dir=snapshot_dir,
        )
