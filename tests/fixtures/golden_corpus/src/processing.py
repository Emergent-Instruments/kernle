"""Processing utilities for the golden corpus test."""


def chunk_text(text, max_size=500):
    """Split text into chunks at paragraph boundaries."""
    paragraphs = text.split("\\n\\n")
    chunks = []
    current = []
    current_size = 0

    for para in paragraphs:
        if current_size + len(para) > max_size and current:
            chunks.append("\\n\\n".join(current))
            current = [para]
            current_size = len(para)
        else:
            current.append(para)
            current_size += len(para)

    if current:
        chunks.append("\\n\\n".join(current))
    return chunks


class Pipeline:
    """Memory processing pipeline."""

    def __init__(self, config=None):
        self._config = config or {}
        self._steps = []

    def add_step(self, step_fn):
        """Add a processing step to the pipeline."""
        self._steps.append(step_fn)

    def run(self, data):
        """Execute all pipeline steps on the data."""
        result = data
        for step in self._steps:
            result = step(result)
        return result
