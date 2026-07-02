#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from typing import List, Iterable, Optional
from presidio_analyzer import BatchAnalyzerEngine, DictAnalyzerResult
from presidio_anonymizer import BatchAnonymizerEngine

"""
Parquet variant of the Presidio batch analyzer/anonymizer.
Reads a parquet file into a column-oriented dict, runs Presidio's batch
analyze/anonymize, then writes the redacted result back to parquet.
"""

class ParquetAnalyzer(BatchAnalyzerEngine):
    def analyze_parquet(
        self,
        parquet_full_path: str,
        language: str,
        keys_to_skip: Optional[List[str]] = None,
        **kwargs,
    ) -> Iterable[DictAnalyzerResult]:
        df = pd.read_parquet(parquet_full_path)
        # Presidio works on {column_name: [values...]} — same shape as before,
        # just sourced from parquet instead of csv. Cast to str so the NER
        # engine always receives text.
        parquet_dict = {col: df[col].astype(str).tolist() for col in df.columns}
        analyzer_results = self.analyze_dict(parquet_dict, language, keys_to_skip)
        return list(analyzer_results)


def write_anonymized_parquet(anonymized_dict: dict, output_path: str) -> None:
    """Write the column-oriented anonymized results back to parquet."""
    # A dict of equal-length lists is exactly a DataFrame; no manual transpose.
    df = pd.DataFrame(anonymized_dict)
    df.to_parquet(output_path, index=False)


if __name__ == "__main__":
    analyzer = ParquetAnalyzer()
    analyzer_results = analyzer.analyze_parquet(
        'joint_eval_ceas_text.parquet',
        language="en",
        keys_to_skip=["email_id", "label", "n_urls"],
    )
    anonymizer = BatchAnonymizerEngine()
    anonymized_results = anonymizer.anonymize_dict(analyzer_results)

    write_anonymized_parquet(anonymized_results, 'joint_eval_ceas_text_redacted.parquet')
    print("Redacted parquet written to 'joint_eval_ceas_text_redacted.parquet'")


# In[ ]:




