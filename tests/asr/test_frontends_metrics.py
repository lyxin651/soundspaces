import inspect
import unittest

import numpy as np

from active_audition.asr import frontends
from active_audition.asr.frontends import FrontendError, apply_frontend
from active_audition.asr.speechbrain_adapter import ASROutput, SpeechBrainASRAdapter, decoder_metadata
from active_audition.evaluation.asr_metrics import ASRMetricError, error_counts, normalize_text


class FrontendAndMetricTest(unittest.TestCase):
    def test_mean_lr_exact_swap_invariant_and_fixed_ears(self):
        waveform = np.asarray([[1.0, 3.0], [-2.0, 2.0]], dtype=np.float32)
        expected = np.asarray([2.0, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(apply_frontend(waveform, "mean_lr"), expected)
        np.testing.assert_array_equal(apply_frontend(waveform[:, ::-1], "mean_lr"), expected)
        np.testing.assert_array_equal(apply_frontend(waveform, "fixed_L"), waveform[:, 0])
        np.testing.assert_array_equal(apply_frontend(waveform, "fixed_R"), waveform[:, 1])

    def test_same_sign_average_and_antiphase_cancellation(self):
        same = np.asarray([[0.25, 0.75]], dtype=np.float32)
        anti = np.asarray([[0.25, -0.25], [-0.5, 0.5]], dtype=np.float32)
        np.testing.assert_array_equal(apply_frontend(same, "mean_lr"), np.asarray([0.5], dtype=np.float32))
        np.testing.assert_array_equal(apply_frontend(anti, "mean_lr"), np.zeros(2, dtype=np.float32))
        with self.assertRaises(FrontendError):
            apply_frontend(same, "best_ear")
        self.assertFalse(hasattr(frontends, "best_ear"))

    def test_text_normalization_and_hand_checkable_word_errors(self):
        self.assertEqual(normalize_text("  It's, 12 o'clock! "), "IT'S 12 O'CLOCK")
        # A B C -> A X C D is one substitution plus one insertion.
        result = error_counts("A B C", "A X C D")
        self.assertEqual((result["S"], result["D"], result["I"], result["N"]), (1, 0, 1, 3))
        self.assertAlmostEqual(result["WER"], 2.0 / 3.0)

    def test_empty_hypothesis_is_all_deletions_and_empty_reference_is_error(self):
        result = error_counts("ONE TWO THREE", "")
        self.assertEqual((result["S"], result["D"], result["I"], result["N"]), (0, 3, 0, 3))
        self.assertEqual(result["WER"], 1.0)
        with self.assertRaises(ASRMetricError):
            error_counts("...", "anything")
        self.assertGreater(error_counts("ONE", "ONE TWO THREE")["WER"], 1.0)

    def test_adapter_transcribe_has_no_reference_parameter(self):
        parameters = inspect.signature(SpeechBrainASRAdapter.transcribe).parameters
        self.assertEqual(list(parameters), ["self", "waveform", "sample_rate", "frontend"])
        self.assertNotIn("reference", parameters)

    def test_adapter_output_records_model_revision_and_artifact_identity(self):
        output = ASROutput(
            hypothesis="TEST",
            score=-1.0,
            score_semantics="raw_decoder_sequence_score_not_confidence_or_probability",
            raw_decoder_metadata={"beam_size": 66},
            input_waveform_sha256="1" * 64,
            frontend="mean_lr",
            model_repo_id="speechbrain/model",
            model_revision="2" * 40,
            model_artifact_sha256="3" * 64,
            decode_contract_sha256="4" * 64,
        ).to_dict()
        self.assertEqual(output["model_revision"], "2" * 40)
        self.assertEqual(output["model_artifact_sha256"], "3" * 64)
        self.assertEqual(output["score_semantics"], "raw_decoder_sequence_score_not_confidence_or_probability")

    def test_decoder_metadata_distinguishes_token_count_from_relative_length(self):
        metadata = decoder_metadata([10, 20, 30], 0.375, np.zeros(8, dtype=np.float32), 66)
        self.assertEqual(metadata["token_count"], 3)
        self.assertEqual(metadata["decoder_length_relative"], 0.375)
        self.assertEqual(metadata["best_log_probs_shape"], [8])


if __name__ == "__main__":
    unittest.main()
