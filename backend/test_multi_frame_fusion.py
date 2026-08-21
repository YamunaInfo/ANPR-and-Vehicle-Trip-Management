"""
Unit tests for Multi-Frame ANPR Recognition & Fusion Engine.
Tests all requirements specified in user prompt.
"""
import unittest
from ai.multi_frame_fusion import (
    MultiFramePlateFusionEngine,
    normalize_ocr_text,
    validate_indian_plate,
    character_confusion_distance,
    plate_similarity
)


class TestMultiFramePlateFusion(unittest.TestCase):

    def setUp(self):
        self.engine = MultiFramePlateFusionEngine(
            window_size=10,
            min_observations=3,
            min_confidence=0.60,
            min_agreement=0.60
        )

    def test_normalization_and_hsrp_cleaning(self):
        self.assertEqual(normalize_ocr_text("wb 12 ab 1234"), "WB12AB1234")
        self.assertEqual(normalize_ocr_text("INDTN38AB1234"), "TN38AB1234")
        self.assertEqual(normalize_ocr_text("TN-38-AB-1234"), "TN38AB1234")
        self.assertEqual(normalize_ocr_text("tn 38 ab 1234"), "TN38AB1234")

    def test_character_confusion_similarity(self):
        # B and 8 are common OCR confusion: distance should be minimal
        dist_b8 = character_confusion_distance("WB12AB1234", "WB12A81234")
        self.assertLess(dist_b8, 0.5)
        sim_b8 = plate_similarity("WB12AB1234", "WB12A81234")
        self.assertGreaterEqual(sim_b8, 0.90)

        # O and 0
        sim_o0 = plate_similarity("DL01AB1234", "DLO1AB1234")
        self.assertGreaterEqual(sim_o0, 0.90)

    def test_1_user_acceptance_weighted_fusion(self):
        """
        TEST 1:
        Frame 1: WB12AB1234 (0.78)
        Frame 2: WB12AB1234 (0.91)
        Frame 3: WB12A81234 (0.66)
        Frame 4: WB12AB1234 (0.94)
        Expected Final Plate: WB12AB1234 (and NOT WB12A81234)
        """
        track_id = 17
        self.engine.add_observation(track_id, "WB12AB1234", ocr_confidence=0.78, frame_number=1)
        self.engine.add_observation(track_id, "WB12AB1234", ocr_confidence=0.91, frame_number=2)
        self.engine.add_observation(track_id, "WB12A81234", ocr_confidence=0.66, frame_number=3)
        res = self.engine.add_observation(track_id, "WB12AB1234", ocr_confidence=0.94, frame_number=4)

        self.assertEqual(res["status"], "finalized")
        self.assertEqual(res["plate_number"], "WB12AB1234")
        self.assertNotEqual(res["plate_number"], "WB12A81234")
        self.assertGreaterEqual(res["final_confidence"], 0.80)
        self.assertEqual(res["frame_count"], 4)
        self.assertEqual(res["evidence_count"], 4)  # All 4 frames support the cluster

    def test_2_track_separation(self):
        """
        TEST 2:
        Two different track IDs must never share recognition history.
        """
        # Track 1
        self.engine.add_observation(1, "WB12AB1234", ocr_confidence=0.78, frame_number=1)
        self.engine.add_observation(1, "WB12AB1234", ocr_confidence=0.91, frame_number=2)
        self.engine.add_observation(1, "WB12AB1234", ocr_confidence=0.94, frame_number=3)

        # Track 2
        self.engine.add_observation(2, "TN38AB1234", ocr_confidence=0.88, frame_number=5)
        self.engine.add_observation(2, "TN38AB1234", ocr_confidence=0.92, frame_number=6)
        res2 = self.engine.add_observation(2, "TN38AB1234", ocr_confidence=0.95, frame_number=7)

        res1 = self.engine.evaluate_track(1)

        self.assertEqual(res1["plate_number"], "WB12AB1234")
        self.assertEqual(res2["plate_number"], "TN38AB1234")
        self.assertEqual(len(self.engine.plate_history[1]), 3)
        self.assertEqual(len(self.engine.plate_history[2]), 3)

    def test_3_minimum_evidence_pending_state(self):
        """
        TEST 3:
        One weak frame: status = pending, display_plate = 'Recognizing...'
        """
        track_id = 5
        res = self.engine.add_observation(track_id, "WB12AB1234", ocr_confidence=0.55, frame_number=1)
        self.assertEqual(res["status"], "pending")
        self.assertEqual(res["display_plate"], "Recognizing...")
        self.assertFalse(res["is_finalized"])

    def test_4_conflicting_results_manual_review(self):
        """
        TEST 4:
        Conflicting results: status = manual_review, display_plate = 'Requires Manual Review'
        """
        track_id = 9
        self.engine.add_observation(track_id, "TN38AB1234", ocr_confidence=0.62, frame_number=1)
        self.engine.add_observation(track_id, "MH12XY9999", ocr_confidence=0.61, frame_number=2)
        self.engine.add_observation(track_id, "KA05ZZ1111", ocr_confidence=0.60, frame_number=3)
        self.engine.add_observation(track_id, "DL01AA0000", ocr_confidence=0.58, frame_number=4)
        self.engine.add_observation(track_id, "GJ18BB2222", ocr_confidence=0.59, frame_number=5)

        # Evaluate track with conflicting evidence
        res = self.engine.evaluate_track(track_id, force_finalize=True)
        self.assertEqual(res["status"], "manual_review")
        self.assertEqual(res["display_plate"], "Requires Manual Review")
        self.assertTrue(res["is_finalized"])

    def test_5_deduplication(self):
        """
        TEST 5:
        100 detections of the same vehicle produce ONE finalized event.
        """
        track_id = 17
        for f in range(1, 101):
            conf = 0.85 + (0.05 if f % 2 == 0 else -0.05)
            # Occasional OCR noise in 1 frame
            plate_text = "WB12A81234" if f == 50 else "WB12AB1234"
            self.engine.add_observation(track_id, plate_text, ocr_confidence=conf, frame_number=f)

        finalized = self.engine.finalize_all_active_tracks()
        self.assertEqual(len(finalized), 1)
        self.assertEqual(finalized[0]["track_id"], 17)
        self.assertEqual(finalized[0]["plate_number"], "WB12AB1234")
        self.assertEqual(finalized[0]["status"], "finalized")


if __name__ == "__main__":
    unittest.main()
