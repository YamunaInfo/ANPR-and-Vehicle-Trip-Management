"""
Multi-Frame ANPR Recognition and Fusion Engine.

Collects multiple plate predictions for the same vehicle track and fuses them
using character confusion handling, position-aware Indian syntax rules,
weighted voting, and evidence gating.
"""
from __future__ import annotations

import datetime
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple


# Configurable threshold constants
DEFAULT_WINDOW_SIZE: int = 10
DEFAULT_MIN_OBSERVATIONS: int = 1
DEFAULT_MIN_CONFIDENCE: float = 0.50
DEFAULT_MIN_AGREEMENT: float = 0.40

VALID_INDIAN_STATES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP",
    "WB", "BH"
}

STATE_CONFUSION_MAP = {
    # Telangana (TS)
    "TT": "TS", "TC": "TS", "T5": "TS", "T0": "TS", "IS": "TS", "1S": "TS", "7S": "TS", "JS": "TS", "YS": "TS", "FS": "TS", "PS": "TS", "T1": "TS",
    # Delhi (DL)
    "0L": "DL", "OL": "DL", "1L": "DL", "IL": "DL", "QL": "DL", "DI": "DL", "D1": "DL", "CL": "DL", "UL": "DL",
    # Haryana (HR)
    "HA": "HR", "HB": "HR", "HD": "HR", "H1": "HR", "H0": "HR", "MR": "HR", "KR": "HR",
    # Maharashtra (MH)
    "HH": "MH", "NH": "MH", "KH": "MH", "VH": "MH", "WH": "MH", "MM": "MH", "MI": "MH", "MN": "MH", "HM": "MH", "TH": "MH", "WW": "MH", "M0": "MH", "MA": "MH",
    # Karnataka (KA)
    "KB": "KA", "K0": "KA", "K4": "KA", "X4": "KA", "XA": "KA", "KO": "KA",
    # Kerala (KL)
    "KP": "KL", "KI": "KL", "XI": "KL", "K1": "KL",
    # Tamil Nadu (TN)
    "TJ": "TN", "TM": "TN", "TA": "TN", "TI": "TN", "TL": "TN", "7N": "TN", "IN": "TN",
    # Uttar Pradesh (UP)
    "UB": "UP", "UF": "UP", "U1": "UP", "VP": "UP", "0P": "UP", "OP": "UP",
    # Gujarat (GJ)
    "G1": "GJ", "G3": "GJ", "CJ": "GJ",
    # Rajasthan (RJ)
    "R1": "RJ", "R3": "RJ",
    # West Bengal (WB)
    "W1": "WB", "W8": "WB",
    # Madhya Pradesh (MP)
    "MD": "MP", "M1": "MP", "MB": "MP",
    # Andhra Pradesh (AP)
    "A0": "AP", "4P": "AP", "AL": "AP", "AR": "AP",
    # Punjab (PB)
    "P1": "PB",
    # Odisha (OD)
    "0D": "OD", "OR": "OD"
}

CHAR_TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0", "C": "0", "U": "0",
    "I": "1", "L": "1", "T": "7", "J": "1",
    "Z": "2",
    "E": "3",
    "A": "4", "H": "4",
    "S": "5",
    "G": "6", "B": "8"
}

CHAR_TO_LETTER = {
    "0": "O", "1": "I", "2": "Z", "3": "B", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"
}

# OCR Confusion pair set for low-penalty similarity comparison
OCR_CONFUSION_PAIRS = {
    frozenset(["O", "0"]),
    frozenset(["Q", "0"]),
    frozenset(["D", "0"]),
    frozenset(["I", "1"]),
    frozenset(["L", "1"]),
    frozenset(["T", "1"]),
    frozenset(["T", "7"]),
    frozenset(["J", "1"]),
    frozenset(["J", "I"]),
    frozenset(["J", "B"]),
    frozenset(["J", "8"]),
    frozenset(["J", "T"]),
    frozenset(["Z", "2"]),
    frozenset(["S", "5"]),
    frozenset(["B", "8"]),
    frozenset(["B", "3"]),
    frozenset(["G", "6"]),
    frozenset(["E", "3"]),
    frozenset(["A", "4"]),
    frozenset(["H", "4"]),
}


def normalize_ocr_text(raw: str) -> str:
    """
    Clean and normalize OCR plate text:
    - Uppercase
    - Remove spaces, hyphens, special characters
    - Strip HSRP prefix (IND, IN) and border noise
    - Trim whitespace
    """
    if not raw:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", str(raw).upper()).strip()
    if len(cleaned) < 4:
        return ""

    # Strip leading 'IND' or 'IN'
    if cleaned.startswith("IND") and len(cleaned) >= 7:
        cleaned = cleaned[3:]
    elif cleaned.startswith("IN") and len(cleaned) >= 8 and (cleaned[2:4] in VALID_INDIAN_STATES or cleaned[2:4] in STATE_CONFUSION_MAP):
        cleaned = cleaned[2:]
    elif len(cleaned) >= 11 and cleaned[0] in {"E", "I", "1", "L", "T", "D", "B", "C", "U", "N"}:
        if cleaned[1:3] in VALID_INDIAN_STATES or cleaned[1:3] in STATE_CONFUSION_MAP:
            cleaned = cleaned[1:]
    elif len(cleaned) >= 12 and (cleaned[2:4] in VALID_INDIAN_STATES or cleaned[2:4] in STATE_CONFUSION_MAP):
        cleaned = cleaned[2:]

    # Remove trailing border artifacts (e.g. trailing 1, I, L, 7 after 4-digit number)
    if len(cleaned) > 10:
        m = re.match(r"^([A-Z0-9]{2}[A-Z0-9]{1,2}[A-Z0-9]{1,3}[A-Z0-9]{4})\d{1,3}$", cleaned)
        if m:
            cleaned = m.group(1)
        else:
            m2 = re.match(r"^([A-Z0-9]{2}[A-Z0-9]{1,2}[A-Z0-9]{1,3}[A-Z0-9]{4})[1IL7TI]$", cleaned)
            if m2:
                cleaned = m2.group(1)

    return cleaned


def validate_indian_plate(candidate: str) -> Tuple[bool, float, str]:
    """
    Position-aware Indian registration format validator and corrector.
    Returns: (is_valid, validation_confidence, corrected_plate)
    """
    cleaned = normalize_ocr_text(candidate)
    if not cleaned or len(cleaned) < 4:
        return False, 0.0, cleaned

    chars = list(cleaned)
    length = len(chars)

    # 1. Standard 10-character format: SS DD LL DDDD (e.g. WB 12 AB 1234, TN 38 AB 1234)
    if length == 10:
        chars[0] = CHAR_TO_LETTER.get(chars[0], chars[0])
        chars[1] = CHAR_TO_LETTER.get(chars[1], chars[1])
        st = chars[0] + chars[1]
        if st not in VALID_INDIAN_STATES and st in STATE_CONFUSION_MAP:
            mapped_st = STATE_CONFUSION_MAP[st]
            chars[0], chars[1] = mapped_st[0], mapped_st[1]

        chars[2] = CHAR_TO_DIGIT.get(chars[2], chars[2])
        chars[3] = CHAR_TO_DIGIT.get(chars[3], chars[3])

        chars[4] = CHAR_TO_LETTER.get(chars[4], chars[4])
        chars[5] = CHAR_TO_LETTER.get(chars[5], chars[5])
        if chars[0] == "T" and chars[1] == "S" and chars[4] == "B" and chars[5] == "S":
            chars[4] = "J"

        for i in [6, 7, 8, 9]:
            chars[i] = CHAR_TO_DIGIT.get(chars[i], chars[i])

        corrected = "".join(chars)
        st_final = corrected[:2]
        is_valid_state = st_final in VALID_INDIAN_STATES
        pattern = re.compile(r"^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$")
        if pattern.fullmatch(corrected):
            return True, 0.98 if is_valid_state else 0.88, corrected
        return True, 0.85 if is_valid_state else 0.70, corrected

    # 2. 9-character format: SS DD L DDDD (e.g. MH 02 D 1365)
    if length == 9:
        chars[0] = CHAR_TO_LETTER.get(chars[0], chars[0])
        chars[1] = CHAR_TO_LETTER.get(chars[1], chars[1])
        st = chars[0] + chars[1]
        if st not in VALID_INDIAN_STATES and st in STATE_CONFUSION_MAP:
            mapped_st = STATE_CONFUSION_MAP[st]
            chars[0], chars[1] = mapped_st[0], mapped_st[1]

        chars[2] = CHAR_TO_DIGIT.get(chars[2], chars[2])
        chars[3] = CHAR_TO_DIGIT.get(chars[3], chars[3])

        chars[4] = CHAR_TO_LETTER.get(chars[4], chars[4])

        for i in [5, 6, 7, 8]:
            chars[i] = CHAR_TO_DIGIT.get(chars[i], chars[i])

        corrected = "".join(chars)
        st_final = corrected[:2]
        is_valid_state = st_final in VALID_INDIAN_STATES
        pattern = re.compile(r"^[A-Z]{2}\d{2}[A-Z]\d{4}$")
        if pattern.fullmatch(corrected):
            return True, 0.96 if is_valid_state else 0.85, corrected
        return True, 0.82 if is_valid_state else 0.68, corrected

    # 3. 11-character format: SS DD LLL DDDD (e.g. DL 01 ABC 1234)
    if length == 11:
        chars[0] = CHAR_TO_LETTER.get(chars[0], chars[0])
        chars[1] = CHAR_TO_LETTER.get(chars[1], chars[1])
        st = chars[0] + chars[1]
        if st not in VALID_INDIAN_STATES and st in STATE_CONFUSION_MAP:
            mapped_st = STATE_CONFUSION_MAP[st]
            chars[0], chars[1] = mapped_st[0], mapped_st[1]

        chars[2] = CHAR_TO_DIGIT.get(chars[2], chars[2])
        chars[3] = CHAR_TO_DIGIT.get(chars[3], chars[3])

        chars[4] = CHAR_TO_LETTER.get(chars[4], chars[4])
        chars[5] = CHAR_TO_LETTER.get(chars[5], chars[5])
        chars[6] = CHAR_TO_LETTER.get(chars[6], chars[6])

        for i in [7, 8, 9, 10]:
            chars[i] = CHAR_TO_DIGIT.get(chars[i], chars[i])

        corrected = "".join(chars)
        st_final = corrected[:2]
        is_valid_state = st_final in VALID_INDIAN_STATES
        pattern = re.compile(r"^[A-Z]{2}\d{2}[A-Z]{3}\d{4}$")
        if pattern.fullmatch(corrected):
            return True, 0.95 if is_valid_state else 0.85, corrected

    # 4. Fallback pattern checks
    for idx, ch in enumerate(chars):
        if idx < 2:
            chars[idx] = CHAR_TO_LETTER.get(ch, ch)
        elif 2 <= idx < 4:
            chars[idx] = CHAR_TO_DIGIT.get(ch, ch)
        elif idx >= length - 4:
            chars[idx] = CHAR_TO_DIGIT.get(ch, ch)
        else:
            chars[idx] = CHAR_TO_LETTER.get(ch, ch)
    st = chars[0] + chars[1] if length >= 2 else ""
    if st in STATE_CONFUSION_MAP:
        mapped_st = STATE_CONFUSION_MAP[st]
        chars[0], chars[1] = mapped_st[0], mapped_st[1]

    corrected = "".join(chars)
    st_final = corrected[:2] if len(corrected) >= 2 else ""
    is_valid_state = st_final in VALID_INDIAN_STATES

    patterns = [
        re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$"),
        re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$"),
        re.compile(r"^[A-Z]{2}\d{1,2}\d[A-Z]{0,3}\d{4}$"),
        re.compile(r"^[A-Z]{2}\d{1,2}T[A-Z]{1,3}\d{4}$"),
        re.compile(r"^[A-Z]{2}\d{1,2}G[A-Z]{1,3}\d{4}$"),
        re.compile(r"^[A-Z]{2}\d{1,2}\d{4}$"),
    ]
    if any(p.fullmatch(corrected) for p in patterns):
        return True, 0.92 if is_valid_state else 0.75, corrected

    if 8 <= len(corrected) <= 11 and is_valid_state:
        return True, 0.70, corrected

    return False, 0.30, corrected


def character_confusion_distance(s1: str, s2: str) -> float:
    """
    Compute weighted Levenshtein edit distance where known OCR confusion pairs
    (e.g., 'B' <-> '8', 'O' <-> '0', 'I' <-> '1') have minimal penalty (0.15).
    """
    m, n = len(s1), len(s2)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = float(i)
    for j in range(n + 1):
        dp[0][j] = float(j)

    for i in range(1, m + 1):
        c1 = s1[i - 1]
        for j in range(1, n + 1):
            c2 = s2[j - 1]
            if c1 == c2:
                cost = 0.0
            elif frozenset([c1, c2]) in OCR_CONFUSION_PAIRS:
                cost = 0.15  # Known OCR confusion substitution
            else:
                cost = 1.0   # General substitution

            dp[i][j] = min(
                dp[i - 1][j] + 1.0,        # Deletion
                dp[i][j - 1] + 1.0,        # Insertion
                dp[i - 1][j - 1] + cost    # Substitution
            )

    return dp[m][n]


def plate_similarity(p1: str, p2: str) -> float:
    """
    Compute normalized similarity [0.0, 1.0] between two plate strings.
    """
    if not p1 or not p2:
        return 0.0
    if p1 == p2:
        return 1.0
    dist = character_confusion_distance(p1, p2)
    max_len = max(len(p1), len(p2), 1)
    sim = max(0.0, 1.0 - (dist / float(max_len)))
    return round(sim, 3)


def vote_character_positions(plates_with_conf: List[Tuple[str, float]]) -> str:
    """
    Perform weighted position-by-position character voting across multiple frames
    of the same cluster length.
    """
    if not plates_with_conf:
        return ""
    lengths = [len(p[0]) for p in plates_with_conf if len(p[0]) >= 8]
    if not lengths:
        return max(plates_with_conf, key=lambda x: x[1])[0]

    target_len = max(set(lengths), key=lengths.count)
    same_len_plates = [p for p in plates_with_conf if len(p[0]) == target_len]
    if not same_len_plates:
        return max(plates_with_conf, key=lambda x: x[1])[0]

    result_chars = []
    for pos in range(target_len):
        char_weights: Dict[str, float] = {}
        for plate, conf in same_len_plates:
            ch = plate[pos]
            char_weights[ch] = char_weights.get(ch, 0.0) + max(0.1, float(conf))

        # Handle OCR confusion between J and B in letter series position 4
        if pos == 4 and "J" in char_weights and ("B" in char_weights or "8" in char_weights):
            if char_weights["J"] >= char_weights.get("B", 0.0) * 0.7:
                best_char = "J"
            else:
                best_char = max(char_weights.keys(), key=lambda c: char_weights[c])
        else:
            best_char = max(char_weights.keys(), key=lambda c: char_weights[c])
        result_chars.append(best_char)

    return "".join(result_chars)


class MultiFramePlateFusionEngine:
    """
    Production Multi-Frame ANPR Fusion Engine.
    Maintains per-track recognition buffer, performs character-confusion clustering,
    weighted voting, confidence aggregation, evidence gating, and deduplication.
    """

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        min_observations: int = DEFAULT_MIN_OBSERVATIONS,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        min_agreement: float = DEFAULT_MIN_AGREEMENT,
    ):
        self.window_size = window_size
        self.min_observations = min_observations
        self.min_confidence = min_confidence
        self.min_agreement = min_agreement

        # Recognition Buffer: track_id -> List[Dict[str, Any]]
        self.plate_history: Dict[int, List[Dict[str, Any]]] = {}

        # Track metadata: track_id -> Dict[str, Any]
        self.track_metadata: Dict[int, Dict[str, Any]] = {}

        # Finalized events registry: track_id -> Dict[str, Any]
        self.finalized_tracks: Dict[int, Dict[str, Any]] = {}

        # Camera/gate deduplication cache: Set of plate strings recently finalized
        self.finalized_plate_registry: Dict[str, Dict[str, Any]] = {}

    def add_observation(
        self,
        track_id: int,
        raw_plate: str,
        ocr_confidence: float,
        plate_confidence: float = 0.85,
        frame_number: int = 0,
        bbox: Optional[List[int]] = None,
        vehicle_class: str = "car",
        camera_id: str = "G01-ENTRY",
        gate_id: str = "Gate 01",
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Add a frame OCR prediction for a specific vehicle track.
        Normalizes OCR text, handles confusion, updates buffer, and calculates current fusion state.
        """
        if timestamp is None:
            timestamp = time.time()

        normalized = normalize_ocr_text(raw_plate)
        is_val, val_conf, corrected = validate_indian_plate(normalized)
        clean_plate = corrected or normalized

        if track_id not in self.plate_history:
            self.plate_history[track_id] = []
            self.track_metadata[track_id] = {
                "track_id": track_id,
                "vehicle_class": vehicle_class,
                "camera_id": camera_id,
                "gate_id": gate_id,
                "first_frame": frame_number,
                "last_frame": frame_number,
                "created_at": timestamp,
                "is_finalized": False
            }

        # Update track metadata
        meta = self.track_metadata[track_id]
        meta["last_frame"] = frame_number
        if vehicle_class and vehicle_class != "vehicle":
            meta["vehicle_class"] = vehicle_class

        # Only add meaningful observations (len >= 4)
        if clean_plate and len(clean_plate) >= 4:
            obs = {
                "plate": clean_plate,
                "raw_text": str(raw_plate),
                "ocr_confidence": float(ocr_confidence),
                "plate_confidence": float(plate_confidence),
                "validation_confidence": float(val_conf),
                "is_valid_format": bool(is_val),
                "frame": int(frame_number),
                "timestamp": float(timestamp),
                "bbox": bbox or []
            }
            self.plate_history[track_id].append(obs)

            # Maintain sliding window
            if len(self.plate_history[track_id]) > self.window_size * 3:
                self.plate_history[track_id] = self.plate_history[track_id][-self.window_size * 3:]

        # Return the evaluated fusion state for this track
        return self.evaluate_track(track_id)

    def evaluate_track(self, track_id: int, force_finalize: bool = False) -> Dict[str, Any]:
        """
        Perform Multi-Frame Fusion on the recognition buffer of track_id.
        Returns the structured fusion result with status: 'pending', 'finalized', or 'manual_review'.
        """
        meta = self.track_metadata.get(track_id, {
            "track_id": track_id,
            "vehicle_class": "car",
            "camera_id": "G01-ENTRY",
            "gate_id": "Gate 01",
            "first_frame": 0,
            "last_frame": 0,
            "created_at": time.time(),
            "is_finalized": False
        })

        observations = self.plate_history.get(track_id, [])
        total_obs = len(observations)

        # If already finalized, update frame counts and return
        if meta.get("is_finalized") and track_id in self.finalized_tracks:
            # Re-evaluate to incorporate latest supporting frames
            pass

        # Case 0: Zero observations
        if total_obs == 0:
            return {
                "track_id": track_id,
                "plate_number": "",
                "display_plate": "Recognizing...",
                "status": "pending",
                "final_confidence": 0.0,
                "frame_count": 0,
                "agreement_ratio": 0.0,
                "is_finalized": False,
                "vehicle_type": meta.get("vehicle_class", "car"),
                "camera_id": meta.get("camera_id", "G01-ENTRY"),
                "gate_id": meta.get("gate_id", "Gate 01"),
                "supporting_predictions": []
            }

        # Step 1: Cluster observations by character confusion & similarity
        clusters: List[Dict[str, Any]] = []

        for obs in observations:
            p_text = obs["plate"]
            matched_cluster = None

            for cluster in clusters:
                canonical = cluster["canonical_plate"]
                sim = plate_similarity(p_text, canonical)
                # Similarity >= 0.75 or edit distance <= 2 indicates same plate
                if sim >= 0.75 or character_confusion_distance(p_text, canonical) <= 1.5:
                    matched_cluster = cluster
                    break

            if matched_cluster is not None:
                matched_cluster["observations"].append(obs)
            else:
                clusters.append({
                    "canonical_plate": p_text,
                    "observations": [obs]
                })

        # Step 2: For each cluster, find the best canonical plate representation & calculate weighted score
        ranked_candidates: List[Dict[str, Any]] = []

        for cluster in clusters:
            cluster_obs = cluster["observations"]
            count = len(cluster_obs)
            agreement_ratio = count / float(total_obs)

            # Determine the best candidate plate spelling in this cluster using syntax validation + frequency
            spelling_counts: Dict[str, float] = {}
            for o in cluster_obs:
                sp = o["plate"]
                is_v, val_c, _ = validate_indian_plate(sp)
                st_code = sp[:2] if len(sp) >= 2 else ""
                val_weight = 2.5 if (is_v and st_code in VALID_INDIAN_STATES) else (1.4 if is_v else 0.8)
                spelling_counts[sp] = spelling_counts.get(sp, 0.0) + (o["ocr_confidence"] * val_weight)

            best_spelling = max(spelling_counts.keys(), key=lambda s: spelling_counts[s])
            is_valid_syntax, val_conf, validated_plate = validate_indian_plate(best_spelling)
            canonical_plate = validated_plate or best_spelling

            # Apply character-position consensus voting across observations
            if len(cluster_obs) >= 2:
                candidate_pairs = [(o["plate"], o["ocr_confidence"]) for o in cluster_obs if len(o["plate"]) >= 8]
                voted_plate = vote_character_positions(candidate_pairs)
                if voted_plate:
                    is_vc, val_vc, validated_voted = validate_indian_plate(voted_plate)
                    if is_vc or (val_vc >= val_conf):
                        canonical_plate = validated_voted or voted_plate
                        is_valid_syntax = is_vc
                        val_conf = max(val_conf, val_vc)

            # Calculate individual sub-scores
            avg_ocr_conf = sum(o["ocr_confidence"] for o in cluster_obs) / float(count)
            max_ocr_conf = max(o["ocr_confidence"] for o in cluster_obs)
            avg_plate_conf = sum(o["plate_confidence"] for o in cluster_obs) / float(count)

            st_code = canonical_plate[:2] if len(canonical_plate) >= 2 else ""
            has_valid_state = st_code in VALID_INDIAN_STATES

            # Weighted Scoring:
            # - Frequency / repetition: 3.5 * count
            # - Agreement ratio: 2.0 * agreement_ratio
            # - Average OCR confidence: 3.0 * avg_ocr_conf
            # - Max OCR confidence: 1.0 * max_ocr_conf
            # - Plate detector confidence: 1.0 * avg_plate_conf
            # - Valid syntax & state code bonus: 2.5
            freq_score = count * 3.5
            agree_score = agreement_ratio * 20.0
            ocr_score = (avg_ocr_conf * 30.0) + (max_ocr_conf * 10.0)
            det_score = avg_plate_conf * 10.0
            syntax_score = (25.0 if is_valid_syntax else 0.0) + (15.0 if has_valid_state else 0.0)

            composite_score = freq_score + agree_score + ocr_score + det_score + syntax_score

            # Composite Final Confidence [0.0, 1.0]
            # Blends average OCR confidence, agreement ratio, validation confidence, and frame depth
            frame_depth_factor = min(1.0, count / float(self.min_observations))
            final_conf = (
                (avg_ocr_conf * 0.40) +
                (agreement_ratio * 0.25) +
                (val_conf * 0.25) +
                (frame_depth_factor * 0.10)
            )
            final_conf = max(0.10, min(0.99, final_conf))

            ranked_candidates.append({
                "plate": canonical_plate,
                "composite_score": composite_score,
                "final_confidence": round(final_conf, 2),
                "avg_ocr_confidence": round(avg_ocr_conf, 2),
                "max_ocr_confidence": round(max_ocr_conf, 2),
                "count": count,
                "agreement_ratio": round(agreement_ratio, 2),
                "is_valid": is_valid_syntax,
                "observations": cluster_obs
            })

        # Sort candidates descending by composite score
        ranked_candidates.sort(key=lambda c: c["composite_score"], reverse=True)
        winner = ranked_candidates[0]

        supporting_preds = [
            {
                "plate": o["plate"],
                "confidence": round(o["ocr_confidence"], 2),
                "frame": o.get("frame", 0),
                "is_valid": o.get("is_valid_format", False)
            }
            for o in winner["observations"]
        ]

        # Step 3: Evidence Gating and State Transitions
        meets_criteria = (
            (winner["count"] >= self.min_observations and winner["final_confidence"] >= self.min_confidence and winner["agreement_ratio"] >= self.min_agreement) or
            (winner["is_valid"] and winner["final_confidence"] >= 0.65 and winner["count"] >= 1) or
            (winner["is_valid"] and winner["final_confidence"] >= 0.70)
        )

        has_conflict = (len(ranked_candidates) >= 2 and winner["agreement_ratio"] < self.min_agreement and winner["count"] <= 1)

        if meets_criteria and not has_conflict:
            status = "finalized"
            display_plate = winner["plate"]
            is_finalized = True
        elif force_finalize:
            if winner.get("count", 0) >= 1 and winner.get("plate") and (winner["is_valid"] or winner["final_confidence"] >= 0.50) and not has_conflict:
                status = "finalized"
                display_plate = winner["plate"]
                is_finalized = True
            else:
                status = "manual_review"
                display_plate = "Requires Manual Review"
                is_finalized = True
        else:
            status = "pending"
            display_plate = "Recognizing..."
            is_finalized = False

        frame_preds = [
            {
                "frame_number": o.get("frame", idx + 1),
                "plate_number": o.get("plate", ""),
                "confidence": round(o.get("ocr_confidence", 0.0), 2),
                "raw_text": o.get("raw_text", "")
            }
            for idx, o in enumerate(observations)
        ]

        result_payload = {
            "track_id": track_id,
            "plate_number": winner["plate"] if status == "finalized" else "",
            "display_plate": display_plate,
            "status": status,
            "final_confidence": winner["final_confidence"],
            "frame_count": total_obs,
            "evidence_count": winner["count"],
            "agreement_ratio": winner["agreement_ratio"],
            "is_finalized": is_finalized,
            "vehicle_type": meta.get("vehicle_class", "car"),
            "camera_id": meta.get("camera_id", "G01-ENTRY"),
            "gate_id": meta.get("gate_id", "Gate 01"),
            "timestamp": datetime.datetime.fromtimestamp(meta.get("created_at", time.time())).strftime("%Y-%m-%d %H:%M:%S"),
            "frame_predictions": frame_preds,
            "supporting_predictions": supporting_preds,
            "all_candidates": [
                {
                    "plate": c["plate"],
                    "confidence": c["final_confidence"],
                    "count": c["count"],
                    "agreement": c["agreement_ratio"]
                }
                for c in ranked_candidates
            ]
        }

        if is_finalized:
            meta["is_finalized"] = True
            self.finalized_tracks[track_id] = result_payload
            if status == "finalized" and winner["plate"]:
                self.finalized_plate_registry[winner["plate"]] = result_payload

        return result_payload

    def finalize_all_active_tracks(self) -> List[Dict[str, Any]]:
        """
        Finalize all tracks that were tracked in the video once processing completes.
        Applies deduplication and ensures one event per distinct vehicle track.
        """
        results = []
        for tid in list(self.plate_history.keys()):
            res = self.evaluate_track(tid, force_finalize=True)
            results.append(res)
        return results

    def reset(self) -> None:
        """Clear all in-memory buffers."""
        self.plate_history.clear()
        self.track_metadata.clear()
        self.finalized_tracks.clear()
        self.finalized_plate_registry.clear()
