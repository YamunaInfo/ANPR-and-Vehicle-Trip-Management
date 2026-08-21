import { IPlateValidator, ValidationResult } from '../core/interfaces';

export class IndianPlateValidator implements IPlateValidator {
  // ─────────────────────────────────────────────────────────────
  // Indian State / Union Territory codes (2 letters)
  // ─────────────────────────────────────────────────────────────
  private validStates = new Set([
    // Union Territories
    'AN', // Andaman and Nicobar Islands
    'CH', // Chandigarh
    'DD', // Daman and Diu
    'DL', // Delhi
    'LD', // Lakshadweep
    'PY', // Puducherry
    'LA', // Ladakh
    'DH', // Dadra and Nagar Haveli

    // States
    'AP', // Andhra Pradesh (old)
    'AR', // Arunachal Pradesh
    'AS', // Assam
    'BR', // Bihar
    'CG', // Chhattisgarh
    'GA', // Goa
    'GJ', // Gujarat
    'HR', // Haryana
    'HP', // Himachal Pradesh
    'JK', // Jammu and Kashmir
    'JH', // Jharkhand
    'KA', // Karnataka
    'KL', // Kerala
    'MP', // Madhya Pradesh
    'MH', // Maharashtra
    'MN', // Manipur
    'ML', // Meghalaya
    'MZ', // Mizoram
    'NL', // Nagaland
    'OD', // Odisha
    'PB', // Punjab
    'RJ', // Rajasthan
    'SK', // Sikkim
    'TN', // Tamil Nadu
    'TS', // Telangana
    'TR', // Tripura
    'UP', // Uttar Pradesh
    'UK', // Uttarakhand
    'WB', // West Bengal

    // Special
    'BH'  // Bharat Series (Electric/Government)
  ]);

  private stateConfusionMap: Record<string, string> = {
    'HH': 'MH', 'NH': 'MH', 'KH': 'MH', 'VH': 'MH', 'WH': 'MH', 'MM': 'MH',
    'MI': 'MH', 'MN': 'MH', 'MR': 'MH', 'HM': 'MH', 'TH': 'MH', 'WW': 'MH',
    '0L': 'DL', 'OL': 'DL', '1L': 'DL', 'IL': 'DL', 'QL': 'DL', 'DI': 'DL', 'D1': 'DL',
    'HA': 'HR', 'HB': 'HR', 'HD': 'HR',
    'KB': 'KA', 'KP': 'KL', 'G3': 'GJ', 'R3': 'RJ', 'W8': 'WB',
    'MB': 'MP', 'MD': 'MP', 'UB': 'UP', 'UF': 'UP', 'TJ': 'TN', 'TM': 'TN'
  };

  // ─────────────────────────────────────────────────────────────
  // STEP 7: INDIAN PLATE FORMAT PATTERNS
  // ─────────────────────────────────────────────────────────────

  // Standard format: STATE(2L) RTO(2D) SERIES(1-3L) REG_NUM(4D)
  // Example: TN 38 AB 1234 (without spaces: TN38AB1234)
  private strictStandard = /^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$/;

  // Bharat/Electric Series: (2D) BH (4D) (1-2L)
  // Example: 22 BH 1234 A (without spaces: 22BH1234A)
  private strictBharat = /^\d{2}BH\d{4}[A-Z]{1,2}$/;

  // Commercial format: STATE(2L) RTO(1-2D) COM(1D) SERIES(1-3L) REG_NUM(4D)
  // Example: TN 38 1 AB 1234 (may be missing series letters)
  private strictCommercial = /^[A-Z]{2}\d{1,2}\d[A-Z]{0,3}\d{4}$/;

  // Temporary format
  private strictTemporary = /^[A-Z]{2}\d{1,2}T[A-Z]{1,3}\d{4}$/;

  // Government format (with IND badge)
  private strictGovernment = /^[A-Z]{2}\d{1,2}G[A-Z]{1,3}\d{4}$/;

  // Old short format: STATE(2L) RTO(2D) REG_NUM(4D)
  // Example: TN 38 1234 (without spaces: TN381234)
  private strictShort = /^[A-Z]{2}\d{1,2}\d{4}$/;

  isValidFormat(plate: string): boolean {
    const v = this.validate(plate);
    return v.isValid;
  }

  normalize(plate: string): string {
    const v = this.validate(plate);
    return v.normalizedPlate;
  }

  validate(rawPlate: string): ValidationResult {
    let cleaned = (rawPlate || '').toUpperCase().replace(/[^A-Z0-9]/g, '');

    if (!cleaned || cleaned.length < 4 || cleaned.length > 15) {
      return {
        isValid: false,
        normalizedPlate: cleaned,
        validationStatus: 'LOW_CONFIDENCE',
        validationConfidence: 0.0,
      };
    }

    // Strip leading 'IND' or 'IN'
    if (cleaned.startsWith('IND') && cleaned.length >= 7) {
      cleaned = cleaned.substring(3);
    } else if (cleaned.startsWith('IN') && cleaned.length >= 8 && (this.validStates.has(cleaned.substring(2, 4)) || this.stateConfusionMap[cleaned.substring(2, 4)])) {
      cleaned = cleaned.substring(2);
    } else if (cleaned.length >= 11 && ['1', 'I', 'L', 'T'].includes(cleaned[0]) && (this.validStates.has(cleaned.substring(1, 3)) || this.stateConfusionMap[cleaned.substring(1, 3)])) {
      cleaned = cleaned.substring(1);
    }

    // Remove trailing border artifacts (e.g. trailing 1, I, L, 7 after 4-digit number)
    const mArtifact = cleaned.match(/^([A-Z0-9]{2}[A-Z0-9]{2}[A-Z0-9]{1,3}[A-Z0-9]{4})[1IL7TI]$/);
    if (mArtifact) {
      cleaned = mArtifact[1];
    } else {
      const m2 = cleaned.match(/^([A-Z0-9]{2}\d{1,2}[A-Z0-9]{1,3}\d{4})\d$/);
      if (m2) {
        cleaned = m2[1];
      }
    }

    // ─────────────────────────────────────────────────────────────
    // STEP 7a: Check patterns without corrections first
    // ─────────────────────────────────────────────────────────────

    if (this.strictStandard.test(cleaned)) {
      const state = cleaned.substring(0, 2);
      const isKnownState = this.validStates.has(state);
      return {
        isValid: true,
        normalizedPlate: cleaned,
        validationStatus: 'VALID',
        validationConfidence: isKnownState ? 1.0 : 0.90,
        patternMatch: 'STANDARD_STRICT',
        correctionsApplied: 0
      };
    }

    if (this.strictBharat.test(cleaned)) {
      return {
        isValid: true,
        normalizedPlate: cleaned,
        validationStatus: 'VALID',
        validationConfidence: 0.98,
        patternMatch: 'BHARAT_SERIES_STRICT',
        correctionsApplied: 0
      };
    }

    if (this.strictCommercial.test(cleaned)) {
      const state = cleaned.substring(0, 2);
      const isKnownState = this.validStates.has(state);
      return {
        isValid: true,
        normalizedPlate: cleaned,
        validationStatus: 'VALID',
        validationConfidence: isKnownState ? 0.95 : 0.85,
        patternMatch: 'COMMERCIAL_STRICT',
        correctionsApplied: 0
      };
    }

    if (this.strictTemporary.test(cleaned)) {
      return {
        isValid: true,
        normalizedPlate: cleaned,
        validationStatus: 'VALID',
        validationConfidence: 0.92,
        patternMatch: 'TEMPORARY_STRICT',
        correctionsApplied: 0
      };
    }

    if (this.strictGovernment.test(cleaned)) {
      return {
        isValid: true,
        normalizedPlate: cleaned,
        validationStatus: 'VALID',
        validationConfidence: 0.95,
        patternMatch: 'GOVERNMENT_STRICT',
        correctionsApplied: 0
      };
    }

    // ─────────────────────────────────────────────────────────────
    // STEP 8: POSITION-AWARE CHARACTER CORRECTION
    // ─────────────────────────────────────────────────────────────

    const { correctedPlate, corrections } = this.correctPositionalConfusions(cleaned);

    // Re-test patterns with corrections
    if (this.strictStandard.test(correctedPlate)) {
      const state = correctedPlate.substring(0, 2);
      const isKnownState = this.validStates.has(state);
      return {
        isValid: true,
        normalizedPlate: correctedPlate,
        validationStatus: 'VALID',
        validationConfidence: isKnownState ? 0.88 : 0.75,
        patternMatch: 'STANDARD_POSITION_CORRECTED',
        correctionsApplied: corrections
      };
    }

    if (this.strictBharat.test(correctedPlate)) {
      return {
        isValid: true,
        normalizedPlate: correctedPlate,
        validationStatus: 'VALID',
        validationConfidence: 0.85,
        patternMatch: 'BHARAT_CORRECTED',
        correctionsApplied: corrections
      };
    }

    if (this.strictCommercial.test(correctedPlate)) {
      const state = correctedPlate.substring(0, 2);
      const isKnownState = this.validStates.has(state);
      return {
        isValid: true,
        normalizedPlate: correctedPlate,
        validationStatus: 'VALID',
        validationConfidence: isKnownState ? 0.82 : 0.70,
        patternMatch: 'COMMERCIAL_CORRECTED',
        correctionsApplied: corrections
      };
    }

    if (this.strictTemporary.test(correctedPlate)) {
      return {
        isValid: true,
        normalizedPlate: correctedPlate,
        validationStatus: 'VALID',
        validationConfidence: 0.80,
        patternMatch: 'TEMPORARY_CORRECTED',
        correctionsApplied: corrections
      };
    }

    if (this.strictGovernment.test(correctedPlate)) {
      return {
        isValid: true,
        normalizedPlate: correctedPlate,
        validationStatus: 'VALID',
        validationConfidence: 0.82,
        patternMatch: 'GOVERNMENT_CORRECTED',
        correctionsApplied: corrections
      };
    }

    if (this.strictShort.test(correctedPlate)) {
      const state = correctedPlate.substring(0, 2);
      if (this.validStates.has(state)) {
        return {
          isValid: true,
          normalizedPlate: correctedPlate,
          validationStatus: 'VALID',
          validationConfidence: 0.80,
          patternMatch: 'SHORT_FORMAT_CORRECTED',
          correctionsApplied: corrections
        };
      }
    }

    // Fallback: Check if state code matches at all
    if (cleaned.length >= 8 && cleaned.length <= 11) {
      const first2 = this.numToLet(cleaned.substring(0, 2));
      if (this.validStates.has(first2)) {
        return {
          isValid: false,
          normalizedPlate: correctedPlate,
          validationStatus: 'INVALID',
          validationConfidence: 0.45,
          patternMatch: 'STATE_CODE_MATCHED_ONLY',
          correctionsApplied: corrections
        };
      }
    }

    return {
      isValid: false,
      normalizedPlate: cleaned,
      validationStatus: 'INVALID',
      validationConfidence: 0.10,
      correctionsApplied: 0
    };
  }

  // ─────────────────────────────────────────────────────────────
  // STEP 8: POSITION-AWARE CHARACTER CORRECTION RULES
  // ─────────────────────────────────────────────────────────────

  private correctPositionalConfusions(plate: string): { correctedPlate: string; corrections: number } {
    let chars = plate.split('');
    let corrections = 0;
    const len = chars.length;

    // Standard Indian plate format:
    // Position 0-1: State Code (Letters)
    // Position 2-3: RTO Code (Digits)
    // Position 4 to len-4: Series Code (Letters, 1-3 chars)
    // Position len-4 to len: Registration Number (Digits, 4 chars)

    if (len >= 8 && len <= 11) {
      // ───────────────────────────────────────────────────────
      // State Code (Position 0-1): MUST be LETTERS
      // ───────────────────────────────────────────────────────
      for (let i = 0; i < 2; i++) {
        const c = chars[i];
        if (this.stateCodeCharMap[c]) {
          chars[i] = this.stateCodeCharMap[c];
          corrections++;
        }
      }
      const st = chars[0] + chars[1];
      if (!this.validStates.has(st) && this.stateConfusionMap[st]) {
        chars[0] = this.stateConfusionMap[st][0];
        chars[1] = this.stateConfusionMap[st][1];
        corrections++;
      }

      // ───────────────────────────────────────────────────────
      // RTO Code (Position 2-3): MUST be DIGITS
      // ───────────────────────────────────────────────────────
      for (let i = 2; i < 4; i++) {
        const c = chars[i];
        if (this.rtoDigitCharMap[c]) {
          chars[i] = this.rtoDigitCharMap[c];
          corrections++;
        }
      }

      // ───────────────────────────────────────────────────────
      // Registration Number (Last 4 positions): MUST be DIGITS
      // ───────────────────────────────────────────────────────
      for (let i = Math.max(4, len - 4); i < len; i++) {
        const c = chars[i];
        if (this.regNumberCharMap[c]) {
          chars[i] = this.regNumberCharMap[c];
          corrections++;
        }
      }

      // ───────────────────────────────────────────────────────
      // Series Code (Between RTO and Reg Number): MUST be LETTERS
      // ───────────────────────────────────────────────────────
      for (let i = 4; i < len - 4; i++) {
        const c = chars[i];
        if (this.seriesCharMap[c]) {
          chars[i] = this.seriesCharMap[c];
          corrections++;
        }
      }
    }

    return {
      correctedPlate: chars.join(''),
      corrections
    };
  }

  private numToLet(str: string): string {
    return str.split('').map(c => this.stateCodeCharMap[c] || c).join('');
  }

  // ─────────────────────────────────────────────────────────────
  // CHARACTER MAPPING FOR EACH POSITION
  // ─────────────────────────────────────────────────────────────

  // State Code position (0-1): Convert digits to letters
  private stateCodeCharMap: Record<string, string> = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '5': 'S',
    '8': 'B',
    '6': 'G',
    '7': 'T'
  };

  // RTO Code position (2-3): Convert letters to digits
  private rtoDigitCharMap: Record<string, string> = {
    'O': '0',
    'Q': '0',
    'D': '0',
    'I': '1',
    'L': '1',
    'Z': '2',
    'S': '5',
    'B': '8',
    'G': '6',
    'T': '7'
  };

  // Series position (4 to len-4): Convert digits to letters
  private seriesCharMap: Record<string, string> = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '5': 'S',
    '8': 'B',
    '6': 'G',
    '7': 'T'
  };

  // Registration Number position (len-4 to len): Convert letters to digits
  private regNumberCharMap: Record<string, string> = {
    'O': '0',
    'Q': '0',
    'D': '0',
    'I': '1',
    'L': '1',
    'Z': '2',
    'S': '5',
    'B': '8',
    'G': '6',
    'T': '7'
  };
}

