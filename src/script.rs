//! Character -> unit mapping for the nine ISCII-aligned Brahmic blocks.
//!
//! Unicode lays out Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil,
//! Telugu, Kannada and Malayalam as contiguous 128-codepoint blocks starting
//! at U+0900, with the same phoneme at the same offset in every block
//! (क U+0915, ক U+0995, ક U+0A95, க U+0B95, ಕ U+0C95 are all offset 0x15).
//!
//! The tokenizer works on a sequence of `u32` *units* instead of bytes:
//!
//! | units          | meaning                                                   |
//! |----------------|-----------------------------------------------------------|
//! | `0..256`       | raw UTF-8 byte                                            |
//! | `256..384`     | shared phoneme = block offset, script-agnostic            |
//! | `384..1536`    | script-specific character = `cp - 0x0900`                 |
//!
//! Token ids: bytes and shared units keep their unit number, followed by one
//! script-switch marker per script, then script-specific units and merges.
//!
//! This file is the ONLY source file that differs between AksharaG2P,
//! VanillaBPE-Char and VanillaBPE-Byte; the differences are confined to the
//! VARIANT section below (and the variant-specific tests at the bottom).

// ============================================================================
// VARIANT: VanillaBPE-Char — every Indic character is its own unit
// (क and ক are different units; no shared phonemes, no script markers)
// ============================================================================

pub const VARIANT: &str = "vanillabpe_char";

/// Number of shared phoneme tokens (ids `256..256 + N_SHARED`).
pub const N_SHARED: u32 = 0;
/// Number of script-switch marker tokens.
pub const N_MARKERS: u32 = 0;

/// No offset is shared across scripts.
#[inline]
pub fn is_shared(_script: usize, _offset: u32) -> bool {
    false
}

/// The unit for a character, or `None` to encode it as UTF-8 bytes.
#[inline]
pub fn char_unit(c: char) -> Option<u32> {
    brahmic(c).map(|(s, off)| specific_unit(s, off))
}

// ============================================================================
// end of VARIANT section
// ============================================================================

pub const N_SCRIPTS: usize = 9;
pub const SCRIPT_NAMES: [&str; N_SCRIPTS] = [
    "deva", "beng", "guru", "gujr", "orya", "taml", "telu", "knda", "mlym",
];

pub const DEVA: usize = 0;
pub const BENG: usize = 1;
pub const GURU: usize = 2;
pub const ORYA: usize = 4;
pub const MLYM: usize = 8;

const BLOCK_START: u32 = 0x0900;
const BLOCK_END: u32 = BLOCK_START + (N_SCRIPTS as u32) * 128;

pub const SHARED_BASE: u32 = 256;
pub const SPECIFIC_BASE: u32 = SHARED_BASE + 128;
pub const N_UNITS: u32 = SPECIFIC_BASE + (N_SCRIPTS as u32) * 128;

/// First script-marker token id; `0..MARKER_BASE` are bytes and shared units.
pub const MARKER_BASE: u32 = SHARED_BASE + N_SHARED;
/// Number of always-present tokens (bytes, shared units, markers).
pub const N_FIXED_TOKENS: u32 = MARKER_BASE + N_MARKERS;

/// Script number and block offset of a Brahmic codepoint.
#[inline]
pub fn brahmic(c: char) -> Option<(usize, u32)> {
    let cp = c as u32;
    if (BLOCK_START..BLOCK_END).contains(&cp) {
        let rel = cp - BLOCK_START;
        Some(((rel >> 7) as usize, rel & 0x7F))
    } else {
        None
    }
}

#[inline]
pub fn specific_unit(script: usize, offset: u32) -> u32 {
    SPECIFIC_BASE + (script as u32) * 128 + offset
}

#[inline]
pub fn is_shared_unit(unit: u32) -> bool {
    (SHARED_BASE..SHARED_BASE + N_SHARED).contains(&unit)
}

#[inline]
pub fn is_specific_unit(unit: u32) -> bool {
    (SPECIFIC_BASE..N_UNITS).contains(&unit)
}

/// The character a non-byte unit stands for, rendering shared units in `script`.
#[inline]
pub fn unit_char(unit: u32, script: usize) -> char {
    let cp = if unit < SPECIFIC_BASE {
        BLOCK_START + (script as u32) * 128 + (unit - SHARED_BASE)
    } else {
        BLOCK_START + (unit - SPECIFIC_BASE)
    };
    // every codepoint in U+0900..U+0D80 is a valid scalar value
    char::from_u32(cp).unwrap()
}

/// Appends the UTF-8 rendering of `units` to `out`.
#[inline]
pub fn render_units(units: &[u32], script: usize, out: &mut Vec<u8>) {
    let mut buf = [0u8; 4];
    for &u in units {
        if u < SHARED_BASE {
            out.push(u as u8);
        } else {
            out.extend_from_slice(unit_char(u, script).encode_utf8(&mut buf).as_bytes());
        }
    }
}

/// Splits `piece` wherever the Brahmic script changes (identically in every
/// variant) and calls `f(sub_piece, marker_script)`, where `marker_script` is
/// the run's script if it contains shared units that need a script marker.
pub fn split_by_script<'a>(piece: &'a str, mut f: impl FnMut(&'a str, Option<usize>)) {
    let mut start = 0;
    let mut run_script: Option<usize> = None;
    let mut run_shared = false;
    for (i, c) in piece.char_indices() {
        if let Some((s, _)) = brahmic(c) {
            match run_script {
                Some(t) if t != s => {
                    f(&piece[start..i], run_script.filter(|_| run_shared));
                    start = i;
                    run_shared = false;
                    run_script = Some(s);
                }
                None => run_script = Some(s),
                _ => {}
            }
            run_shared |= char_unit(c).is_some_and(is_shared_unit);
        }
    }
    if start < piece.len() {
        f(&piece[start..], run_script.filter(|_| run_shared));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_block_codepoint_round_trips() {
        for cp in BLOCK_START..BLOCK_END {
            let c = char::from_u32(cp).unwrap();
            let (s, _) = brahmic(c).unwrap();
            if let Some(u) = char_unit(c) {
                assert_eq!(unit_char(u, s), c);
            }
        }
    }

    #[test]
    fn split_is_variant_independent() {
        let mut parts = Vec::new();
        split_by_script("हैदराबాద్ abc", |s, _| parts.push(s));
        assert_eq!(parts, ["हैदराब", "ాద్ abc"]);
    }

    // ---------------------------------------------- variant-specific tests

    #[test]
    fn every_indic_char_is_its_own_unit() {
        assert_ne!(char_unit('क'), char_unit('ক'));
        for cp in BLOCK_START..BLOCK_END {
            let u = char_unit(char::from_u32(cp).unwrap()).unwrap();
            assert!(is_specific_unit(u));
        }
        assert_eq!(char_unit('a'), None);
    }

    #[test]
    fn cross_script_word_is_not_shared() {
        use crate::{train::Trainer, CoreBPE, DEFAULT_PATTERN};
        let corpus: Vec<String> = ["भारत सरकार\n", "বাংলাদেশ সরকার\n"]
            .iter()
            .flat_map(|s| std::iter::repeat(s.to_string()).take(20))
            .collect();
        let mut tr = Trainer::new(DEFAULT_PATTERN).unwrap();
        tr.feed(&corpus);
        let res = tr.finish(600, 2, 1, |_, _| {});
        let bpe = CoreBPE::new(DEFAULT_PATTERN, &res.specific_units, &res.merges, Default::default()).unwrap();
        let hi = bpe.encode_ordinary(" सरकार");
        let bn = bpe.encode_ordinary(" সরকার");
        assert_eq!(hi.len(), 1);
        assert_eq!(bn.len(), 1);
        assert_ne!(hi, bn);
    }
}
