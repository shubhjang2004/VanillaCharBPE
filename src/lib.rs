//! AksharaG2P: corpus-level cross-script phonemic BPE for Indic scripts.
//!
//! The BPE core is adapted from tiktoken, operating on `u32` units
//! (bytes + Brahmic phoneme units, see [`script`]) instead of raw bytes.

use std::collections::BinaryHeap;

use fancy_regex::Regex;
use rayon::prelude::*;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

pub mod script;
pub mod train;
#[cfg(feature = "python")]
mod py;

use script::*;

pub type Rank = u32;

/// GPT-4 style pre-tokenizer that keeps combining marks (matras, virama,
/// nukta) and ZWNJ/ZWJ inside words; cl100k's `\p{L}+` splits Indic words apart.
pub const DEFAULT_PATTERN: &str = concat!(
    r"'(?i:[sdmt]|ll|ve|re)",
    r"|[^\r\n\p{L}\p{N}\p{M}]?[\p{L}\p{M}",
    "\u{200C}\u{200D}",
    r"]+",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}\p{M}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

const MAX_NUM_THREADS: usize = 128;

/// One compiled regex per thread slot: sharing a single `Regex` across
/// threads serializes on its internal cache pool (same trick as tiktoken).
pub(crate) struct RegexPool(Vec<Regex>);

impl RegexPool {
    pub(crate) fn new(pattern: &str) -> Result<Self, String> {
        let regex = Regex::new(pattern).map_err(|e| e.to_string())?;
        Ok(Self((0..MAX_NUM_THREADS).map(|_| regex.clone()).collect()))
    }

    #[inline]
    pub(crate) fn get(&self) -> &Regex {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        std::thread::current().id().hash(&mut h);
        &self.0[h.finish() as usize % MAX_NUM_THREADS]
    }
}

/// Calls `f` for every regex piece of `text`, including any gaps between
/// matches, so that no input is ever dropped.
pub(crate) fn for_each_piece<'a>(regex: &Regex, text: &'a str, mut f: impl FnMut(&'a str)) {
    let mut last = 0;
    for mat in regex.find_iter(text) {
        let Ok(mat) = mat else { break };
        if mat.start() > last {
            f(&text[last..mat.start()]);
        }
        if mat.end() > mat.start() {
            f(mat.as_str());
        }
        last = mat.end();
    }
    if last < text.len() {
        f(&text[last..]);
    }
}

// ---------------------------------------------------------------------------
// BPE merge (tiktoken's algorithm over unit slices)
// ---------------------------------------------------------------------------

#[derive(Eq, PartialEq, Clone, Copy)]
struct Merge {
    start: usize,
    rank: Rank,
}

impl Ord for Merge {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.start.cmp(&self.start))
    }
}

impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

struct State {
    prev: usize,
    end: usize,
    next_end: usize,
    next_rank: Rank,
    cur_rank: Rank,
}

fn unit_pair_merge_large(ranks: &HashMap<Vec<u32>, Rank>, piece: &[u32]) -> Vec<Rank> {
    let mut state = Vec::with_capacity(piece.len());
    state.push(State {
        prev: usize::MAX,
        end: 1,
        next_end: 2,
        next_rank: Rank::MAX,
        cur_rank: Rank::MAX,
    });

    let mut heap = BinaryHeap::with_capacity(piece.len());
    for i in 0..piece.len() - 1 {
        if let Some(&rank) = ranks.get(&piece[i..i + 2]) {
            heap.push(Merge { start: i, rank });
            state[i].next_rank = rank;
        }
        // note this is happening offset by 1
        state.push(State {
            prev: i,
            end: i + 2,
            next_end: i + 3,
            next_rank: Rank::MAX,
            cur_rank: Rank::MAX,
        });
    }

    let potential_merge = |state: &mut Vec<State>,
                           heap: &mut BinaryHeap<Merge>,
                           start: usize,
                           next_end_item: usize| {
        state[start].next_end = next_end_item;
        state[start].next_rank = Rank::MAX; // Always invalidate the old merge
        if next_end_item <= piece.len() {
            if let Some(&rank) = ranks.get(&piece[start..next_end_item]) {
                heap.push(Merge { start, rank });
                state[start].next_rank = rank;
            }
        }
    };

    while let Some(left) = heap.pop() {
        if left.rank == Rank::MAX {
            break;
        }
        if left.rank != state[left.start].next_rank {
            continue; // This merge was invalidated, ignore it
        }

        let left_start = left.start;
        let right_start = state[left_start].end;
        let right_end = state[left_start].next_end;
        let right_next_end = state[right_start].next_end;

        state[left_start].cur_rank = state[left_start].next_rank;
        state[left_start].end = right_end;
        potential_merge(&mut state, &mut heap, left_start, right_next_end);
        if right_end < state.len() {
            state[right_end].prev = left_start;
        }
        if left_start > 0 {
            let prev_start = state[left_start].prev;
            potential_merge(&mut state, &mut heap, prev_start, right_end);
        }
        state[right_start].next_rank = Rank::MAX;
    }

    let mut result = Vec::new();
    let mut i = 0;
    while i < state.len() {
        if state[i].cur_rank != Rank::MAX {
            result.push(state[i].cur_rank);
        } else {
            result.push(ranks[&piece[i..state[i].end]]);
        }
        i = state[i].end;
    }
    result
}

fn unit_pair_merge(ranks: &HashMap<Vec<u32>, Rank>, piece: &[u32]) -> Vec<(usize, Rank)> {
    // (start, rank of the pair starting at start)
    let mut parts = Vec::with_capacity(piece.len() + 1);

    let mut min_rank: (Rank, usize) = (Rank::MAX, usize::MAX);
    for i in 0..piece.len() - 1 {
        let rank = *ranks.get(&piece[i..i + 2]).unwrap_or(&Rank::MAX);
        if rank < min_rank.0 {
            min_rank = (rank, i);
        }
        parts.push((i, rank));
    }
    parts.push((piece.len() - 1, Rank::MAX));
    parts.push((piece.len(), Rank::MAX));

    let get_rank = |parts: &Vec<(usize, Rank)>, i: usize| {
        if (i + 3) < parts.len() {
            *ranks
                .get(&piece[parts[i].0..parts[i + 3].0])
                .unwrap_or(&Rank::MAX)
        } else {
            Rank::MAX
        }
    };

    while min_rank.0 != Rank::MAX {
        let i = min_rank.1;
        if i > 0 {
            parts[i - 1].1 = get_rank(&parts, i - 1);
        }
        parts[i].1 = get_rank(&parts, i);
        parts.remove(i + 1);

        min_rank = (Rank::MAX, usize::MAX);
        for (i, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
            if rank < min_rank.0 {
                min_rank = (rank, i);
            }
        }
    }
    parts
}

pub fn unit_pair_encode(piece: &[u32], ranks: &HashMap<Vec<u32>, Rank>) -> Vec<Rank> {
    if piece.len() == 1 {
        return vec![ranks[piece]];
    }
    if piece.len() < 100 {
        return unit_pair_merge(ranks, piece)
            .windows(2)
            .map(|part| ranks[&piece[part[0].0..part[1].0]])
            .collect();
    }
    unit_pair_merge_large(ranks, piece)
}

// ---------------------------------------------------------------------------
// CoreBPE
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DecodeKeyError {
    pub token: Rank,
}

impl std::fmt::Display for DecodeKeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Invalid token for decoding: {}", self.token)
    }
}

pub struct CoreBPE {
    encoder: HashMap<Vec<u32>, Rank>,
    /// units of every ordinary token; empty for script markers
    decoder: Vec<Vec<u32>>,
    special_tokens_encoder: HashMap<String, Rank>,
    special_tokens_decoder: HashMap<Rank, Vec<u8>>,
    regex: RegexPool,
    special_regex: Option<RegexPool>,
    /// indexed by `unit - SPECIFIC_BASE`: whether that script-specific unit has a token
    specific_in_vocab: Vec<bool>,
}

impl CoreBPE {
    /// Builds the vocabulary: fixed tokens, then `specific_units`, then one
    /// token per merge `(left, right)`, then special tokens.
    pub fn new(
        pattern: &str,
        specific_units: &[u32],
        merges: &[(Rank, Rank)],
        special_tokens: HashMap<String, Rank>,
    ) -> Result<Self, String> {
        let regex = RegexPool::new(pattern)?;

        let mut decoder: Vec<Vec<u32>> = Vec::with_capacity(
            N_FIXED_TOKENS as usize + specific_units.len() + merges.len(),
        );
        for b in 0..SHARED_BASE {
            decoder.push(vec![b]);
        }
        for u in SHARED_BASE..MARKER_BASE {
            decoder.push(vec![u]);
        }
        for _ in 0..N_MARKERS {
            decoder.push(Vec::new());
        }

        let mut specific_in_vocab = vec![false; (N_UNITS - SPECIFIC_BASE) as usize];
        for &u in specific_units {
            if !is_specific_unit(u) || char_unit(unit_char(u, 0)) != Some(u) {
                return Err(format!("{u} is not a script-specific unit of {VARIANT}"));
            }
            let slot = &mut specific_in_vocab[(u - SPECIFIC_BASE) as usize];
            if *slot {
                return Err(format!("duplicate specific unit {u}"));
            }
            *slot = true;
            decoder.push(vec![u]);
        }

        for &(l, r) in merges {
            let n = decoder.len() as Rank;
            let ok = |t: Rank| t < n && !(MARKER_BASE..N_FIXED_TOKENS).contains(&t);
            if !ok(l) || !ok(r) {
                return Err(format!("merge ({l}, {r}) refers to an unknown token"));
            }
            let mut units = decoder[l as usize].clone();
            units.extend_from_slice(&decoder[r as usize]);
            decoder.push(units);
        }

        let mut encoder: HashMap<Vec<u32>, Rank> = HashMap::default();
        for (id, units) in decoder.iter().enumerate() {
            if !units.is_empty() {
                // the lowest rank wins if two merge paths yield the same units
                encoder.entry(units.clone()).or_insert(id as Rank);
            }
        }

        let n_ordinary = decoder.len() as Rank;
        let mut special_tokens_decoder = HashMap::default();
        for (s, &id) in &special_tokens {
            if id < n_ordinary {
                return Err(format!(
                    "special token {s:?} has id {id}, which collides with an ordinary token"
                ));
            }
            if special_tokens_decoder.insert(id, s.as_bytes().to_vec()).is_some() {
                return Err(format!("duplicate special token id {id}"));
            }
        }

        let special_regex = if special_tokens.is_empty() {
            None
        } else {
            let mut keys: Vec<&String> = special_tokens.keys().collect();
            // longest first so that overlapping specials match greedily
            keys.sort_by_key(|k| std::cmp::Reverse(k.len()));
            let alt = keys
                .iter()
                .map(|s| fancy_regex::escape(s))
                .collect::<Vec<_>>()
                .join("|");
            Some(RegexPool::new(&alt)?)
        };

        Ok(Self {
            encoder,
            decoder,
            special_tokens_encoder: special_tokens,
            special_tokens_decoder,
            regex,
            special_regex,
            specific_in_vocab,
        })
    }

    pub fn n_vocab(&self) -> usize {
        let max_special = self.special_tokens_decoder.keys().max().map(|&m| m as usize + 1);
        max_special.unwrap_or(0).max(self.decoder.len())
    }

    pub fn special_tokens(&self) -> &HashMap<String, Rank> {
        &self.special_tokens_encoder
    }

    // ----------------------------------------------------------------- encode

    fn push_units(&self, sub: &str, units: &mut Vec<u32>) {
        let mut buf = [0u8; 4];
        for c in sub.chars() {
            if let Some(u) = char_unit(c) {
                if !is_specific_unit(u) || self.specific_in_vocab[(u - SPECIFIC_BASE) as usize] {
                    units.push(u);
                    continue;
                }
            }
            units.extend(c.encode_utf8(&mut buf).bytes().map(u32::from));
        }
    }

    fn encode_ordinary_into(&self, text: &str, cur_script: &mut Option<usize>, out: &mut Vec<Rank>) {
        let mut units: Vec<u32> = Vec::new();
        for_each_piece(self.regex.get(), text, |piece| {
            split_by_script(piece, |sub, script| {
                if let Some(s) = script {
                    if *cur_script != Some(s) {
                        out.push(MARKER_BASE + s as Rank);
                        *cur_script = Some(s);
                    }
                }
                units.clear();
                self.push_units(sub, &mut units);
                match self.encoder.get(units.as_slice()) {
                    Some(&t) => out.push(t),
                    None => out.extend(unit_pair_encode(&units, &self.encoder)),
                }
            });
        });
    }

    /// Encodes `text`, treating special-token strings as ordinary text.
    pub fn encode_ordinary(&self, text: &str) -> Vec<Rank> {
        let mut out = Vec::new();
        self.encode_ordinary_into(text, &mut None, &mut out);
        out
    }

    /// Encodes `text`, turning occurrences of `allowed_special` into special tokens.
    pub fn encode(&self, text: &str, allowed_special: &HashSet<&str>) -> Vec<Rank> {
        let Some(special_regex) = &self.special_regex else {
            return self.encode_ordinary(text);
        };
        let special_regex = special_regex.get();
        let mut out = Vec::new();
        let mut cur_script = None;
        let mut start = 0;
        loop {
            let mut next_special = None;
            let mut start_find = start;
            while let Ok(Some(m)) = special_regex.find_from_pos(text, start_find) {
                if allowed_special.contains(m.as_str()) {
                    next_special = Some(m);
                    break;
                }
                start_find = m.start() + m.as_str().chars().next().map_or(1, char::len_utf8);
            }
            let end = next_special.map_or(text.len(), |m| m.start());
            self.encode_ordinary_into(&text[start..end], &mut cur_script, &mut out);
            match next_special {
                Some(m) => {
                    out.push(self.special_tokens_encoder[m.as_str()]);
                    start = m.end();
                }
                None => break,
            }
        }
        out
    }

    pub fn encode_ordinary_batch(&self, texts: &[String]) -> Vec<Vec<Rank>> {
        texts.par_iter().map(|t| self.encode_ordinary(t)).collect()
    }

    pub fn encode_batch(&self, texts: &[String], allowed_special: &HashSet<&str>) -> Vec<Vec<Rank>> {
        texts.par_iter().map(|t| self.encode(t, allowed_special)).collect()
    }

    // ----------------------------------------------------------------- decode

    /// Decodes tokens to the exact original bytes. Shared phonemes are
    /// rendered in the script of the most recent script marker (Devanagari
    /// before the first marker).
    pub fn decode_bytes(&self, tokens: &[Rank]) -> Result<Vec<u8>, DecodeKeyError> {
        let mut out = Vec::with_capacity(tokens.len() * 4);
        let mut script = DEVA;
        for &t in tokens {
            if (MARKER_BASE..N_FIXED_TOKENS).contains(&t) {
                script = (t - MARKER_BASE) as usize;
            } else if let Some(units) = self.decoder.get(t as usize) {
                render_units(units, script, &mut out);
            } else if let Some(bytes) = self.special_tokens_decoder.get(&t) {
                out.extend_from_slice(bytes);
            } else {
                return Err(DecodeKeyError { token: t });
            }
        }
        Ok(out)
    }

    pub fn decode_batch_bytes(&self, batch: &[Vec<Rank>]) -> Result<Vec<Vec<u8>>, DecodeKeyError> {
        batch.par_iter().map(|t| self.decode_bytes(t)).collect()
    }

    /// Bytes of a single token out of context: shared phonemes in `script`,
    /// script markers as the empty string.
    pub fn decode_single_token_bytes(&self, token: Rank, script: usize) -> Result<Vec<u8>, DecodeKeyError> {
        if let Some(units) = self.decoder.get(token as usize) {
            let mut out = Vec::new();
            render_units(units, script.min(N_SCRIPTS - 1), &mut out);
            return Ok(out);
        }
        self.special_tokens_decoder
            .get(&token)
            .cloned()
            .ok_or(DecodeKeyError { token })
    }

    /// The raw unit sequence of an ordinary token (empty for script markers).
    pub fn token_units(&self, token: Rank) -> Option<&[u32]> {
        self.decoder.get(token as usize).map(Vec::as_slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy() -> CoreBPE {
        let corpus: Vec<String> = ["काम কাম ಕಾಮ\n", "హైదరాబాద్\u{200C}లో ഞാൻ ਪੰਜਾਬ। hello\n"]
            .iter()
            .flat_map(|s| std::iter::repeat(s.to_string()).take(10))
            .collect();
        let mut tr = train::Trainer::new(DEFAULT_PATTERN).unwrap();
        tr.feed(&corpus);
        let res = tr.finish(700, 2, 1, |_, _| {});
        let first_special = N_FIXED_TOKENS + (res.specific_units.len() + res.merges.len()) as Rank;
        let mut specials = HashMap::default();
        specials.insert("<|endoftext|>".to_string(), first_special);
        CoreBPE::new(DEFAULT_PATTERN, &res.specific_units, &res.merges, specials).unwrap()
    }

    #[test]
    fn lossless_round_trip() {
        let bpe = toy();
        let texts = [
            "काम কাম  ಕಾಮ\n\n\tकाम",
            "హైదరాబాద్\u{200C}లో ഞാൻ ਪੰਜਾਬ। hello 123 ૧૨૩ 😀",
            "\u{093E}orphan matra, ।danda।, ൻ",
            "",
            "   ",
        ];
        for t in texts {
            let ids = bpe.encode_ordinary(t);
            assert_eq!(bpe.decode_bytes(&ids).unwrap(), t.as_bytes(), "{t:?}");
        }
    }

    #[test]
    fn specials() {
        let bpe = toy();
        let eot = bpe.special_tokens()["<|endoftext|>"];
        let mut allowed = HashSet::default();
        allowed.insert("<|endoftext|>");
        let ids = bpe.encode("काम<|endoftext|>কাম", &allowed);
        assert!(ids.contains(&eot));
        assert_eq!(bpe.decode_bytes(&ids).unwrap(), "काम<|endoftext|>কাম".as_bytes());
        assert!(!bpe.encode_ordinary("<|endoftext|>").contains(&eot));
    }
}
