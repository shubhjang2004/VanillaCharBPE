//! Corpus-level BPE training over phoneme units.
//!
//! Text is pre-tokenized with the same regex + script split as the encoder,
//! piece frequencies are counted in parallel, and merges are learned with an
//! incremental pair-count heap.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use rayon::prelude::*;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::script::*;
use crate::{for_each_piece, Rank, RegexPool};

pub struct Trainer {
    regex: RegexPool,
    /// full unit sequence of each piece (every Brahmic char as a unit) -> count
    piece_counts: HashMap<Vec<u32>, u64>,
}

pub struct TrainResult {
    pub specific_units: Vec<u32>,
    pub merges: Vec<(Rank, Rank)>,
}

fn full_units(sub: &str, out: &mut Vec<u32>) {
    let mut buf = [0u8; 4];
    for c in sub.chars() {
        match char_unit(c) {
            Some(u) => out.push(u),
            None => out.extend(c.encode_utf8(&mut buf).bytes().map(u32::from)),
        }
    }
}

impl Trainer {
    pub fn new(pattern: &str) -> Result<Self, String> {
        Ok(Self {
            regex: RegexPool::new(pattern)?,
            piece_counts: HashMap::default(),
        })
    }

    pub fn n_unique_pieces(&self) -> usize {
        self.piece_counts.len()
    }

    /// Counts the pieces of a batch of documents.
    pub fn feed(&mut self, texts: &[String]) {
        let regex = &self.regex;
        let local = texts
            .par_iter()
            .fold(HashMap::<Vec<u32>, u64>::default, |mut counts, text| {
                let mut units = Vec::new();
                for_each_piece(regex.get(), text, |piece| {
                    split_by_script(piece, |sub, _| {
                        units.clear();
                        full_units(sub, &mut units);
                        *counts.entry(units.clone()).or_default() += 1;
                    });
                });
                counts
            })
            .reduce(HashMap::default, |a, b| {
                let (mut big, small) = if a.len() >= b.len() { (a, b) } else { (b, a) };
                for (k, v) in small {
                    *big.entry(k).or_default() += v;
                }
                big
            });
        for (k, v) in local {
            *self.piece_counts.entry(k).or_default() += v;
        }
    }

    /// Learns merges until the ordinary vocabulary has `n_ordinary` tokens
    /// or no pair occurs at least `min_frequency` times. Script-specific
    /// units seen fewer than `min_unit_frequency` times fall back to bytes.
    pub fn finish(
        &self,
        n_ordinary: usize,
        min_frequency: u64,
        min_unit_frequency: u64,
        mut progress: impl FnMut(usize, usize),
    ) -> TrainResult {
        // 1. script-specific base tokens
        let mut unit_freq: HashMap<u32, u64> = HashMap::default();
        for (units, &c) in &self.piece_counts {
            for &u in units {
                if is_specific_unit(u) {
                    *unit_freq.entry(u).or_default() += c;
                }
            }
        }
        let mut specific_units: Vec<u32> = unit_freq
            .into_iter()
            .filter(|&(_, f)| f >= min_unit_frequency)
            .map(|(u, _)| u)
            .collect();
        specific_units.sort_unstable();

        let mut tok_units: Vec<Vec<u32>> = (0..MARKER_BASE).map(|u| vec![u]).collect();
        tok_units.extend((0..N_MARKERS).map(|_| Vec::new()));
        let mut unit_to_tok: HashMap<u32, Rank> = HashMap::default();
        for &u in &specific_units {
            unit_to_tok.insert(u, tok_units.len() as Rank);
            tok_units.push(vec![u]);
        }

        // 2. pieces as base-token sequences
        let mut word_counts: HashMap<Vec<Rank>, i64> = HashMap::default();
        let mut buf = [0u8; 4];
        for (units, &c) in &self.piece_counts {
            let mut toks = Vec::with_capacity(units.len());
            for &u in units {
                if !is_specific_unit(u) {
                    toks.push(u);
                } else if let Some(&t) = unit_to_tok.get(&u) {
                    toks.push(t);
                } else {
                    let ch = unit_char(u, 0);
                    toks.extend(ch.encode_utf8(&mut buf).bytes().map(u32::from));
                }
            }
            *word_counts.entry(toks).or_default() += c as i64;
        }
        let (mut words, freqs): (Vec<Vec<Rank>>, Vec<i64>) = word_counts.into_iter().unzip();

        // 3. pair statistics
        type Pair = (Rank, Rank);
        let mut pair_counts: HashMap<Pair, i64> = HashMap::default();
        let mut where_: HashMap<Pair, HashSet<u32>> = HashMap::default();
        for (idx, w) in words.iter().enumerate() {
            for p in w.windows(2) {
                let p = (p[0], p[1]);
                *pair_counts.entry(p).or_default() += freqs[idx];
                where_.entry(p).or_default().insert(idx as u32);
            }
        }
        let mut heap: BinaryHeap<(i64, Reverse<Pair>)> =
            pair_counts.iter().map(|(&p, &c)| (c, Reverse(p))).collect();

        let mut seq_to_tok: HashMap<Vec<u32>, Rank> = HashMap::default();
        for (t, u) in tok_units.iter().enumerate() {
            if !u.is_empty() {
                seq_to_tok.insert(u.clone(), t as Rank);
            }
        }

        // 4. merge loop
        let n_merges_target = n_ordinary.saturating_sub(tok_units.len());
        let mut merges: Vec<Pair> = Vec::with_capacity(n_merges_target);
        let mut touched: HashSet<Pair> = HashSet::default();
        while merges.len() < n_merges_target {
            let best = loop {
                match heap.pop() {
                    None => break None,
                    Some((c, Reverse(p))) if pair_counts.get(&p) == Some(&c) => break Some((p, c)),
                    Some(_) => continue,
                }
            };
            let Some((best, count)) = best else { break };
            if (count as u64) < min_frequency {
                break;
            }

            let mut merged = tok_units[best.0 as usize].clone();
            merged.extend_from_slice(&tok_units[best.1 as usize]);
            let new_tok = match seq_to_tok.get(&merged) {
                // a different merge path already produced these units; reuse its token
                Some(&t) => t,
                None => {
                    let t = tok_units.len() as Rank;
                    seq_to_tok.insert(merged.clone(), t);
                    tok_units.push(merged);
                    merges.push(best);
                    progress(merges.len(), n_merges_target);
                    t
                }
            };

            pair_counts.remove(&best);
            let affected = where_.remove(&best).unwrap_or_default();
            touched.clear();
            for idx in affected {
                let freq = freqs[idx as usize];
                let old = &words[idx as usize];
                for p in old.windows(2) {
                    let p = (p[0], p[1]);
                    if p == best {
                        continue;
                    }
                    if let Some(c) = pair_counts.get_mut(&p) {
                        *c -= freq;
                        if *c <= 0 {
                            pair_counts.remove(&p);
                        }
                    }
                    if let Some(set) = where_.get_mut(&p) {
                        set.remove(&idx);
                    }
                    touched.insert(p);
                }
                let mut new_w = Vec::with_capacity(old.len());
                let mut i = 0;
                while i < old.len() {
                    if i + 1 < old.len() && old[i] == best.0 && old[i + 1] == best.1 {
                        new_w.push(new_tok);
                        i += 2;
                    } else {
                        new_w.push(old[i]);
                        i += 1;
                    }
                }
                for p in new_w.windows(2) {
                    let p = (p[0], p[1]);
                    *pair_counts.entry(p).or_default() += freq;
                    where_.entry(p).or_default().insert(idx);
                    touched.insert(p);
                }
                words[idx as usize] = new_w;
            }
            for p in &touched {
                if let Some(&c) = pair_counts.get(p) {
                    heap.push((c, Reverse(*p)));
                }
            }
        }

        TrainResult { specific_units, merges }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CoreBPE, DEFAULT_PATTERN};

    #[test]
    fn train_then_round_trip() {
        let corpus: Vec<String> = [
            "भारत सरकार नई दिल्ली में है।\n",
            "বাংলাদেশ সরকার ঢাকায় আছে।\n",
            "தமிழ்நாடு அரசு சென்னையில் உள்ளது\n",
            "కేరళ సర్కార్ తిరువనంతపురం\n",
            "കേരള സർക്കാർ തിരുവനന്തപുരത്ത് ആണ്\n",
        ]
        .iter()
        .flat_map(|s| std::iter::repeat(s.to_string()).take(20))
        .collect();

        let mut tr = Trainer::new(DEFAULT_PATTERN).unwrap();
        tr.feed(&corpus);
        let res = tr.finish(600, 2, 1, |_, _| {});
        assert!(!res.merges.is_empty());

        let bpe = CoreBPE::new(DEFAULT_PATTERN, &res.specific_units, &res.merges, HashMap::default()).unwrap();
        for t in &corpus[..] {
            assert_eq!(bpe.decode_bytes(&bpe.encode_ordinary(t)).unwrap(), t.as_bytes());
        }
        // unseen script (Gujarati, Odia) still round-trips
        for t in ["ગુજરાત સરકાર", "ଓଡ଼ିଶା ସରକାର"] {
            assert_eq!(bpe.decode_bytes(&bpe.encode_ordinary(t)).unwrap(), t.as_bytes());
        }
    }
}
