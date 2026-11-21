// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Contextual biasing: a beam search (CTC's over the likeliest alignments, a transducer's over its
//! decisions) whose hypotheses get a bonus for each character that follows one of the
//! [`Hotwords`] from the start of a word, taken back when the word turns off it before a word's
//! end or stem. The words are matched on the text of the pieces, lowercased, so a word gets its
//! bonus however the vocabulary cuts it.

/// Words and phrases the decoder favours, for [`Options::hotwords`](crate::Options::hotwords):
/// names, terms and acronyms a model mishears as more common words. Start from
/// [`Hotwords::new`] and change the fields.
///
/// A word is matched from the start of a spoken word, case-insensitively, as the model spells
/// it: write `кубернетес` for a Russian model that writes the word in Cyrillic, `GigaAM` where it
/// writes Latin. A word counts as heard when the decoded word reaches its end, or, for a word of 4
/// or more letters ending in a vowel, `ь` or `й`, its stem without that letter; letters after
/// that (Russian case endings: `докера`, `докером`) keep the bonus. A phrase (`гига ам`) also
/// matches with its words run together (`гигаам`).
///
/// # Examples
///
/// ```
/// let mut hotwords = wavo::Hotwords::new(["кубернетес", "GigaAM"]);
/// hotwords.boost = 1.5;
/// let mut options = wavo::Options::default();
/// options.hotwords = Some(hotwords);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Hotwords {
  /// The words and phrases; empty ones are skipped.
  pub words: Vec<String>,
  /// The bonus per matching character, in natural-log probability: 0 matches nothing, and from
  /// about 2 the decoder writes them over common words that sound alike (`доке` as `докере`).
  /// Default 1.5.
  pub boost: f32,
  /// Hypotheses kept per frame, at least 1, and half as many again of the likeliest without the
  /// bonus of unfinished words. Default 8.
  pub beam: usize,
}

impl Hotwords {
  /// `words` with the default [`boost`](Self::boost) and [`beam`](Self::beam).
  pub fn new<S: Into<String>>(words: impl IntoIterator<Item = S>) -> Self {
    Self { words: words.into_iter().map(Into::into).collect(), boost: 1.5, beam: 8 }
  }
}

/// Characters a word ending in one of them may drop for its stem: `москва` → `москв`.
const ENDINGS: &str = "аяоеёыиьйуюэ";

/// A word boundary: SentencePiece's `▁`, whitespace or punctuation.
fn boundary(c: char) -> bool {
  !(c.is_alphanumeric() || c == '-' || c == '\'')
}

/// The words' characters as a tree; node 0 is the root.
pub(crate) struct Trie {
  /// Per node its children `(character, node)` and whether a word ends there.
  nodes: Vec<(Vec<(char, u32)>, bool)>,
}

impl Trie {
  pub fn new(words: &[String]) -> Self {
    let mut trie = Self { nodes: vec![(Vec::new(), false)] };
    for word in words {
      let spaced: Vec<String> =
        word.split(boundary).filter(|w| !w.is_empty()).map(str::to_lowercase).collect();
      if spaced.is_empty() {
        continue;
      }
      trie.insert(&spaced.join(" "));
      if spaced.len() > 1 {
        trie.insert(&spaced.concat());
      }
    }
    trie
  }

  fn insert(&mut self, word: &str) {
    let chars: Vec<char> = word.chars().collect();
    let mut node = 0;
    for (i, &c) in chars.iter().enumerate() {
      node = match self.child(node, c) {
        Some(n) => n,
        None => {
          self.nodes.push((Vec::new(), false));
          let n = (self.nodes.len() - 1) as u32;
          self.nodes[node as usize].0.push((c, n));
          n
        }
      };
      let stem = i + 2 == chars.len() && chars.len() >= 4 && ENDINGS.contains(chars[i + 1]);
      if stem || i + 1 == chars.len() {
        self.nodes[node as usize].1 = true;
      }
    }
  }

  fn child(&self, node: u32, c: char) -> Option<u32> {
    self.nodes[node as usize].0.iter().find(|e| e.0 == c).map(|e| e.1)
  }

  pub fn is_empty(&self) -> bool {
    self.nodes[0].0.is_empty()
  }

  /// `ctx` after the characters of `text`, and the change to the hypothesis' bonus.
  fn advance(&self, mut ctx: Ctx, text: &[char], boost: f32) -> (Ctx, f32) {
    let mut delta = 0.0;
    for &c in text {
      if boundary(c) {
        let phrase = c == '▁' || c.is_whitespace();
        match ctx.node.filter(|&n| n != 0 && phrase).and_then(|n| self.child(n, ' ')) {
          Some(n) => {
            ctx.node = Some(n);
            (ctx.spaced, ctx.gap) = (true, true);
          }
          None => {
            delta -= ctx.open();
            ctx = Ctx::START;
          }
        }
        continue;
      }
      let lower = c.to_lowercase().next().unwrap_or(c);
      // Past a space in a phrase, a letter the phrase doesn't go on with ends it there: the
      // words before keep what they earned, and a word starting here starts from the tree's
      // root, not as their case ending (`докер образ` with `докер компоуз`, `докер` lost its
      // bonus at `обра` and kept it at `обз`).
      if ctx.spaced && ctx.node.and_then(|n| self.child(n, lower)).is_none() {
        delta -= ctx.open();
        ctx = if ctx.gap { Ctx::START } else { Ctx::OFF };
      }
      ctx.gap = false;
      match ctx.node.and_then(|n| self.child(n, lower)) {
        Some(n) => {
          delta += boost;
          ctx.word += boost;
          if self.nodes[n as usize].1 {
            ctx.kept = Some(ctx.word);
          }
          ctx.node = Some(n);
        }
        // A case ending after a word's end or stem.
        None if ctx.kept.is_some() && ctx.tail < TAIL && ('а'..='я').contains(&lower) => {
          ctx.tail += 1;
          ctx.node = None;
        }
        None => {
          delta -= ctx.word;
          ctx = Ctx::OFF;
        }
      }
    }
    (ctx, delta)
  }
}

/// Letters a word may go on for past a word's end or stem and keep the bonus: `докер` +
/// `ами`.
const TAIL: u8 = 3;

/// Where a hypothesis is in the tree: `node` while its current word follows a word from the
/// start (`Some(0)` at a word's start), `None` in a case ending or off the tree until the next
/// boundary; `word` the bonus the word got, `kept` how much of it a word end or stem has
/// earned, and `tail` the letters after the last of them; `spaced` once a phrase went past a
/// space, `gap` right after it.
#[derive(Clone, Copy, Debug)]
struct Ctx {
  node: Option<u32>,
  word: f32,
  kept: Option<f32>,
  tail: u8,
  spaced: bool,
  gap: bool,
}

impl Ctx {
  const START: Ctx =
    Ctx { node: Some(0), word: 0.0, kept: None, tail: 0, spaced: false, gap: false };
  const OFF: Ctx = Ctx { node: None, ..Ctx::START };

  /// The bonus taken back if the word ended here.
  fn open(&self) -> f32 {
    self.word - self.kept.unwrap_or(0.0)
  }
}

/// One prefix: its last token and its parent.
struct Prefix {
  parent: u32,
  token: u32,
  ctx: Ctx,
  bonus: f32,
}

#[derive(Clone, Copy)]
struct Hyp {
  prefix: u32,
  /// Log probabilities of the prefix ending in a blank and in its last token.
  blank: f32,
  token: f32,
  /// The frame its likelier path emitted the last token on.
  since: u32,
}

impl Hyp {
  fn new(prefix: u32) -> Self {
    Hyp { prefix, blank: f32::NEG_INFINITY, token: f32::NEG_INFINITY, since: 0 }
  }

  /// A path ending in a blank (`blank`) or in the last token, of log probability `p`, whose last
  /// token was emitted on frame `since`.
  fn take(&mut self, blank: bool, p: f32, since: u32) {
    if p > add(self.blank, self.token) {
      self.since = since;
    }
    let to = if blank { &mut self.blank } else { &mut self.token };
    *to = add(*to, p);
  }
}

/// Frames a word's partial bonus fully counts for after its last letter; then half as much per
/// frame. Without it a hypothesis that stops in a word's middle could stay on blanks over the
/// next words (`парак…верь` for `паракет и проверь`) and push the others out of the beam
/// before it pays the bonus back. A model's letters come a few frames apart.
const GRACE: u32 = 3;

/// How far below a frame's best logit a token is still tried, at least: past it a token needs a
/// bonus no piece gets.
const PRUNE: f32 = 10.0;

/// A transducer's token other than greedy decoding's is taken for a word only within this much,
/// in log probability, of the decision after its own bonus: at 2 the reading page lost 8 of 64
/// terms found at 4, at 6 it found 6 more in twice the time.
const REACH: f32 = 4.0;

/// Two paths into one prefix keep the likelier: the beam then follows the likeliest alignment,
/// so without a bonus it ends on greedy decoding's (summing them, as prefix beam search does,
/// can rank another cut of the same sounds higher, `SS H` over greedy's `SSH`).
fn add(a: f32, b: f32) -> f32 {
  a.max(b)
}

/// CTC beam search over `logits` `[frames][classes]` with the bonus of `trie`: `texts[id]`
/// is each token's text as decoded (`▁` for a space). Returns `(token, frame)` pairs as greedy
/// CTC does, a token's frame being the first of its run on the likeliest alignment.
pub(crate) fn ctc(
  logits: &[f32],
  classes: usize,
  blank: usize,
  texts: &[Vec<char>],
  trie: &Trie,
  boost: f32,
  beam: usize,
) -> Vec<(u32, u32)> {
  const ROOT: u32 = u32::MAX;
  let beam = beam.max(1);
  let mut prefixes = vec![Prefix { parent: ROOT, token: ROOT, ctx: Ctx::START, bonus: 0.0 }];
  let mut children: std::collections::HashMap<(u32, u32), u32> = Default::default();
  let mut beams = vec![Hyp { blank: 0.0, ..Hyp::new(0) }];
  let mut order: Vec<usize> = Vec::with_capacity(classes);
  let longest = texts.iter().map(Vec::len).max().unwrap_or(0);
  let prune = PRUNE.max(boost * longest as f32);
  let mut next: Vec<Hyp> = Vec::new();

  for (t, logits) in logits.chunks_exact(classes).enumerate() {
    let t = t as u32;
    // Raw logits: every alignment takes one entry of each row, so a row's log-softmax would
    // shift them all alike.
    let row = logits;
    // A prefix grows only by the frame's `beam` likeliest tokens within `prune` of its best:
    // a model's rows are peaked, so mostly one or two.
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    order.clear();
    order.extend((0..classes).filter(|&c| row[c] > max - prune));
    let top = beam.min(order.len());
    if top < order.len() {
      order.select_nth_unstable_by(top - 1, |&a, &b| row[b].total_cmp(&row[a]));
    }

    next.clear();
    let slot = |prefix: u32, next: &mut Vec<Hyp>| match next.iter().position(|h| h.prefix == prefix)
    {
      Some(i) => i,
      None => {
        next.push(Hyp::new(prefix));
        next.len() - 1
      }
    };
    for h in &beams {
      let total = add(h.blank, h.token);
      let last = prefixes[h.prefix as usize].token;
      let i = slot(h.prefix, &mut next);
      next[i].take(true, total + row[blank], h.since);
      if last != ROOT {
        next[i].take(false, h.token + row[last as usize], h.since);
      }
      for &c in &order[..top] {
        if c == blank {
          continue;
        }
        let (pctx, pbonus) = (prefixes[h.prefix as usize].ctx, prefixes[h.prefix as usize].bonus);
        let child = *children.entry((h.prefix, c as u32)).or_insert_with(|| {
          let (ctx, delta) = trie.advance(pctx, &texts[c], boost);
          let bonus = pbonus + delta;
          prefixes.push(Prefix { parent: h.prefix, token: c as u32, ctx, bonus });
          (prefixes.len() - 1) as u32
        });
        // A repeat of the last token extends the prefix only across a blank.
        let from = if c as u32 == last { h.blank } else { total };
        let i = slot(child, &mut next);
        next[i].take(false, from + row[c], t);
      }
    }
    // Ranked with the bonus of the words being spelled, so that a hotword survives until its
    // end; and the best as if decoding stopped here, so that the words a partial bonus would
    // crowd out (greedy's, mostly) do too.
    let settled = |h: &Hyp| {
      let p = &prefixes[h.prefix as usize];
      add(h.blank, h.token) + p.bonus - p.ctx.open()
    };
    let score = |h: &Hyp| {
      let p = &prefixes[h.prefix as usize];
      let late = (t - h.since).saturating_sub(GRACE).min(30) as i32;
      settled(h) + p.ctx.open() * 0.5f32.powi(late)
    };
    next.sort_by(|a, b| settled(b).total_cmp(&settled(a)).then(a.prefix.cmp(&b.prefix)));
    let sure = (beam / 2).max(1).min(next.len());
    next[sure..].sort_by(|a, b| score(b).total_cmp(&score(a)).then(a.prefix.cmp(&b.prefix)));
    next.truncate(beam + sure);
    std::mem::swap(&mut beams, &mut next);
  }

  // The last word's bonus counts only if it ended a word: the beam's first is the best so.
  let best = beams.first();
  let mut tokens = Vec::new();
  let mut p = best.map_or(0, |h| h.prefix);
  while p != 0 {
    tokens.push(prefixes[p as usize].token);
    p = prefixes[p as usize].parent;
  }
  tokens.reverse();
  let tokens: Vec<u32> = lone_spaces(&tokens, texts).map(|i| tokens[i]).collect();
  let frames = align(logits, classes, blank, &tokens);
  tokens.into_iter().zip(frames).collect()
}

/// The indices of `tokens` but a lone `▁` before a word that starts with its own: an empty word.
/// The model writes one before Latin letters, and the bonus may have spelled the word in Cyrillic.
fn lone_spaces<'a>(tokens: &'a [u32], texts: &'a [Vec<char>]) -> impl Iterator<Item = usize> + 'a {
  let space = |id: u32| texts[id as usize] == ['▁'];
  let starts = |id: u32| texts[id as usize].first() == Some(&'▁');
  (0..tokens.len())
    .filter(move |&i| !(space(tokens[i]) && tokens.get(i + 1).is_some_and(|&n| starts(n))))
}

/// A transducer's predictor and joint, for [`transducer`]: GigaAM's RNN-T, Parakeet's TDT.
pub(crate) trait Transducer {
  /// The predictor's state after some tokens.
  type State;
  /// The state before any token, and the joint's projection of it.
  fn start(&self) -> (Self::State, Vec<f32>);
  /// The state after `token`, and the joint's projection of it.
  fn step(&self, state: &Self::State, token: usize) -> (Self::State, Vec<f32>);
  /// [`Transducer::step`] for each of `steps`, in one go where the model can.
  fn steps(&self, steps: &[(&Self::State, usize)]) -> Vec<(Self::State, Vec<f32>)> {
    steps.iter().map(|&(state, token)| self.step(state, token)).collect()
  }
  /// For frame `t` and each of `preds`: the log probabilities of the tokens (blank included) and
  /// of the durations (none for RNN-T).
  fn joint(&self, t: usize, preds: &[&[f32]]) -> Vec<(Vec<f32>, Vec<f32>)>;
}

/// How a transducer moves on: with no `durations` (RNN-T) a token stays on its frame and a blank
/// moves one on; with them (TDT) a decision moves on by its likeliest duration, a blank by at
/// least one. At most `max_symbols` tokens are emitted on one frame, as greedy decoding does.
pub(crate) struct Steps<'a> {
  pub blank: usize,
  pub durations: &'a [usize],
  pub max_symbols: usize,
}

/// A prefix's predictor: its state and projection, or its parent's and the token to step on,
/// computed when the prefix is first scored.
enum Pred<S> {
  Ready(std::rc::Rc<(S, Vec<f32>)>),
  After(std::rc::Rc<(S, Vec<f32>)>, usize),
}

impl<S> Clone for Pred<S> {
  fn clone(&self) -> Self {
    match self {
      Pred::Ready(r) => Pred::Ready(r.clone()),
      Pred::After(r, token) => Pred::After(r.clone(), *token),
    }
  }
}

/// The predictor of every path, stepped in one go.
fn ready<T: Transducer>(model: &T, paths: &mut [Path<T::State>]) {
  let after: Vec<(&T::State, usize)> = (paths.iter())
    .filter_map(|p| match &p.pred {
      Pred::After(parent, token) => Some((&parent.0, *token)),
      Pred::Ready(_) => None,
    })
    .collect();
  if after.is_empty() {
    return;
  }
  let mut stepped = model.steps(&after).into_iter();
  for p in paths.iter_mut() {
    if matches!(p.pred, Pred::After(..)) {
      p.pred = Pred::Ready(std::rc::Rc::new(stepped.next().unwrap()));
    }
  }
}

/// A transducer hypothesis: a prefix on a frame, its log probability, the frame of its last
/// token, the tokens emitted on this frame so far, and whether it is a guess.
struct Path<S> {
  prefix: u32,
  score: f32,
  since: u32,
  symbols: usize,
  pred: Pred<S>,
  /// It left greedy decoding's decisions for a word it hasn't finished.
  guess: bool,
  /// The frames of its tokens, the last first.
  frames: Option<std::rc::Rc<Frames>>,
}

/// A path's token frames as a list shared by the paths that branch from it.
struct Frames {
  frame: u32,
  before: Option<std::rc::Rc<Frames>>,
}

/// One prefix of a transducer search: its last token and the tree's state after it. The frames
/// are each path's: two paths into one prefix may have emitted its tokens on other frames.
struct Node {
  parent: u32,
  token: u32,
  ctx: Ctx,
  bonus: f32,
}

/// The hypotheses of one frame, as [`ctc`] keeps them: one per prefix (the likelier; the
/// predictor's state follows from the prefix alone), the `beam / 2` best as if decoding stopped
/// here and the `beam` best with the bonus of the words being spelled, and the best that is no
/// guess. The first is the best without unfinished words.
fn keep<S>(mut paths: Vec<Path<S>>, nodes: &[Node], t: usize, beam: usize) -> Vec<Path<S>> {
  // Paths into one prefix merge into the likelier; it is a guess only if both are.
  paths.sort_by(|a, b| a.prefix.cmp(&b.prefix).then(b.score.total_cmp(&a.score)));
  paths.dedup_by(|b, a| {
    let same = a.prefix == b.prefix;
    if same {
      a.guess &= b.guess;
    }
    same
  });
  let settled = |h: &Path<S>| {
    let n = &nodes[h.prefix as usize];
    h.score + n.bonus - n.ctx.open()
  };
  let score = |h: &Path<S>| {
    let n = &nodes[h.prefix as usize];
    let late = (t as u32).saturating_sub(h.since).saturating_sub(GRACE).min(30) as i32;
    settled(h) + n.ctx.open() * 0.5f32.powi(late)
  };
  paths.sort_by(|a, b| settled(b).total_cmp(&settled(a)).then(a.prefix.cmp(&b.prefix)));
  let sure = (beam / 2).max(1).min(paths.len());
  paths[sure..].sort_by(|a, b| score(b).total_cmp(&score(a)).then(a.prefix.cmp(&b.prefix)));
  // The best path that is no guess stays too: guesses may all end, and without it nothing
  // would be left to go on with.
  let sure_path = paths.iter().position(|p| !p.guess).filter(|&i| i >= beam + sure);
  if let Some(i) = sure_path {
    paths.swap(beam + sure - 1, i);
  }
  paths.truncate(beam + sure);
  paths
}

/// Transducer beam search over `frames` with the bonus of `trie`, as [`ctc`]'s: hypotheses are
/// compared on the same frame only, each frame's kept as [`keep`] does, and a frame's tokens are
/// tried again until none stays on it. A hypothesis leaves greedy decoding's decision only for a
/// word of `trie`, within the bonus's reach, so with none in reach it decodes as greedy does (a
/// transducer's beam over all tokens would rewrite everyday speech: `к 9:00` as `к девяти`).
/// Returns `(token, frame)` pairs as greedy decoding does.
pub(crate) fn transducer<T: Transducer>(
  model: &T,
  frames: usize,
  steps: &Steps,
  texts: &[Vec<char>],
  trie: &Trie,
  boost: f32,
  beam: usize,
) -> Vec<(u32, u32)> {
  const ROOT: u32 = u32::MAX;
  // Each hypothesis costs a predictor step and a joint row: half CTC's beam keeps the cost near
  // greedy decoding's for as many words found (64 of 98 terms of wavoscribe's reading page with
  // GigaAM RNN-T at 4, 66 at 8 in twice the time).
  let beam = (beam / 2).max(1);
  let blank = steps.blank;
  let longest = texts.iter().map(Vec::len).max().unwrap_or(0);
  let prune = PRUNE.max(boost * longest as f32);
  let mut nodes = vec![Node { parent: ROOT, token: ROOT, ctx: Ctx::START, bonus: 0.0 }];
  let mut children: std::collections::HashMap<(u32, u32), u32> = Default::default();
  let mut buckets: Vec<Vec<Path<T::State>>> = (0..=frames).map(|_| Vec::new()).collect();
  let start = Pred::Ready(std::rc::Rc::new(model.start()));
  buckets[0].push(Path {
    prefix: 0,
    score: 0.0,
    since: 0,
    symbols: 0,
    pred: start,
    guess: false,
    frames: None,
  });
  let mut order: Vec<usize> = Vec::new();

  for t in 0..frames {
    let mut paths = keep(std::mem::take(&mut buckets[t]), &nodes, t, beam);
    while !paths.is_empty() {
      ready(model, &mut paths);
      let preds: Vec<&[f32]> = (paths.iter())
        .map(|p| match &p.pred {
          Pred::Ready(r) => r.1.as_slice(),
          Pred::After(..) => unreachable!(),
        })
        .collect();
      let scores = model.joint(t, &preds);
      let mut stay = Vec::new();
      for (p, (tokens, durations)) in paths.iter().zip(scores) {
        let (duration, d) = match durations.is_empty() {
          true => (None, 0.0),
          false => {
            let i = argmax(&durations);
            (Some(steps.durations[i]), durations[i])
          }
        };
        let to = |on: usize| (t + on).min(frames);
        // Off a word, a hypothesis makes greedy decoding's decision, or starts a word; spelling
        // one, it may also wait for its next letter or take the one that goes on with it. So
        // without a word in reach the search is greedy decoding.
        let choice = argmax(&tokens);
        let max = tokens[choice];
        let spelling = |n: &Node| n.ctx.word > 0.0;
        let on_word = spelling(&nodes[p.prefix as usize]);
        // A guess waits for its next letter at most `GRACE` frames: longer, it would skip the
        // words in between (`и постмания` for `и подключись PS. Модель` with `постман`).
        let waits = !p.guess || t as u32 <= p.since + GRACE;
        if choice == blank || (on_word && waits) {
          let blank_path = Path {
            prefix: p.prefix,
            score: p.score + tokens[blank] + d,
            since: p.since,
            symbols: 0,
            pred: p.pred.clone(),
            guess: p.guess || choice != blank,
            frames: p.frames.clone(),
          };
          buckets[to(duration.unwrap_or(1).max(1))].push(blank_path);
        }

        order.clear();
        // A token other than greedy decoding's is taken for a word only if its own bonus pays
        // for most of what it costs: else a guess would spell a word out on one frame (a
        // transducer may emit many), whatever was said. Its bonus is at most `boost` a letter.
        let reach = REACH.min(prune);
        let worth =
          |c: usize| c == choice || tokens[c] > max - reach - boost * texts[c].len() as f32;
        for c in (0..tokens.len()).filter(|&c| c != blank && worth(c)) {
          let parent = &nodes[p.prefix as usize];
          let (ctx, bonus) = match children.get(&(p.prefix, c as u32)) {
            Some(&n) => (nodes[n as usize].ctx, nodes[n as usize].bonus),
            None => {
              let (ctx, delta) = trie.advance(parent.ctx, &texts[c], boost);
              (ctx, parent.bonus + delta)
            }
          };
          let word = ctx.word > 0.0;
          // A guess whose word turns off before its end was wrong: greedy decoding's path has
          // that stretch, so it ends here rather than go on as everyday speech.
          if p.guess && bonus < parent.bonus && !word {
            continue;
          }
          if c == choice || (word && tokens[c] + (bonus - parent.bonus) > max - reach) {
            children.entry((p.prefix, c as u32)).or_insert_with(|| {
              nodes.push(Node { parent: p.prefix, token: c as u32, ctx, bonus });
              (nodes.len() - 1) as u32
            });
            order.push(c);
          }
        }
        let top = beam.min(order.len());
        if top < order.len() {
          order.select_nth_unstable_by(top - 1, |&a, &b| tokens[b].total_cmp(&tokens[a]));
        }
        let Pred::Ready(parent) = &p.pred else { unreachable!() };
        for &c in &order[..top] {
          let child = children[&(p.prefix, c as u32)];
          let mut on = duration.unwrap_or(0);
          let symbols = if on == 0 { p.symbols + 1 } else { 0 };
          if on == 0 && symbols >= steps.max_symbols {
            on = 1;
          }
          // A word finished (its bonus kept past the word) is no guess any more.
          let (before, node) = (&nodes[p.prefix as usize], &nodes[child as usize]);
          let finished = !spelling(node) && node.bonus >= before.bonus;
          let path = Path {
            prefix: child,
            score: p.score + tokens[c] + d,
            since: t as u32,
            symbols: if on == 0 { symbols } else { 0 },
            pred: Pred::After(parent.clone(), c),
            guess: (p.guess || c != choice) && !finished,
            frames: Some(std::rc::Rc::new(Frames { frame: t as u32, before: p.frames.clone() })),
          };
          match on {
            0 => stay.push(path),
            on => buckets[to(on)].push(path),
          }
        }
      }
      paths = keep(stay, &nodes, t, beam);
    }
  }

  // The last word's bonus counts only if it ended a word: the first kept is the best so.
  let last = keep(std::mem::take(&mut buckets[frames]), &nodes, frames, beam);
  let mut out = Vec::new();
  let (mut p, mut frames) = last.first().map_or((0, None), |h| (h.prefix, h.frames.clone()));
  while p != 0 {
    let f = frames.expect("a frame per token");
    out.push((nodes[p as usize].token, f.frame));
    (p, frames) = (nodes[p as usize].parent, f.before.clone());
  }
  out.reverse();
  let tokens: Vec<u32> = out.iter().map(|o| o.0).collect();
  lone_spaces(&tokens, texts).map(|i| out[i]).collect()
}

/// The index of the first maximum.
fn argmax(v: &[f32]) -> usize {
  let mut best = 0;
  for (i, &x) in v.iter().enumerate() {
    if x > v[best] {
      best = i;
    }
  }
  best
}

/// `logits` as log probabilities.
pub(crate) fn log_softmax(logits: &mut [f32]) {
  let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
  let sum: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
  let shift = max + sum.ln();
  logits.iter_mut().for_each(|x| *x -= shift);
}

/// The first frame of each token's run on the likeliest CTC alignment of `tokens` to `logits`
/// (Viterbi over blank, token, blank, …; a row's log-softmax only shifts every path alike).
fn align(logits: &[f32], classes: usize, blank: usize, tokens: &[u32]) -> Vec<u32> {
  let (frames, states) = (logits.len() / classes, 2 * tokens.len() + 1);
  if tokens.is_empty() {
    return Vec::new();
  }
  let label = |s: usize| if s.is_multiple_of(2) { blank } else { tokens[s / 2] as usize };
  // How far back each state came from: 0, 1 or 2 states.
  let mut back = vec![0u8; frames * states];
  let mut score = vec![f32::NEG_INFINITY; states];
  score[0] = logits[blank];
  score[1] = logits[label(1)];
  let mut prev = score.clone();
  for t in 1..frames {
    std::mem::swap(&mut prev, &mut score);
    let row = &logits[t * classes..][..classes];
    for s in 0..states {
      let mut best = (prev[s], 0u8);
      if s >= 1 && prev[s - 1] > best.0 {
        best = (prev[s - 1], 1);
      }
      if s >= 2 && s % 2 == 1 && label(s) != label(s - 2) && prev[s - 2] > best.0 {
        best = (prev[s - 2], 2);
      }
      score[s] = best.0 + row[label(s)];
      back[t * states + s] = best.1;
    }
  }
  let mut s =
    if states > 1 && score[states - 2] > score[states - 1] { states - 2 } else { states - 1 };
  let mut out = vec![0u32; tokens.len()];
  for t in (0..frames).rev() {
    if s % 2 == 1 {
      out[s / 2] = t as u32;
    }
    if t > 0 {
      s -= back[t * states + s] as usize;
    }
  }
  out
}

#[cfg(test)]
mod tests {
  use super::*;

  /// A small BPE vocabulary: id 0 is the blank.
  const VOCAB: &[&str] = &[
    "<blk>",
    "▁при",
    "вет",
    "▁мир",
    "▁док",
    "ер",
    "ор",
    "тор",
    "а",
    "ом",
    "▁куб",
    "ерн",
    "етес",
    "итис",
    "▁гига",
    "▁ам",
    "ам",
  ];

  fn texts() -> Vec<Vec<char>> {
    VOCAB.iter().map(|p| p.chars().collect()).collect()
  }

  fn id(piece: &str) -> usize {
    VOCAB.iter().position(|p| *p == piece).unwrap()
  }

  /// One row per frame: the given pieces at their log probabilities, the rest far below.
  fn logits(frames: &[&[(&str, f32)]]) -> Vec<f32> {
    let mut out = vec![-20.0f32; frames.len() * VOCAB.len()];
    for (t, row) in frames.iter().enumerate() {
      for &(piece, lp) in *row {
        out[t * VOCAB.len() + id(piece)] = lp;
      }
    }
    out
  }

  fn decode(logits: &[f32], words: &[&str], boost: f32) -> String {
    let words: Vec<String> = words.iter().map(|w| w.to_string()).collect();
    let ids = ctc(logits, VOCAB.len(), 0, &texts(), &Trie::new(&words), boost, 8);
    ids
      .iter()
      .map(|&(id, _)| VOCAB[id as usize])
      .collect::<String>()
      .replace('▁', " ")
      .trim()
      .to_string()
  }

  fn greedy(logits: &[f32]) -> String {
    let ids = crate::gigaam::decoder::ctc(logits, VOCAB.len(), 0);
    ids
      .iter()
      .map(|&(id, _)| VOCAB[id as usize])
      .collect::<String>()
      .replace('▁', " ")
      .trim()
      .to_string()
  }

  #[test]
  fn a_near_miss_becomes_the_hotword() {
    // "докор" is a little likelier than "докер".
    let l = logits(&[&[("▁док", -0.1)], &[("<blk>", -0.1)], &[("ор", -0.5), ("ер", -1.0)]]);
    assert_eq!(decode(&l, &[], 1.0), "докор");
    assert_eq!(decode(&l, &["докер"], 0.0), "докор");
    assert_eq!(decode(&l, &["Докер"], 1.0), "докер");

    let l = logits(&[&[("▁куб", -0.1)], &[("ерн", -0.1)], &[("итис", -0.3), ("етес", -1.2)]]);
    assert_eq!(decode(&l, &[], 1.0), "кубернитис");
    assert_eq!(decode(&l, &["кубернетес"], 1.0), "кубернетес");
  }

  #[test]
  fn an_inflected_word_keeps_the_bonus() {
    let l = logits(&[&[("▁док", -0.1)], &[("ор", -0.5), ("ер", -1.0)], &[("ом", -0.1)]]);
    assert_eq!(decode(&l, &["докер"], 1.0), "докером");
  }

  #[test]
  fn a_word_that_turns_off_gets_no_bonus() {
    // "доктор" shares "док" with the hotword; the bonus for "док" is taken back.
    let l = logits(&[&[("▁док", -0.1)], &[("тор", -0.2), ("ер", -8.0)]]);
    assert_eq!(decode(&l, &["докер"], 1.0), "доктор");
    // And the beam doesn't stop on a hotword's prefix: "док" alone isn't a word end.
    let l = logits(&[&[("▁док", -0.1)], &[("<blk>", -0.1)]]);
    assert_eq!(decode(&l, &["докер"], 1.0), "док");
  }

  #[test]
  fn an_unfinished_word_drops_no_words_after_it() {
    // "▁док" starts the hotword, the next letters don't follow it, and blanks are close behind
    // each token: a hypothesis holding "док"'s partial bonus over blanks would drop "ор при мир".
    let l = logits(&[
      &[("▁док", -0.1)],
      &[("ор", -0.1), ("<blk>", -1.0)],
      &[("▁при", -0.1), ("<blk>", -1.0)],
      &[("вет", -0.1), ("<blk>", -1.0)],
      &[("▁мир", -0.1), ("<blk>", -1.0)],
      &[("<blk>", -0.1)],
    ]);
    let words = vec!["докер".to_string()];
    for beam in [1, 2, 8] {
      let ids = ctc(&l, VOCAB.len(), 0, &texts(), &Trie::new(&words), 1.5, beam);
      let text: String = ids.iter().map(|&(id, _)| VOCAB[id as usize]).collect();
      assert_eq!(text.replace('▁', " ").trim(), "докор привет мир", "beam {beam}");
    }
  }

  #[test]
  fn a_lone_space_before_a_word_goes() {
    let vocab = ["<blk>", "▁", "▁до", "к", "ер", "D"];
    let texts: Vec<Vec<char>> = vocab.iter().map(|p| p.chars().collect()).collect();
    let mut l = vec![-20.0f32; 4 * vocab.len()];
    for (t, row) in
      [&[(1, -0.1)][..], &[(5, -0.5), (2, -1.0)], &[(3, -0.1)], &[(4, -0.1)]].iter().enumerate()
    {
      row.iter().for_each(|&(id, lp)| l[t * vocab.len() + id] = lp);
    }
    let trie = Trie::new(&["докер".to_string()]);
    let ids: Vec<u32> =
      ctc(&l, vocab.len(), 0, &texts, &trie, 1.5, 8).iter().map(|t| t.0).collect();
    assert_eq!(ids, [2, 3, 4]);
  }

  /// A frame's scripted scores, given the frame and the last token.
  type Row = fn(usize, Option<usize>) -> Vec<(&'static str, f32)>;

  /// A scripted transducer over [`VOCAB`]: its prediction is the last token, and `script(t, last)`
  /// gives that frame's scores `(token, score)`, the rest at -20 (blank's id 0 included).
  struct Script {
    script: Row,
    durations: usize,
  }

  impl Transducer for Script {
    type State = Option<usize>;

    fn start(&self) -> (Option<usize>, Vec<f32>) {
      (None, vec![-1.0])
    }

    fn step(&self, _: &Option<usize>, token: usize) -> (Option<usize>, Vec<f32>) {
      (Some(token), vec![token as f32])
    }

    fn joint(&self, t: usize, preds: &[&[f32]]) -> Vec<(Vec<f32>, Vec<f32>)> {
      (preds.iter())
        .map(|p| {
          let last = (p[0] >= 0.0).then_some(p[0] as usize);
          let mut row = vec![-20.0; VOCAB.len()];
          for (piece, score) in (self.script)(t, last) {
            row[VOCAB.iter().position(|v| *v == piece).unwrap()] = score;
          }
          log_softmax(&mut row);
          // TDT: every decision moves on one frame.
          let mut durations: Vec<f32> =
            (0..self.durations).map(|d| if d == 1 { 0.0 } else { -9.0 }).collect();
          log_softmax(&mut durations);
          (row, durations)
        })
        .collect()
    }
  }

  fn run(script: &Script, frames: usize, words: &[&str]) -> String {
    let words: Vec<String> = words.iter().map(|w| w.to_string()).collect();
    let durations = [0, 1, 2];
    let steps = Steps { blank: 0, durations: &durations[..script.durations], max_symbols: 10 };
    let ids = transducer(script, frames, &steps, &texts(), &Trie::new(&words), 1.5, 8);
    ids
      .iter()
      .map(|&(id, _)| VOCAB[id as usize])
      .collect::<String>()
      .replace('▁', " ")
      .trim()
      .to_string()
  }

  /// Frame 0 `▁док`, frame 1 `тор` (`ер` 1.5 below it), frame 2 `▁мир` (`▁куб` 3 below it),
  /// each after the one before; else the blank. RNN-T scores a frame again after its token, and
  /// gets the blank then; TDT moves on one frame with each decision.
  fn doctor(t: usize, last: Option<usize>) -> Vec<(&'static str, f32)> {
    match (t, last.map(|l| VOCAB[l])) {
      (0, None) => vec![("▁док", 0.0), ("<blk>", -3.0)],
      (1, Some("▁док")) => vec![("тор", 0.0), ("ер", -1.5), ("<blk>", -3.0)],
      (2, Some("тор" | "ер")) => vec![("▁мир", 0.0), ("▁куб", -3.0), ("<blk>", -4.0)],
      _ => vec![("<blk>", 0.0)],
    }
  }

  #[test]
  fn a_transducer_finds_the_word_and_otherwise_decodes_as_greedy() {
    for durations in [0, 3] {
      let model = Script { script: doctor, durations };
      assert_eq!(run(&model, 4, &[]), "доктор мир", "{durations} durations");
      assert_eq!(run(&model, 4, &["докер"]), "докер мир", "{durations} durations");
      // A word in reach but not said (`▁куб` of `кубок`), and one out of reach, change nothing.
      assert_eq!(run(&model, 4, &["кубок", "паровоз"]), "доктор мир", "{durations} durations");
    }
  }

  #[test]
  fn a_transducer_spells_no_word_the_audio_doesnt_have() {
    // `ер` 7.3 below `тор`: the whole word's 7.5 of bonus would pay for it, but its own 3
    // leave it 4.3 short, past `REACH`, so `доктор` stays: a word is heard letter by letter.
    fn far(t: usize, last: Option<usize>) -> Vec<(&'static str, f32)> {
      match (t, last.map(|l| VOCAB[l])) {
        (1, Some("▁док")) => vec![("тор", 0.0), ("ер", -7.3), ("<blk>", -9.0)],
        _ => doctor(t, last),
      }
    }
    assert_eq!(run(&Script { script: far, durations: 0 }, 4, &["докер"]), "доктор мир");
  }

  #[test]
  fn a_word_after_a_phrase_that_turned_off_is_no_case_ending() {
    let trie = Trie::new(&["докер".to_string(), "докер компоуз".to_string()]);
    let bonus = |text: &str| trie.advance(Ctx::START, &text.chars().collect::<Vec<_>>(), 1.0).1;
    assert_eq!(bonus("докер▁образ▁"), 5.0);
    assert_eq!(bonus("докер▁обз▁"), 5.0);
    assert_eq!(bonus("докер▁кот▁"), 5.0);
    assert_eq!(bonus("докер▁компоуз▁"), 12.0);
    assert_eq!(bonus("докер▁докер▁"), 10.0);
  }

  #[test]
  fn a_phrase_matches_spaced_and_run_together() {
    let l = logits(&[&[("▁гига", -0.1)], &[("ам", -0.9), ("▁ам", -1.2), ("<blk>", -0.6)]]);
    assert_eq!(decode(&l, &[], 1.0), "гига");
    assert_eq!(decode(&l, &["гига ам"], 1.0), "гигаам");
  }

  #[test]
  fn without_a_match_the_beam_is_greedy() {
    // Peaked rows, as a model's mostly are: repeats collapse, blanks split them.
    let l = logits(&[
      &[("▁при", -0.1), ("▁мир", -2.0)],
      &[("▁при", -0.2), ("<blk>", -1.0)],
      &[("<blk>", -0.1)],
      &[("вет", -0.1), ("ер", -3.0)],
      &[("<blk>", -0.3), ("вет", -1.5)],
      &[("▁мир", -0.1)],
      &[("▁мир", -0.4), ("<blk>", -1.2)],
      &[("<blk>", -0.1)],
      &[("▁мир", -0.1)],
    ]);
    assert_eq!(greedy(&l), "привет мир мир");
    assert_eq!(decode(&l, &[], 1.0), greedy(&l));
    assert_eq!(decode(&l, &["кубернетес"], 1.0), greedy(&l));
  }

  #[test]
  fn frames_are_where_tokens_start() {
    let l = logits(&[&[("<blk>", -0.1)], &[("▁док", -0.1)], &[("▁док", -0.1)], &[("ер", -0.1)]]);
    let words = vec!["докер".to_string()];
    let ids = ctc(&l, VOCAB.len(), 0, &texts(), &Trie::new(&words), 1.0, 8);
    assert_eq!(ids, [(id("▁док") as u32, 1), (id("ер") as u32, 3)]);
  }

  #[test]
  fn a_character_vocabulary_matches_too() {
    // GigaAM `ctc`: one letter per token, a space token between words.
    let vocab = ["<blk>", " ", "д", "о", "к", "е", "р", "а"];
    let texts: Vec<Vec<char>> = vocab.iter().map(|p| p.chars().collect()).collect();
    let mut l = vec![-20.0f32; 6 * vocab.len()];
    for (t, row) in [
      &[(2, -0.1)][..],
      &[(3, -0.1)],
      &[(4, -0.1)],
      &[(3, -0.4), (5, -0.9)],
      &[(6, -0.1)],
      &[(1, -0.1)],
    ]
    .iter()
    .enumerate()
    {
      row.iter().for_each(|&(id, lp)| l[t * vocab.len() + id] = lp);
    }
    let spell =
      |ids: Vec<(u32, u32)>| ids.iter().map(|&(id, _)| vocab[id as usize]).collect::<String>();
    let none = Trie::new(&[]);
    assert_eq!(spell(ctc(&l, vocab.len(), 0, &texts, &none, 1.5, 8)), "докор ");
    let trie = Trie::new(&["докер".to_string()]);
    assert_eq!(spell(ctc(&l, vocab.len(), 0, &texts, &trie, 1.5, 8)), "докер ");
  }

  #[test]
  fn stems_endings_and_empty_words() {
    let trie = Trie::new(&["".to_string(), " , ".to_string()]);
    assert!(trie.is_empty());
    let trie = Trie::new(&["Москва".to_string()]);
    let word = |w: &str| {
      let chars: Vec<char> = w.chars().collect();
      let (ctx, delta) = trie.advance(Ctx::START, &chars, 1.0);
      delta - ctx.open()
    };
    assert_eq!(word("москв"), 5.0);
    assert_eq!(word("москвы"), 5.0);
    assert_eq!(word("моск"), 0.0);
    assert_eq!(word("МОСКВА"), 6.0);
    let trie = Trie::new(&["докер".to_string()]);
    let word = |w: &str| {
      let chars: Vec<char> = w.chars().collect();
      let (ctx, delta) = trie.advance(Ctx::START, &chars, 1.0);
      delta - ctx.open()
    };
    assert_eq!(word("докерами"), 5.0);
    assert_eq!(word("докерамиии"), 0.0);
    assert_eq!(word("докерbraz"), 0.0);
    assert_eq!(word("доктор"), 0.0);
  }

  /// `cargo test --release --lib hotwords::tests::speed -- --ignored --nocapture`, also on a phone:
  /// 25 s of GigaAM-sized random logits (no row peaked, the worst case) with and without the
  /// words, the fastest of 5.
  #[test]
  #[ignore]
  fn speed() {
    let (frames, classes) = (625, 1025);
    let mut seed = 1u64;
    let logits: Vec<f32> = (0..frames * classes)
      .map(|_| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 40) as f32 / (1u64 << 24) as f32 * 8.0
      })
      .collect();
    let texts: Vec<Vec<char>> =
      (0..classes).map(|i| format!("▁д{}", i % 7).chars().collect()).collect();
    let words: Vec<String> = ["кубернетес", "докер", "эс эс эйч", "GigaAM", "паракит"]
      .iter()
      .map(|w| w.to_string())
      .collect();
    let trie = Trie::new(&words);
    let best = |f: &dyn Fn()| {
      (0..5)
        .map(|_| {
          let t = std::time::Instant::now();
          f();
          t.elapsed().as_secs_f64() * 1000.0
        })
        .fold(f64::MAX, f64::min)
    };
    let greedy =
      best(&|| drop(std::hint::black_box(crate::gigaam::decoder::ctc(&logits, classes, 0))));
    let beam =
      best(&|| drop(std::hint::black_box(ctc(&logits, classes, 0, &texts, &trie, 1.5, 8))));
    eprintln!(
      "25 s of logits: greedy {greedy:.2} ms, hotwords {beam:.2} ms, +{:.3} ms per second",
      (beam - greedy) / 25.0
    );
  }
}
