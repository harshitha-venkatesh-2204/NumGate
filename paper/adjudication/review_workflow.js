export const meta = {
  name: 'numgate-blind-review',
  description: 'Blind review of gate flags and sampled numbers: two independent reviewers per batch, a third on disagreements',
  phases: [
    { title: 'Review', detail: 'two independent reviewers per batch' },
    { title: 'Resolve', detail: 'a third reviewer for disagreements' },
  ],
}

// Blind review of the full study, run as a Claude Code workflow: two independent reviewers per batch of 60 items
// (one forward, one in reverse order) and a third reviewer for disagreements. args = {root, batches: [item files]}.
// Its output is turned into verdicts_<run>.csv and reviews_<run>.csv by collect_verdicts.py.

const ROOT = args.root
const GUIDE = `${ROOT}/paper/adjudication/REVIEW_GUIDE.md`
const VERDICTS = ['correct', 'wrong', 'unclear', 'not_a_measurement']
const REVIEW = {
  type: 'object',
  properties: {
    items: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          item_id: { type: 'string' },
          verdict: { type: 'string', enum: VERDICTS },
          reading: { type: 'string', description: 'what the sentence says the number is' },
          expected: { type: 'string', description: 'what the table gives for that reading, with the computation for anything derived' },
          reason: { type: 'string' },
        },
        required: ['item_id', 'verdict', 'reading', 'expected', 'reason'],
      },
    },
  },
  required: ['items'],
}
const FINAL = {
  type: 'object',
  properties: {
    items: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          item_id: { type: 'string' },
          verdict: { type: 'string', enum: VERDICTS },
          reason: { type: 'string' },
        },
        required: ['item_id', 'verdict', 'reason'],
      },
    },
  },
  required: ['items'],
}

function reviewPrompt(batch, order) {
  return `You are reviewing numbers in weekly retail sales notes against their source tables. Work in ${ROOT}.

Read the reviewer guide first: ${GUIDE}. Follow it exactly.
Then review every item in ${batch}${order === 'reverse' ? ', working through the list from the LAST item to the first' : ''}.
Each item gives the note's lines, the line number, the marked number and the table CSV path (relative to ${ROOT}).
Use Python with pandas to read each table and compute values; do not estimate in your head. Open only the item file
and the table CSVs it names: do not open anything under runs/, results/, src/ or paper/ other than these, and do not
try to find out which model or method wrote a note or what any automatic checker said about it.

Return one entry per item in the file (same item_id values), with verdict, reading, expected and reason.`
}

function resolvePrompt(batch, items) {
  return `You are the third reviewer for disputed numbers in weekly retail sales notes. Work in ${ROOT}.

Read the reviewer guide first: ${GUIDE}. The items are in ${batch}; only the item_ids listed below are disputed.
For each, two earlier reviewers disagreed. Their readings and computations are below, labelled Review 1 and Review 2
(they are not ranked). Re-derive the answer yourself from the table with Python and pandas, decide which reading the
sentence supports, and give the final verdict under the guide's rules. Do not open anything under runs/, results/ or
src/, and do not try to find out which model or checker produced anything.

${JSON.stringify(items, null, 1)}

Return one entry per disputed item_id.`
}

phase('Review')
const results = await pipeline(args.batches,
  (batch, _, i) => parallel([
    () => agent(reviewPrompt(batch, 'forward'), { label: `review A ${i + 1}`, phase: 'Review', schema: REVIEW }),
    () => agent(reviewPrompt(batch, 'reverse'), { label: `review B ${i + 1}`, phase: 'Review', schema: REVIEW }),
  ]),
  async (pair, batch, i) => {
    const [a, b] = pair
    if (!a || !b) return { batch, error: 'a reviewer failed', a, b }
    const mb = new Map(b.items.map(x => [x.item_id, x]))
    const disputes = []
    for (const x of a.items) {
      const y = mb.get(x.item_id)
      if (!y || y.verdict !== x.verdict) disputes.push({ item_id: x.item_id, review_1: x, review_2: y || null })
    }
    let third = null
    if (disputes.length) {
      third = await agent(resolvePrompt(batch, disputes), { label: `resolve ${i + 1} (${disputes.length})`, phase: 'Resolve', schema: FINAL, effort: 'high' })
    }
    return { batch, a: a.items, b: b.items, third: third ? third.items : [] }
  },
)
const done = results.filter(Boolean)
log(`${done.length} of ${args.batches.length} batches reviewed`)
return { results: done }
