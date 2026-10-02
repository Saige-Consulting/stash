package brain

// nextEpisodeCheckpoint returns the highest id in ids (ascending, as fetched) such that every
// id up to and including it is done. It never returns less than prev.
//
// "Done" means the episode's stage-1 outcome is final: it yielded a fact, was linked to the
// fact it duplicates, had nothing to extract, was already mined, or produced output that is
// unusable for its text. An episode that failed transiently is not done, and the checkpoint
// stops just before it so the next pass retries it. Episodes after it that are done are not
// lost: they are re-fetched next pass and recognised as already mined, never re-extracted.
//
// This replaces an all-or-nothing rule under which one persistently failing cluster pinned
// the checkpoint and every other episode in the batch was re-mined on every pass.
func nextEpisodeCheckpoint(prev int64, ids []int64, done map[int64]bool) int64 {
	next := prev
	for _, id := range ids {
		if !done[id] {
			break
		}
		if id > next {
			next = id
		}
	}
	return next
}
