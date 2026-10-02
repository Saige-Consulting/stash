package brain

import "testing"

// nextEpisodeCheckpoint is the whole of the stage-1 checkpoint rule: advance
// over every episode whose outcome is final, stop just before the first one
// that failed transiently, and never move backwards.
func TestNextEpisodeCheckpoint(t *testing.T) {
	cases := []struct {
		name string
		prev int64
		ids  []int64
		done map[int64]bool
		want int64
	}{
		{"all done advances to the last id", 10, []int64{11, 12, 15}, map[int64]bool{11: true, 12: true, 15: true}, 15},
		{"stops before the first not-done id", 10, []int64{11, 12, 15}, map[int64]bool{11: true, 15: true}, 11},
		{"first id not done keeps prev", 10, []int64{11, 12, 15}, map[int64]bool{12: true, 15: true}, 10},
		{"empty batch keeps prev", 10, nil, map[int64]bool{}, 10},
		{"nil done map keeps prev", 10, []int64{11}, nil, 10},
		{"gaps in ids are fine", 0, []int64{3, 40, 41}, map[int64]bool{3: true, 40: true, 41: true}, 41},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := nextEpisodeCheckpoint(tc.prev, tc.ids, tc.done); got != tc.want {
				t.Fatalf("nextEpisodeCheckpoint(%d, %v, %v) = %d, want %d", tc.prev, tc.ids, tc.done, got, tc.want)
			}
		})
	}
}

// The batch is fetched with id > prev, so a done id at or below prev can only
// come from a caller bug. Whatever the input, the checkpoint must not regress:
// a regression re-mines every episode between the two values.
func TestNextEpisodeCheckpoint_NeverBelowPrev(t *testing.T) {
	inputs := []struct {
		prev int64
		ids  []int64
		done map[int64]bool
	}{
		{50, []int64{3, 4, 5}, map[int64]bool{3: true, 4: true, 5: true}},
		{50, []int64{3, 4, 5}, map[int64]bool{}},
		{50, []int64{51, 52}, map[int64]bool{52: true}},
		{50, []int64{}, nil},
	}
	for _, in := range inputs {
		if got := nextEpisodeCheckpoint(in.prev, in.ids, in.done); got < in.prev {
			t.Fatalf("nextEpisodeCheckpoint(%d, %v, %v) = %d, below prev", in.prev, in.ids, in.done, got)
		}
	}
}
