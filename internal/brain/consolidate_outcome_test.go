package brain

import (
	"context"
	"errors"
	"fmt"
	"net"
	"reflect"
	"strconv"
	"testing"

	"github.com/alash3al/stash/internal/models"
	"github.com/alash3al/stash/internal/reasoner"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgtype"
)

// A failure counts toward giving up on an episode only when it says something
// about the episode. An outage must never count: N attempts over a time floor
// would otherwise turn a long outage into skipped episodes.
func TestCountsAgainstEpisodes(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"plain reasoner error", errors.New("reasoner: no response from LLM"), true},
		{"plain 400", fmt.Errorf("chat.completions call failed: %w", errors.New("400 Bad Request")), true},
		{"provider unavailable", fmt.Errorf("reason structured: %w", reasoner.ErrUnavailable), false},
		{"cancelled pass", fmt.Errorf("insert fact: %w", context.Canceled), false},
		{"timeout", context.DeadlineExceeded, false},
		{"network", &net.OpError{Op: "read", Err: errors.New("connection reset")}, false},
		{"db connect", &pgconn.ConnectError{}, false},
		{"fk violation", &pgconn.PgError{Code: "23503"}, true},
		{"value too long", &pgconn.PgError{Code: "22001"}, true},
		{"db connection failure", &pgconn.PgError{Code: "08006"}, false},
		{"db admin shutdown", &pgconn.PgError{Code: "57P01"}, false},
		{"serialization failure", &pgconn.PgError{Code: "40001"}, false},
		{"too many connections", &pgconn.PgError{Code: "53300"}, false},
	}
	for _, tc := range cases {
		if got := countsAgainstEpisodes(tc.err); got != tc.want {
			t.Errorf("countsAgainstEpisodes(%s) = %v, want %v", tc.name, got, tc.want)
		}
	}
}

// --- readEpisodeBatch over a scripted result set ---

// fakeRow is one fetch_episodes row. scanErr makes Scan fail on it; rawID is
// what RawValues reports for its id column (nil: unrecoverable).
type fakeRow struct {
	id      int64
	mined   bool
	scanErr error
	rawID   []byte
}

// fakeRows mimics pgx: a Scan error is fatal and closes the result set, so
// Next returns false afterwards.
type fakeRows struct {
	rows   []fakeRow
	i      int
	closed bool
	err    error
}

func (f *fakeRows) Close()                        { f.closed = true }
func (f *fakeRows) Err() error                    { return f.err }
func (f *fakeRows) CommandTag() pgconn.CommandTag { return pgconn.CommandTag{} }
func (f *fakeRows) Conn() *pgx.Conn               { return nil }
func (f *fakeRows) Values() ([]any, error)        { return nil, errors.New("fakeRows: Values not supported") }
func (f *fakeRows) FieldDescriptions() []pgconn.FieldDescription {
	return []pgconn.FieldDescription{{Name: "id", DataTypeOID: pgtype.Int8OID, Format: pgtype.TextFormatCode}}
}

func (f *fakeRows) Next() bool {
	if f.closed || f.i >= len(f.rows) {
		f.closed = true
		return false
	}
	f.i++
	return true
}

func (f *fakeRows) cur() fakeRow { return f.rows[f.i-1] }

func (f *fakeRows) RawValues() [][]byte {
	r := f.cur()
	raw := r.rawID
	if raw == nil && r.scanErr == nil {
		raw = []byte(strconv.FormatInt(r.id, 10))
	}
	return [][]byte{raw}
}

func (f *fakeRows) Scan(dest ...any) error {
	r := f.cur()
	if r.scanErr != nil {
		f.err = r.scanErr
		f.closed = true
		return r.scanErr
	}
	*dest[0].(*int64) = r.id
	*dest[2].(*string) = fmt.Sprintf("episode %d", r.id)
	*dest[7].(*bool) = r.mined
	return nil
}

func episodeIDs(eps []models.Episode) []int64 {
	var ids []int64
	for _, e := range eps {
		ids = append(ids, e.ID)
	}
	return ids
}

func TestReadEpisodeBatch_SplitsMinedAndUnmined(t *testing.T) {
	batch, err := readEpisodeBatch(&fakeRows{rows: []fakeRow{{id: 1}, {id: 2, mined: true}, {id: 3}}})
	if err != nil {
		t.Fatalf("readEpisodeBatch: %v", err)
	}
	if want := []int64{1, 2, 3}; !reflect.DeepEqual(batch.ids, want) {
		t.Errorf("ids = %v, want %v", batch.ids, want)
	}
	if want := []int64{1, 3}; !reflect.DeepEqual(episodeIDs(batch.unmined), want) {
		t.Errorf("unmined = %v, want %v", episodeIDs(batch.unmined), want)
	}
	if want := []int64{2}; !reflect.DeepEqual(batch.mined, want) {
		t.Errorf("mined = %v, want %v", batch.mined, want)
	}
}

// A row that will not scan keeps its id: it holds the checkpoint just before
// it and its failure counts toward giving up on it, instead of blocking the
// namespace for good. pgx closes the result set on a scan error, so the rows
// after it wait for the next pass.
func TestReadEpisodeBatch_UnreadableRowKeepsItsID(t *testing.T) {
	boom := errors.New("can't scan into dest[3]")
	batch, err := readEpisodeBatch(&fakeRows{rows: []fakeRow{
		{id: 1}, {id: 2, scanErr: boom, rawID: []byte("2")}, {id: 3},
	}})
	if err != nil {
		t.Fatalf("readEpisodeBatch: %v", err)
	}
	if want := []int64{1, 2}; !reflect.DeepEqual(batch.ids, want) {
		t.Errorf("ids = %v, want %v", batch.ids, want)
	}
	if want := []int64{1}; !reflect.DeepEqual(episodeIDs(batch.unmined), want) {
		t.Errorf("unmined = %v, want %v", episodeIDs(batch.unmined), want)
	}
	if len(batch.unreadable) != 1 || batch.unreadable[0].id != 2 || !errors.Is(batch.unreadable[0].err, boom) {
		t.Errorf("unreadable = %+v, want episode 2 with the scan error", batch.unreadable)
	}
	if batch.cutErr != nil {
		t.Errorf("cutErr = %v, want nil (the id was recovered)", batch.cutErr)
	}
}

// Without an id the row cannot be held or counted. The batch ends before it,
// which is safe: every id in the batch is below it, so the checkpoint cannot
// pass it.
func TestReadEpisodeBatch_UnrecoverableRowEndsTheBatch(t *testing.T) {
	boom := errors.New("can't scan")
	batch, err := readEpisodeBatch(&fakeRows{rows: []fakeRow{{id: 1}, {id: 2, scanErr: boom}, {id: 3}}})
	if err != nil {
		t.Fatalf("readEpisodeBatch: %v", err)
	}
	if want := []int64{1}; !reflect.DeepEqual(batch.ids, want) {
		t.Errorf("ids = %v, want %v", batch.ids, want)
	}
	if !errors.Is(batch.cutErr, boom) {
		t.Errorf("cutErr = %v, want the scan error", batch.cutErr)
	}
	if len(batch.unreadable) != 0 {
		t.Errorf("unreadable = %+v, want none", batch.unreadable)
	}
}
