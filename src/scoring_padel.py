"""
Padel / tennis scoring engine.

Points:  0 → 15 → 30 → 40 → Game
         At 40-40 (deuce): advantage → game, or golden point option
Games:   first to 6 (win by 2); at 6-6 → tiebreak to 7 (win by 2)
Sets:    best of 3 (first to 2 sets wins match)
"""


class PadelMatch:
    _DISP = ['0', '15', '30', '40']

    def __init__(self, first_server: str = 'A', golden_point: bool = False,
                 initial_score: dict = None):
        self.golden_point = golden_point
        self._server = first_server

        # Raw point counts in current game (0-N)
        self._pts = {'A': 0, 'B': 0}
        # Games in current set
        self._games = {'A': 0, 'B': 0}
        # Sets won
        self._sets = {'A': 0, 'B': 0}

        self._in_tiebreak = False
        self._tb_pts = {'A': 0, 'B': 0}
        self._match_over = False
        self._winner = None

        if initial_score:
            self._games['A'] = initial_score.get('A', 0)
            self._games['B'] = initial_score.get('B', 0)

    # ── public ────────────────────────────────────────────────────────────────

    def award(self, winner: str):
        if self._match_over:
            return
        if self._in_tiebreak:
            self._tb_pts[winner] += 1
            a, b = self._tb_pts['A'], self._tb_pts['B']
            if max(a, b) >= 7 and abs(a - b) >= 2:
                self._win_set(winner)
        else:
            self._score_point(winner)

    def snapshot(self) -> dict:
        pa, pb = self._pts['A'], self._pts['B']
        if self._in_tiebreak:
            pts_a = str(self._tb_pts['A'])
            pts_b = str(self._tb_pts['B'])
        else:
            pts_a = self._disp(pa, pb)
            pts_b = self._disp(pb, pa)

        return {
            'a':          self._games['A'],
            'b':          self._games['B'],
            'pts_a':      pts_a,
            'pts_b':      pts_b,
            'sets_a':     self._sets['A'],
            'sets_b':     self._sets['B'],
            'server':     self._server,
            'in_tiebreak': self._in_tiebreak,
            'match_over': self._match_over,
            'winner':     self._winner,
        }

    # ── internals ─────────────────────────────────────────────────────────────

    def _score_point(self, w: str):
        l = 'B' if w == 'A' else 'A'
        pw, pl = self._pts[w], self._pts[l]

        if pw == 4:
            # had advantage → game
            self._win_game(w)
        elif pl == 4:
            # opponent had advantage → back to deuce
            self._pts = {'A': 3, 'B': 3}
        elif pw == 3 and pl == 3:
            # deuce
            if self.golden_point:
                self._win_game(w)
            else:
                self._pts[w] = 4
        elif pw == 3:
            # 40 vs <40 → game
            self._win_game(w)
        else:
            self._pts[w] += 1
            # reaching 40-40 after increment → deuce (handled next award call)

    def _win_game(self, w: str):
        self._pts = {'A': 0, 'B': 0}
        self._games[w] += 1
        self._server = 'B' if self._server == 'A' else 'A'

        ga, gb = self._games['A'], self._games['B']
        if ga == 6 and gb == 6:
            self._in_tiebreak = True
        elif max(ga, gb) >= 6 and abs(ga - gb) >= 2:
            self._win_set(w)
        elif max(ga, gb) >= 7:
            self._win_set(w)

    def _win_set(self, w: str):
        self._in_tiebreak = False
        self._tb_pts = {'A': 0, 'B': 0}
        self._games = {'A': 0, 'B': 0}
        self._sets[w] += 1
        if self._sets[w] >= 2:
            self._match_over = True
            self._winner = w

    def _disp(self, p: int, opp: int) -> str:
        if p <= 2:
            return self._DISP[p]
        if p == 4:
            return 'AD'
        # p == 3
        return '40'
