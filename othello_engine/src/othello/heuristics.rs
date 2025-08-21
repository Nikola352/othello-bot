use crate::othello::state::GameState;

impl GameState {
    pub fn get_relative_piece_diff(&self) -> f32 {
        let (black_cnt, white_cnt) = self.board.piece_counts();
        if black_cnt + white_cnt == 0 {
            return 0f32;
        }
        ((black_cnt - white_cnt) as f32) / ((black_cnt + white_cnt) as f32)
    }

    pub fn get_relative_corner_diff(&self) -> f32 {
        let (black_cnt, white_cnt) = self.board.get_corner_counts();
        if black_cnt + white_cnt == 0 {
            return 0f32;
        }
        ((black_cnt - white_cnt) as f32) / ((black_cnt + white_cnt) as f32)
    }

    pub fn get_relative_stability_diff(&self) -> f32 {
        let (black, white) = self.board.get_stable_by_color();
        let (black_cnt, white_cnt) = (black.count_ones(), white.count_ones());
        if black_cnt + white_cnt == 0 {
            return 0f32;
        }
        ((black_cnt - white_cnt) as f32) / ((black_cnt + white_cnt) as f32)
    }
}