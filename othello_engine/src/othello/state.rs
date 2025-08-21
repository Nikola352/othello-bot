use crate::othello::board::{OthelloBoard, Square, BOARD_SIZE};
use crate::othello::color::Color;
use crate::othello::color::Color::{Black, White};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameState {
    pub board: OthelloBoard,
    pub turn: Color,
}

impl GameState {
    pub fn new() -> GameState {
        GameState {
            board: OthelloBoard::new(),
            turn: Black,
        }
    }

    pub fn get_legal_moves(&self) -> Vec<(u8, u8)> {
        let mask = self.board.get_legal_moves(&self.turn);

        let mut moves: Vec<(u8, u8)> = Vec::new();

        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                if mask & (1u64 << row*BOARD_SIZE + col) != 0 {
                    moves.push((row, col))
                }
            }
        }

        moves
    }

    pub fn play_move(&self, square: &Square) -> GameState {
        GameState {
            board: self.board.with_move(square, &self.turn),
            turn: self.turn.opponent(),
        }
    }

    pub fn skip_turn(&self) -> GameState {
        GameState {
            board: self.board.clone(),
            turn: self.turn.opponent(),
        }
    }

    pub fn get_score(&self) -> i32 {
        self.board.get_score()
    }

    pub fn is_final(&self) -> bool {
        self.get_legal_moves().is_empty() && self.skip_turn().get_legal_moves().is_empty()
    }

    pub fn get_winner(&self) -> Option<Color> {
        if !self.is_final() {
            return None;
        }

        match self.get_score() {
            score if score > 0 => Some(Black),
            score if score < 0 => Some(White),
            _ => None,
        }
    }

    pub fn get_legal_moves_mask(&self) -> Vec<Vec<f32>> {
        let mask = self.board.get_legal_moves(&self.turn);
        Self::bitmask_to_matrix(mask)
    }

    fn bitmask_to_matrix(mask: u64) -> Vec<Vec<f32>> {
        let mut board = vec![vec![0f32; BOARD_SIZE as usize]; BOARD_SIZE as usize];

        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                if mask & (1u64 << row*BOARD_SIZE + col) != 0 {
                    board[row as usize][col as usize] = 1f32;
                }
            }
        }

        board
    }
}
