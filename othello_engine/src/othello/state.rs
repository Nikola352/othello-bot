use crate::othello::board::{OthelloBoard, Piece, Square};
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

    pub fn get_legal_moves(&self) -> Vec<Square> {
        let mut moves: Vec<Square> = Vec::new();

        for i in 0..crate::othello::board::BOARD_SIZE {
            moves.extend(self.get_moves_in_dir((i, 0), (0, 1)));
            moves.extend(self.get_moves_in_dir((0, i), (1, 0)));
            moves.extend(self.get_moves_in_dir((i, 0), (1, 1)));
            moves.extend(self.get_moves_in_dir((0, i), (1, 1)));
            moves.extend(self.get_moves_in_dir((0, i), (1, -1)));
            moves
                .extend(self.get_moves_in_dir((i, crate::othello::board::BOARD_SIZE - 1), (1, -1)));
            moves
                .extend(self.get_moves_in_dir((i, crate::othello::board::BOARD_SIZE - 1), (0, -1)));
            moves
                .extend(self.get_moves_in_dir((crate::othello::board::BOARD_SIZE - 1, i), (-1, 0)));
            moves.extend(
                self.get_moves_in_dir((i, crate::othello::board::BOARD_SIZE - 1), (-1, -1)),
            );
            moves.extend(
                self.get_moves_in_dir((crate::othello::board::BOARD_SIZE - 1, i), (-1, -1)),
            );
            moves
                .extend(self.get_moves_in_dir((crate::othello::board::BOARD_SIZE - 1, i), (-1, 1)));
            moves.extend(self.get_moves_in_dir((i, 0), (-1, 1)));
        }

        moves
    }

    /// Get all possible moves on a straight line starting at (row, col) in the direction of delta
    fn get_moves_in_dir(&self, start: (u8, u8), delta: (i8, i8)) -> Vec<Square> {
        let mut moves: Vec<Square> = Vec::new();
        let mut seen_my_before = false;
        let mut seen_opponent = false;
        let mut row = start.0 as i16;
        let mut col = start.1 as i16;
        while OthelloBoard::is_valid_position(row, col) {
            match &self.board.get(&Square::from_int(row, col)) {
                Piece::Occupied(col) if *col == self.turn => {
                    seen_my_before = true;
                }
                Piece::Occupied(_) => {
                    if seen_my_before {
                        seen_opponent = true;
                    }
                }
                Piece::Empty => {
                    if seen_opponent {
                        moves.push(Square::from_int(row, col))
                    }
                    seen_my_before = false;
                    seen_opponent = false;
                }
            };
            row += delta.0 as i16;
            col += delta.1 as i16;
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
}
