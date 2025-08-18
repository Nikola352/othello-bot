use crate::othello::board::Piece::{Empty, Occupied};
use crate::othello::color::Color;
use crate::othello::color::Color::{Black, White};
use std::fmt;
use std::fmt::Formatter;

pub(crate) const BOARD_SIZE: u8 = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Square {
    pub row: u8,
    pub col: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Piece {
    Empty,
    Occupied(Color),
}

/// Bitboard representation of an 8x8 Othello board.
#[derive(Clone, PartialEq, Eq)]
pub struct OthelloBoard {
    black: u64,
    white: u64,
}

impl OthelloBoard {
    /// Create a new Othello board with default initial setup.
    pub fn new() -> OthelloBoard {
        OthelloBoard {
            black: (1u64 << (3 * 8 + 4)) + (1u64 << (4 * 8 + 3)),
            white: (1u64 << (3 * 8 + 3)) + (1u64 << (4 * 8 + 4)),
        }
    }

    /// Get a piece on a specified square
    pub fn get(&self, square: &Square) -> Piece {
        let index = square.row * BOARD_SIZE + square.col;
        if (self.black >> index) & 1 == 1 {
            Occupied(Black)
        } else if (self.white >> index) & 1 == 1 {
            Occupied(White)
        } else {
            Empty
        }
    }

    /// Set a piece on a square. Updates only one square, does not make a real move.
    pub fn set(&mut self, square: &Square, piece: Piece) {
        let index = square.row * BOARD_SIZE + square.col;
        match piece {
            Empty => {
                self.black &= !(1u64 << index);
                self.white &= !(1u64 << index);
            }
            Occupied(color) => match color {
                Black => {
                    self.black |= 1u64 << index;
                    self.white &= !(1u64 << index);
                }
                White => {
                    self.white |= 1u64 << index;
                    self.black &= !(1u64 << index);
                }
            },
        }
    }

    /// Execute a move: place a piece and flip all affected pieces.
    /// This is a low-level operation that assumes the move is legal.
    pub fn play_move(&mut self, square: &Square, turn: &Color) {
        const DIRS: [(i16, i16); 8] = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ];

        self.set(square, Piece::from(turn));

        for dir in &DIRS {
            let flips = self.get_flips_in_direction(square, *dir, turn);
            for flip_sq in flips {
                self.set(&flip_sq, Piece::from(turn))
            }
        }
    }

    /// Get all pieces that would be flipped in a specific direction for a given move
    fn get_flips_in_direction(
        &self,
        square: &Square,
        (dr, dc): (i16, i16),
        color: &Color,
    ) -> Vec<Square> {
        let mut flips = Vec::new();
        let mut row = square.row as i16 + dr;
        let mut col = square.col as i16 + dc;

        // Collect potential flips
        while Self::is_valid_position(row, col) {
            let current_square = Square::from_int(row, col);
            match self.get(&current_square) {
                Occupied(piece_color) if piece_color != *color => {
                    flips.push(current_square);
                }
                Occupied(_) => {
                    // Found our own piece - this direction is valid
                    return flips;
                }
                Empty => {
                    // Hit empty space - this direction is invalid
                    break;
                }
            }
            row += dr;
            col += dc;
        }

        Vec::new()
    }

    pub fn with_move(&self, square: &Square, turn: &Color) -> OthelloBoard {
        let mut board = self.clone();
        board.play_move(square, turn);
        board
    }

    /// Get count of pieces for black and white
    pub fn piece_counts(&self) -> (u32, u32) {
        (self.black.count_ones(), self.white.count_ones())
    }

    /// Get the score (positive for black advantage, negative for white advantage)
    pub fn get_score(&self) -> i32 {
        let (black_count, white_count) = self.piece_counts();
        black_count as i32 - white_count as i32
    }

    pub fn is_valid_position(row: i16, col: i16) -> bool {
        row >= 0 && row < BOARD_SIZE as i16 && col >= 0 && col < BOARD_SIZE as i16
    }
}

impl Square {
    pub fn new(row: u8, col: u8) -> Square {
        Square { row, col }
    }

    /// Create Square from i16 coordinates (with bounds checking in debug mode)
    pub fn from_int(row: i16, col: i16) -> Square {
        debug_assert!(row >= 0 && row < BOARD_SIZE as i16);
        debug_assert!(col >= 0 && col < BOARD_SIZE as i16);
        Square {
            row: row as u8,
            col: col as u8,
        }
    }

    pub fn row(&self) -> u8 {
        self.row
    }

    pub fn col(&self) -> u8 {
        self.col
    }
}

impl From<&Color> for Piece {
    fn from(color: &Color) -> Self {
        Occupied(color.clone())
    }
}

impl Default for OthelloBoard {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for OthelloBoard {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "  a b c d e f g h")?;

        for row in 0..BOARD_SIZE {
            write!(f, "{} ", row + 1)?;

            for col in 0..BOARD_SIZE {
                let pos = 1u64 << (row * BOARD_SIZE + col);
                let c = if (self.black & pos) != 0 {
                    'B'
                } else if (self.white & pos) != 0 {
                    'W'
                } else {
                    '.'
                };
                write!(f, "{} ", c)?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}
