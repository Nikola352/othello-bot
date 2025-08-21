use crate::othello::board::Piece::{Empty, Occupied};
use crate::othello::color::Color;
use crate::othello::color::Color::{Black, White};
use crate::othello::mask_shift::Direction::*;
use crate::othello::mask_shift::{shift, Direction, BOTTOM_ROW_MASK, DIRECTIONS, LEFT_COL_MASK, MAIN_DIAGONAL_MASK, MINOR_DIAGONAL_MASK, RIGHT_COL_MASK, TOP_ROW_MASK};
use std::fmt;
use std::fmt::Formatter;

pub const BOARD_SIZE: u8 = 8;

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
        self.set(&square, Piece::from(turn));

        let player_mask = match turn {
            Black => self.black,
            White => self.white,
        };
        let opp_mask = match turn {
            Black => self.white,
            White => self.black,
        };

        let index = square.row * BOARD_SIZE + square.col;
        let new_piece = 1u64 << index;

        let mut flipped_piece_mask = 0u64;

        for dir in DIRECTIONS {
            let mut flipped = shift(new_piece, dir) & opp_mask;
            for _ in 0..BOARD_SIZE - 2 {
                flipped |= shift(flipped, dir) & opp_mask;
            }
            if shift(flipped, dir) & player_mask != 0 {
                flipped_piece_mask |= flipped;
            }
        }

        match turn {
            Black => {
                self.black |= flipped_piece_mask;
                self.white &= !flipped_piece_mask;
            }
            White => {
                self.white |= flipped_piece_mask;
                self.black &= !flipped_piece_mask;
            }
        }
    }

    pub fn get_legal_moves(&self, turn: &Color) -> u64 {
        let player_mask = match turn {
            Black => self.black,
            White => self.white,
        };
        let opp_mask = match turn {
            Black => self.white,
            White => self.black,
        };

        let mut legal_move_mask = 0u64;

        for dir in DIRECTIONS {
            let mut flippable = shift(player_mask, dir) & opp_mask;
            for _ in 0..BOARD_SIZE - 2 {
                flippable |= shift(flippable, dir) & opp_mask;
            }
            legal_move_mask |= shift(flippable, dir) & !(player_mask | opp_mask);
        }

        legal_move_mask
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

    pub fn get_corner_counts(&self) -> (u32, u32) {
        const CORNER_MASK: u64 = 0x8100000000000081;
        let black = (self.black & CORNER_MASK).count_ones();
        let white = (self.white & CORNER_MASK).count_ones();
        (black, white)
    }

    pub fn get_stable_by_color(&self) -> (u64, u64) {
        let mask = self.get_stable_pieces();
        (mask & self.black, mask & self.white)
    }

    pub fn get_stable_pieces(&self) -> u64 {
        let mut stable = 1u64;
        let occupied = self.black | self.white;

        // Occupancy pass
        stable &= Self::stability_sweep(TOP_ROW_MASK, South, occupied);
        stable &= Self::stability_sweep(LEFT_COL_MASK, East, occupied);
        stable &= Self::stability_sweep(MAIN_DIAGONAL_MASK, NorthWest, occupied);
        stable &= Self::stability_sweep(MAIN_DIAGONAL_MASK, SouthEast, occupied);
        stable &= Self::stability_sweep(MINOR_DIAGONAL_MASK, NorthEast, occupied);
        stable &= Self::stability_sweep(MINOR_DIAGONAL_MASK, SouthWest, occupied);

        // Neighbour-based induction loop
        let mut new_stable = stable;
        loop {
            stable = new_stable;

            // Pieces adjacent to stable pieces on all sides are stable (regardless of color)
            let horizontally = occupied
                & (shift(stable, East) | LEFT_COL_MASK)
                & (shift(stable, West) | RIGHT_COL_MASK);
            let vertically = occupied
                & (shift(stable, North) | BOTTOM_ROW_MASK)
                & (shift(stable, South) | TOP_ROW_MASK);
            let diagonally_main = occupied
                & (shift(stable, NorthEast) | LEFT_COL_MASK | BOTTOM_ROW_MASK)
                & (shift(stable, SouthWest) | RIGHT_COL_MASK | TOP_ROW_MASK);
            let diagonally_minor = occupied
                & (shift(stable, NorthWest) | RIGHT_COL_MASK | BOTTOM_ROW_MASK)
                & (shift(stable, SouthEast) | LEFT_COL_MASK | TOP_ROW_MASK);
            new_stable |= horizontally & vertically & diagonally_main & diagonally_minor;

            // Pieces adjacent to a stable piece of the same color is stable
            let horizontally =
                ((shift(stable & self.black, East) | LEFT_COL_MASK)  & self.black) |
                    ((shift(stable & self.black, West) | RIGHT_COL_MASK) & self.black) |
                    ((shift(stable & self.white, East) | LEFT_COL_MASK)  & self.white) |
                    ((shift(stable & self.white, West) | RIGHT_COL_MASK) & self.white);
            let vertically =
                ((shift(stable & self.black, North) | BOTTOM_ROW_MASK)  & self.black) |
                    ((shift(stable & self.black, South) | TOP_ROW_MASK) & self.black) |
                    ((shift(stable & self.white, North) | BOTTOM_ROW_MASK)  & self.white) |
                    ((shift(stable & self.white, South) | TOP_ROW_MASK) & self.white);
            let diagonally_main =
                ((shift(stable & self.black, NorthEast) | LEFT_COL_MASK | BOTTOM_ROW_MASK)  & self.black) |
                    ((shift(stable & self.black, SouthWest) | RIGHT_COL_MASK | TOP_ROW_MASK) & self.black) |
                    ((shift(stable & self.white, NorthEast) | LEFT_COL_MASK | BOTTOM_ROW_MASK)  & self.white) |
                    ((shift(stable & self.white, SouthWest) | RIGHT_COL_MASK | TOP_ROW_MASK) & self.white);
            let diagonally_minor =
                ((shift(stable & self.black, NorthWest) | RIGHT_COL_MASK | BOTTOM_ROW_MASK)  & self.black) |
                    ((shift(stable & self.black, SouthEast) | LEFT_COL_MASK | TOP_ROW_MASK) & self.black) |
                    ((shift(stable & self.white, NorthWest) | RIGHT_COL_MASK | BOTTOM_ROW_MASK)  & self.white) |
                    ((shift(stable & self.white, SouthEast) | LEFT_COL_MASK | TOP_ROW_MASK) & self.white);
            new_stable |= horizontally & vertically & diagonally_main & diagonally_minor;

            if new_stable == stable {
                break;
            }
        }

        stable
    }

    fn stability_sweep(mut mask: u64, direction: Direction, occupied: u64) -> u64 {
        let mut stable = 064;
        while mask != 0u64 {
            if occupied & mask == mask {
                stable |= mask;
            }
            mask = shift(mask, direction);
        }
        stable
    }
}

impl Square {
    pub fn new(row: u8, col: u8) -> Square {
        Square { row, col }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::othello::color::Color::{Black, White};

    fn idx(row: u8, col: u8) -> u64 {
        1u64 << (row * BOARD_SIZE + col)
    }

    #[test]
    fn test_initial_legal_moves_black() {
        let board = OthelloBoard::new();
        let legal = board.get_legal_moves(&Black);

        // Expected: c4, d3, e6, f5
        let expected = idx(2, 3) | idx(3, 2) | idx(4, 5) | idx(5, 4);
        assert_eq!(legal, expected, "Black initial legal moves wrong");
    }

    #[test]
    fn test_initial_legal_moves_white() {
        let board = OthelloBoard::new();
        let legal = board.get_legal_moves(&White);

        // Expected: c5, d6, e3, f4
        let expected = idx(2, 4) | idx(3, 5) | idx(4, 2) | idx(5, 3);
        assert_eq!(legal, expected, "White initial legal moves wrong");
    }

    #[test]
    fn test_play_move_flips_correctly() {
        let mut board = OthelloBoard::new();

        // Black plays f5 (row=4, col=5)
        let move_square = Square::new(4, 5);
        board.play_move(&move_square, &Black);

        // e5 (row=4, col=4) should now be black
        assert_eq!(board.get(&Square::new(4, 4)), Occupied(Black));
        // f5 should be black
        assert_eq!(board.get(&Square::new(4, 5)), Occupied(Black));

        // White's count should have decreased, Black's increased
        let (b, w) = board.piece_counts();
        assert_eq!(b, 4);
        assert_eq!(w, 1);
    }
}
